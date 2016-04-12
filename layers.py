import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import numpy
from numpy import zeros, ones, sqrt
from numpy.linalg import svd
import theano
from theano import tensor
from theano.sandbox.cuda.dnn import dnn_conv
from theano.sandbox.cuda.dnn import GpuDnnConvDesc, GpuDnnConvGradI
from theano.sandbox.cuda.basic_ops import gpu_contiguous, gpu_alloc_empty

from utils import unroll, _p

profile = False

layers = {'ff': 'FeedForward',
          'post_norm': 'PostNorm',
          'convnet': 'ConvNet',
          'rnn': 'RNN',
          'gru': 'GRU',
          'lstm': 'LSTM',
          'classifier': 'Classifier'}


def get_layer(name):
    fns = layers[name]
    return eval(fns)


def ortho_weight(nin, rng=None):
    floatX = theano.config.floatX
    W = rng.randn(nin, nin)
    u, s, v = svd(W)
    return u.astype(floatX)


def norm_weight(nin, nout=None, scale=0.01, ortho=True, rng=None):
    floatX = theano.config.floatX
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin, rng=rng)
    else:
        W = random_weight(nin, nout, scale, rng=rng)
    return W.astype(floatX)


def random_weight(nin, nout, scale=0.1, rng=None):
    floatX = theano.config.floatX
    W = scale * rng.randn(nin, nout)
    return W.astype(floatX)


def dropout(state_before, suppress_dropout, trng, p=0.5):
    """Dropout

    Parameters
    ----------
    state_before : theano variable
        The layer to apply dropout on
    suppress_dropout : bool
        If True, dropout will be temporary disabled
    trng : theano.sandbox.rng_mrg.MRG_RandomStreams
        A theano random numbers generator
    p : float
        The probability of being dropped out
    """
    return tensor.switch(
        suppress_dropout,
        state_before * p,
        state_before * trng.binomial(
            state_before.shape, p=1-p, n=1, dtype=state_before.dtype))


# def normalization(tparams, state_below, nin, use_noise, options,
#                   prefix='norm'):
#     m_shared = theano.shared(zeros(nin).astype(theano.config.floatX),
#                              borrow=True)
#     s_shared = theano.shared(ones(nin).astype(theano.config.floatX),
#                              borrow=True)
#
#     next_proj0 = state_below - m_shared[None, :]
#     next_proj0 = next_proj0 / tensor.sqrt(1e-5 + s_shared[None, :])
#
#     m = state_below.mean(axis=0)
#     s = state_below.var(axis=0)
#     next_proj1 = state_below - m[None, :]
#     next_proj1 = next_proj1 / tensor.sqrt(1e-5 + s[None, :])
#
#     next_proj = tensor.switch(use_noise, next_proj1, next_proj0)
#
#     return next_proj, m_shared, s_shared, m, s


# ############ ACTIVATIONS ################


def identity(state_below):
    return state_below


def sigmoid(state_below):
    return tensor.nnet.sigmoid(state_below)


def tanh(state_below):
    return tensor.tanh(state_below)


def rectifier(state_below):
    return tensor.switch(state_below > 0, state_below, 0)


def softplus(state_below):
    return tensor.nnet.softplus(state_below)


def softmax(state_below):
    return tensor.nnet.softmax(state_below)

# ############ LAYERS ################


class ALayer(object):
    __metaclass__ = ABCMeta

    children = []
    _params = {}
    initialized = False

    @abstractmethod
    def __init__(self):
        self.__name__ = self.get_prefix()

        try:
            if self.rng is None:
                warnings.warn("No rng has been provided to " +
                              self.get_prefix())
                self.rng = numpy.random.RandomState(0xbeef)
        except AttributeError:
            pass

        try:
            if self.trng is None:
                warnings.warn("No trng has been provided to " +
                              self.get_prefix())
                self.trng = theano.sandbox.rng_mrg.MRG_RandomStreams(0xbeef)
        except AttributeError:
            pass

    @abstractmethod
    def __param_init__(self):
        pass

    def param_init(self):
        if not self.initialized:
            # call object specific param_init
            self.__param_init__()
            # set object params with theano shared
            for (k, v) in self.params.iteritems():
                setattr(self, k, theano.shared(
                    v, name=_p(self.get_prefix(), k), borrow=True))
            # fill params with the theano shared
            self._params = OrderedDict([
                (_p(self.get_prefix(), k), getattr(self, k)) for (k, v) in
                self.params.iteritems() if self.params])

            self.initialized = True
        return self.params

    @abstractmethod
    def build_graph(self):
        pass

    # common methods to all subclasses
    def get_dim(self):
        return self.dim

    def get_nin(self):
        return self.nin

    @property
    def params(self):
        """The current params."""
        return self._params

    def get_prefix(self):
        if self.prefix is '':
            return self.baseprefix
        else:
            return _p(self.baseprefix, self.prefix)

    def get_all_params(self, prev_prefix=None):
        '''Return an OrderedDict with all the parameters.

        Return an OrderedDict with the parameters of self and
        of every child, renamed so that they contain the full
        inclusion path in their name.

        Corresponds to:
        for k, v in self.params.iteritems():
            part = [(k, v)]
            for child in unroll(self.children):
                if child:
                    for k, v in child.get_all_params().iteritems():
                        part += [(_p(self.get_prefix(), k), v)]
        return OrderedDict(part)
        '''
        if prev_prefix:
            self.baseprefix = _p(prev_prefix, self.baseprefix)
        if self.children == []:
            return self.param_init()
        else:
            return OrderedDict([
                (k, v) for k, v in self.param_init().iteritems()] +
                [(k, v) for child in
                 unroll(self.children) if child
                 for k, v in child.get_all_params(
                     self.get_prefix()).iteritems()])

    def _pname(self, name):
        return _p(self.get_prefix(), name)


class FeedForward(ALayer):
    def __init__(self, nin, dim, activ, rng=None, prefix='', baseprefix='ff',
                 mask=None):
        '''A FeedForward layer

        The FeedForward layer will apply an affine transformation and an
        activation function. The activation function can be an Identity,
        in which case the output will be the output of the affine
        transformation, or a non-linearity.

        Parameters
        ----------
        nin : int
            The number of channels of the input
        dim : int
            The number of hidden units of the layer
        activ : string
            The activation function to be used
        rng : numpy.random.randomState
            A numpy random generator
        prefix : string
            Baseprefix_prefix will be used as a name for this object
        baseprefix : string
            Baseprefix_prefix will be used as a name for this object
        mask :
            The
        '''
        self.nin = nin
        self.dim = dim
        self.activ = activ  # options['activ']
        self.rng = rng
        self.prefix = prefix
        self.baseprefix = baseprefix
        self.mask = mask
        super(FeedForward, self).__init__()

    def __param_init__(self):
        nin = self.nin
        dim = self.dim
        rng = self.rng

        W = random_weight(nin, dim, rng=rng)
        b = zeros((dim,)).astype(theano.config.floatX)

        self._params = {'W': W, 'b': b}

    def build_graph(self, state_below):
        return eval(self.activ)(tensor.dot(state_below, self.W) + self.b)


class PostNorm(ALayer):
    def __init__(self, nin, prefix='', baseprefix='post_norm'):
        '''A PostNorm layer

        Parameters
        ----------
        nin : int
            The number of channels of the input
        prefix : string
            Baseprefix_prefix will be used as a name for this object
        baseprefix : string
            Baseprefix_prefix will be used as a name for this object
        '''
        self.nin = nin
        self.prefix = prefix
        self.baseprefix = baseprefix
        super(PostNorm, self).__init__()

    def __param_init__(self):
        nin = self.nin

        m = zeros(nin).astype(theano.config.floatX)
        s = ones(nin).astype(theano.config.floatX)

        self._params = {'m': m, 's': s}

    def build_graph(self, state_below):
        state_below *= self.s[None, :]
        state_below *= self.m[None, :]
        return state_below


class ConvNet(ALayer):
    def __init__(self, nin, nfilters, filter_size, input_shape=None,
                 border_mode='full', stride=(1, 1), activ='identity',
                 init='glorot', rng=None, prefix='', baseprefix='convnet'):
        """A convolutional layer

        A convolutional layer that applies the filters on the image and
        an activation function and returns the resulting feature map. Note that
        the activation function can be an Identity, in which case the
        output of the convolution will be the output of the layer.

        Parameters
        ----------
        nin : int
            The number of channels of the input
        nfilters : int
            The number of filters
        filter_size : list
            The size of the filter expressed as [h, w]
        input_shape : list
            The shape of the input expressed as [batch_size, h, w,
            nchannels]
        border_mode : string
            The border mode of the convolution
        stride : list
            The stride of the convolution
        activ : string
            The activation function
        init : string
            The init function
        rng : numpy.random.randomState
            A numpy random generator
        prefix : string
            Baseprefix_prefix will be used as a name for this object
        baseprefix : string
            Baseprefix_prefix will be used as a name for this object
        """
        self.nin = nin
        self.nfilters = nfilters
        self.filter_size = filter_size
        self.input_shape = input_shape
        self.border_mode = border_mode
        self.stride = stride
        self.activ = activ
        self.init = init
        self.rng = rng
        self.prefix = prefix
        self.baseprefix = baseprefix
        if self.init not in ['he', 'he_normal', 'glorot', 'glorot_normal',
                             'orthonormal']:
            raise NotImplementedError()
        fheight = self.filter_size[0]
        fwidth = self.filter_size[1]
        self.filters_shape = [nfilters, nin, fheight, fwidth]
        super(ConvNet, self).__init__()

    def __param_init__(self):
        nin = self.nin
        nfilters = self.nfilters
        rng = self.rng
        filters_shape = self.filters_shape

        # Kaiming He et al. (2015): Delving deep into rectifiers: Surpassing
        # human-level performance on imagenet classification.
        # arXiv:1502.01852.
        if self.init == 'he':
            init_n = nfilters * numpy.prod(self.filter_size)
            filters = rng.randn(filters_shape).astype(
                theano.config.floatX) * sqrt(2 / init_n)
        # Xavier Glorot and Yoshua Bengio (2010): Understanding the difficulty
        # of training deep feedforward neural networks. International
        # conference on artificial intelligence and statistics.
        elif self.init == 'glorot':
            init_n = sqrt(6. / ((nin + nfilters) * numpy.prod(
                self.filter_size)))
            filters = rng.uniform(
                -init_n, init_n, filters_shape).astype(
                theano.config.floatX)
        # if (self.init in ['he', 'he_normal', 'glorot', 'glorot_normal'] and
        #         self.activ == 'rectifier'):
        #     filters = filters * sqrt(2.0)

        b = zeros((nfilters,)).astype(theano.config.floatX)
        self._params = {'b': b, 'filters': filters}

    # override default method
    def get_dim(self):
        return self.nfilters

    def infer_output_shape(self):
        input_shape = self.input_shape
        filter_size = self.filter_size  # inverted
        stride = self.stride
        if not self.input_shape or any(
                [el is None for el in self.input_shape[1:3]]):
            return [input_shape[0], None, None, self.nfilters]
        out_size = [ConvNet.infer_size(i, f, s, self.border_mode) for i, f, s
                    in zip(input_shape[1:3], filter_size, stride)]
        return [input_shape[0], out_size[0], out_size[1], self.nfilters]

    def build_graph(self, state_below):
        # input_shape = self.input_shape
        # input_shape = state_below.shape[0], input_shape[3], input_shape[1],
        #               input_shape[2]
        # filters_shape = self.filters.shape
        border_mode = self.border_mode
        subsample = self.stride
        activ = self.activ

        if state_below.ndim != 4:
            raise NotImplementedError("The input must be 4D.")

        if border_mode == 'same':
            if self.stride != (1, 1):
                raise NotImplementedError(
                    "'Same' convolution with stride different than (1, 1) "
                    "has not been implemented.")
            border_mode = 'full'
            filter_size = self.filter_size
            [batch_size, nrows, ncolumns, _] = state_below.shape
            s_crop = [(filter_size[0] - 1) // 2,
                      (filter_size[1] - 1) // 2]
            e_crop = [s_crop[0] if (s_crop[0] % 2) != 0 else s_crop[0] + 1,
                      s_crop[1] if (s_crop[1] % 2) != 0 else s_crop[1] + 1]

        state_below = state_below.dimshuffle(0, 3, 1, 2)
        # pred = tensor.nnet.conv2d( --> TODO substitute when new interface
        # will be completed
        # pred = tensor.nnet.abstract_conv2d.conv2d(
        #     state_below, self.filters, input_shape=input_shape,
        #     filter_shape=filters_shape, border_mode=border_mode,
        #     subsample=subsample)
        pred = dnn_conv(
            state_below, self.filters,
            border_mode=border_mode,
            subsample=subsample)
        pred = pred.dimshuffle(0, 2, 3, 1)
        if self.border_mode == 'same':
            pred = pred[:, s_crop[0]:-e_crop[0], s_crop[1]:-e_crop[1], :]
        pred = pred + self.b.dimshuffle('x', 'x', 'x', 0)

        return eval(activ)(pred)

    @staticmethod
    def infer_size(input_dim, filter_dim, stride_dim, border_mode):

        if not input_dim:
            return None
        try:
            if type(input_dim) is list and any(el is None for el in input_dim):
                return [None] * len(input_dim)
        except TypeError:
            pass

        if border_mode == 'full':
            output_dim = (input_dim + filter_dim + stride_dim -
                          2) / stride_dim
        elif border_mode == 'valid':
            output_dim = 1 + (input_dim - filter_dim) / stride_dim
            if type(input_dim) == list:
                if ((input_dim[0] - filter_dim[0]) % stride_dim[0]):
                    print("Valid convolution is cropping some pixels with "
                          "input dim %s on dim 0" % str(input_dim[0]))
                if ((input_dim[1] - filter_dim[1]) % stride_dim[1]):
                    print("Valid convolution is cropping some pixels with "
                          "input dim %s on dim 1" % str(input_dim[1]))
            else:
                if ((input_dim - filter_dim) % stride_dim):
                    print("Valid convolution is cropping some pixels with "
                          "input dim %s" % str(input_dim))

        elif border_mode == 'same':
            output_dim = input_dim
        return output_dim


class DeConvNet(ALayer):
    def __init__(self, nin, nfilters, filter_size, input_shape=None,
                 stride=(1, 1), activ='identity', rng=None, prefix='',
                 baseprefix='deconvnet'):
        """A convolutional layer

        A convolutional layer that exploits the gradient of a
        convolution operator to perform upsampling. The idea is to use
        the gradient in the forward pass, as a way to spread the
        contribution of each element of the input feature map to a patch
        in the output feature map. After the contribution of each
        element of the input feature map to the output feature map
        has been computed, an activation function is applied on the
        resulting feature map. The activation function can be an
        Identity, in which case the output feature map will not be
        modified.

        Parameters
        ----------
        nin : int
            The number of channels of the input
        nfilters : int
            The number of filters
        filter_size : list
            The size of the filter expressed as [h, w]
        input_shape : list
            The shape of the input expressed as [batch_size, h, w,
            nchannels]
        stride : list
            The stride of the convolution
        activ : string
            The activation function
        rng : numpy.random.randomState
            A numpy random generator
        prefix : string
            Baseprefix_prefix will be used as a name for this object
        baseprefix : string
            Baseprefix_prefix will be used as a name for this object
        """
        self.nin = nin
        self.nfilters = nfilters
        self.filter_size = filter_size
        self.input_shape = input_shape
        self.stride = stride
        self.activ = activ
        self.rng = rng
        self.prefix = prefix
        self.baseprefix = baseprefix
        self.border_mode = 'valid'

        fheight = self.filter_size[0]
        fwidth = self.filter_size[1]
        # To use the GpuDnnConvGradI op we should pretend the filter has
        # been used for a convolution. We therefore have to invert nin and
        # nfilters
        # usual filters_shape: [nfilters, nin, fheight, fwidth]
        self.filters_shape = [nin, nfilters, fheight, fwidth]

        super(DeConvNet, self).__init__()

    def __param_init__(self):
        # glorot
        init_n = sqrt(6. / ((self.nin + self.nfilters) * numpy.prod(
            self.filter_size)))
        filters = self.rng.uniform(
            -init_n, init_n, self.filters_shape).astype(
            theano.config.floatX)
        b = numpy.zeros((self.nfilters,),
                        dtype=theano.config.floatX)

        self._params = {'b': b, 'filters': filters}

    # override default method
    def get_dim(self):
        return self.nfilters

    def infer_output_shape(self):
        input_shape = self.input_shape
        filter_size = self.filter_size
        stride = self.stride
        out_size = [DeConvNet.infer_size(i, f, s, self.border_mode) for i, f, s
                    in zip(input_shape[1:3], filter_size, stride)]
        return [input_shape[0], out_size[0], out_size[1], self.nfilters]

    def build_graph(self, state_below):
        filters = self.filters
        nfilters = self.nfilters
        b = self.b
        border_mode = self.border_mode
        # activ = self.activ
        batch_size = state_below.shape[0]

        out_size = DeConvNet.infer_size(state_below.shape[1:3],
                                        filters.shape[2:], self.stride,
                                        self.border_mode)
        out_shape = [batch_size, nfilters, out_size[0], out_size[1]]
        state_below = state_below.dimshuffle(0, 3, 1, 2)

        filters = gpu_contiguous(filters)
        state_below = gpu_contiguous(state_below)
        out_shape = tensor.stack(out_shape)

        desc = GpuDnnConvDesc(border_mode=border_mode, subsample=self.stride,
                              conv_mode='conv')(out_shape, filters.shape)
        pred = GpuDnnConvGradI()(
            filters, state_below, gpu_alloc_empty(*out_shape), desc)
        pred += b.dimshuffle('x', 0, 'x', 'x')
        pred = pred.dimshuffle(0, 2, 3, 1)

        return eval(self.activ)(pred)

    @staticmethod
    def infer_size(input_dim, filter_dim, stride_dim, border_mode):
        if border_mode == 'full':
            output_dim = input_dim * stride_dim - filter_dim - stride_dim + 2
        elif border_mode == 'valid':
            output_dim = (input_dim - 1) * stride_dim + filter_dim
        elif border_mode == 'same':
            output_dim = input_dim
        return output_dim


class RNN(ALayer):
    def __init__(self, nin, dim, rng=None, prefix='', baseprefix='rnn',
                 mask=None):
        '''An RNN layer

        Parameters
        ----------
        nin : int
            The number of channels of the input
        dim : int
            The number of hidden units of the layer
        rng : numpy.random.randomState
            A numpy random generator
        prefix : string
            Baseprefix_prefix will be used as a name for this object
        baseprefix : string
            Baseprefix_prefix will be used as a name for this object
        mask :
            The
        '''
        self.nin = nin
        self.dim = dim
        self.rng = rng
        self.prefix = prefix
        self.baseprefix = baseprefix
        self.mask = mask
        super(RNN, self).__init__()

    def __param_init__(self):
        nin = self.nin
        dim = self.dim
        rng = self.rng

        Wx = norm_weight(nin, dim, rng=rng)
        Ux = ortho_weight(dim, rng=rng)
        bx = zeros((dim,)).astype(theano.config.floatX)

        self._params = {'Wx': Wx, 'Ux': Ux, 'bx': bx}

    def build_graph(self, state_below):
        dim = self.dim
        nsteps = state_below.shape[0]
        if state_below.ndim == 3:
            batch_size = state_below.shape[1]
        else:
            batch_size = 1
        init_state = tensor.alloc(0., batch_size, dim)
        if self.mask is None:
            self.mask = tensor.alloc(1., nsteps, 1)

        def _slice(_x, n, dim):
            if _x.ndim == 3:
                return _x[:, :, n*dim:(n+1)*dim]
            elif _x.ndim == 2:
                return _x[:, n*dim:(n+1)*dim]
            return _x[n*dim:(n+1)*dim]

        state_belowx = tensor.dot(state_below, self.Wx) + self.bx

        def _step(m_, xx_, h_, Ux):
            preactx = tensor.dot(h_, Ux)
            preactx = preactx + xx_

            h = tensor.tanh(preactx)

            h = m_[:, None] * h + (1. - m_)[:, None] * h_

            return h  # , r, u, preact, preactx

        rval, updates = theano.scan(
            _step,
            sequences=[self.mask, state_belowx],
            outputs_info=[init_state],
            # None, None, None, None],
            non_sequences=[self.Ux],
            name=self.get_prefix() + '_scan_layer',
            n_steps=nsteps,
            strict=True,
            profile=profile)
        return rval


class LSTM(ALayer):
    def __init__(self, nin, dim, rng=None, prefix='', baseprefix='lstm',
                 mask=None):
        '''An LSTM layer

        Parameters
        ----------
        nin : int
            The number of channels of the input
        dim : int
            The number of hidden units of the layer
        rng : numpy.random.randomState
            A numpy random generator
        prefix : string
            Baseprefix_prefix will be used as a name for this object
        baseprefix : string
            Baseprefix_prefix will be used as a name for this object
        mask :
            The
        '''
        self.nin = nin
        self.dim = dim
        self.rng = rng
        self.prefix = prefix
        self.baseprefix = baseprefix
        self.mask = mask
        super(LSTM, self).__init__()

    def __param_init__(self):
        dim = self.dim
        nin = self.nin
        rng = self.rng
        W = numpy.numpy.concatenate([norm_weight(nin, dim, rng=rng),
                                     norm_weight(nin, dim, rng=rng),
                                     norm_weight(nin, dim, rng=rng),
                                     norm_weight(nin, dim, rng=rng)], axis=1)
        U = numpy.numpy.concatenate([ortho_weight(dim, rng=rng),
                                     ortho_weight(dim, rng=rng),
                                     ortho_weight(dim, rng=rng),
                                     ortho_weight(dim, rng=rng)], axis=1)
        b = numpy.zeros((4 * dim,)).astype(theano.config.floatX)

        self._params = {'W': W,  'U': U, 'b': b}

    def build_graph(self, state_below):
        # state_below: (k,m,d)
        dim = self.dim
        # dim = tparams[_p(prefix, 'U')].shape[1]//4
        nsteps = state_below.shape[0]
        if state_below.ndim == 3:
            batch_size = state_below.shape[1]
        else:
            batch_size = 1
        if self.mask is None:
            self.mask = tensor.alloc(1., state_below.shape[0],
                                     state_below.shape[1])

        init_state = tensor.alloc(0., batch_size, dim)
        init_memory = tensor.alloc(0., batch_size, dim)

        def _slice(_x, n, dim):
            if _x.ndim == 3:
                return _x[:, :, n * dim:(n + 1) * dim]
            elif _x.ndim == 2:
                return _x[:, n * dim:(n + 1) * dim]
            return _x[n * dim:(n + 1) * dim]

        def _step(m_, x_, h_, c_):
            preact = tensor.dot(h_, self.U)
            preact += x_
            preact += self.b

            i = tensor.nnet.sigmoid(_slice(preact, 0, dim))
            f = tensor.nnet.sigmoid(_slice(preact, 1, dim))
            o = tensor.nnet.sigmoid(_slice(preact, 2, dim))
            c = _slice(preact, 3, dim)

            c = f * c_ + i * c
            c = m_[:, None] * c + (1. - m_)[:, None] * c_

            h = o * tensor.tanh(c)
            h = m_[:, None] * h + (1. - m_)[:, None] * h_

            return h, c
            # li -> return h, c, i, f, o, preact

        state_below = tensor.dot(state_below, self.W) + self.b
        rval, updates = theano.scan(
            _step,
            sequences=[self.mask, state_below],
            outputs_info=[init_state, init_memory],
            n_steps=nsteps,
            name=self.get_prefix() + '_scan_layer',
            profile=False)
        return rval[0]
        # return rval


class GRU(ALayer):
    def __init__(self, nin, dim, hiero=False, rng=None, prefix='',
                 baseprefix='gru', mask=None):
        """A GRU layer

        Parameters
        ----------
        nin : int
            The number of channels of the input
        dim : int
            The number of hidden units of the layer
        hiero :
            The
        rng : numpy.random.randomState
            A numpy random generator
        prefix : string
            Baseprefix_prefix will be used as a name for this object
        baseprefix : string
            Baseprefix_prefix will be used as a name for this object
        mask :
            The
        """
        self.nin = nin
        self.dim = dim
        self.hiero = hiero
        self.rng = rng
        self.prefix = prefix
        self.baseprefix = baseprefix
        self.mask = mask
        super(GRU, self).__init__()

    def __param_init__(self):
        nin = self.nin
        dim = self.dim
        rng = self.rng

        W = numpy.concatenate([norm_weight(nin, dim, rng=rng),
                               norm_weight(nin, dim, rng=rng)], axis=1)
        U = numpy.concatenate([ortho_weight(dim, rng=rng),
                               ortho_weight(dim, rng=rng)], axis=1)
        b = zeros((2 * dim,)).astype(theano.config.floatX)

        Wx = norm_weight(nin, dim, rng=rng)
        Ux = ortho_weight(dim, rng=rng)
        bx = zeros((dim,)).astype(theano.config.floatX)

        self._params = {'W': W,  'U': U, 'b': b, 'Wx': Wx, 'Ux': Ux, 'bx': bx}

    def build_graph(self, state_below):
        dim = self.Ux.shape[1]   # output size (i.e. n_neurons)
        nsteps = state_below.shape[0]
        if state_below.ndim == 3:
            batch_size = state_below.shape[1]
        else:
            batch_size = 1
        init_state = tensor.alloc(0., batch_size, dim)

        if self.mask is None:
            self.mask = tensor.alloc(1., state_below.shape[0], 1)

        def _slice(_x, n, dim):
            if _x.ndim == 3:
                return _x[:, :, n * dim:(n + 1) * dim]
            elif _x.ndim == 2:
                return _x[:, n * dim:(n + 1) * dim]
            return _x[n * dim:(n + 1) * dim]

        state_below_ = tensor.dot(state_below, self.W) + self.b
        state_belowx = tensor.dot(state_below, self.Wx) + self.bx

        def _step_slice(m_, x_, xx_, h_, U, Ux):
            """_step_slice

            Parameter
            ---------
            m_ : mask, ones if not set
            x_ : state_below_, W*input + b
            xx_ : state_belowx, Wx*input + bx
            h_ : previous step status, size=(batch_size, dim), init=0
            U : U, fixed
            Ux : Ux, fixed

            Note: `m_`, `x_` and `xx_` are part of `seq`

            """
            # preact = (W*input+b) + U*h_prev
            preact = tensor.dot(h_, U)
            preact += x_

            r = tensor.nnet.sigmoid(_slice(preact, 0, dim))  # reset gate
            u = tensor.nnet.sigmoid(_slice(preact, 1, dim))  # update gate

            # h = candidate activation = tanh[(Wx*input+bx) + r(*)Ux*h_prev]
            # (*) = elemwise numpy.prod
            preactx = tensor.dot(h_, Ux)
            preactx = preactx * r
            preactx = preactx + xx_

            h = tensor.tanh(preactx)  # candidate activation

            # output = mask * (u*h_prev + (1-u)*h_candidate ) +
            #          (1-mask)(h_prev)
            h = u * h_ + (1. - u) * h
            h = m_[:, None] * h + (1. - m_)[:, None] * h_

            return h

        rval, updates = theano.scan(
            _step_slice,
            sequences=[self.mask, state_below_, state_belowx],
            outputs_info=[init_state],
            non_sequences=[self.U, self.Ux],
            name=self.get_prefix() + '_scan_layer',
            n_steps=nsteps,
            profile=profile,
            strict=True)
        return rval
