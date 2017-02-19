from blocks.bricks import application, lazy, Initializable, Tanh, Linear, MLP
from blocks.bricks.parallel import Fork
from blocks.bricks.recurrent import (Bidirectional, SimpleRecurrent, LSTM,
                                     GatedRecurrent)

from blocks.bricks.lookup import LookupTable
from blocks.utils import dict_union
from picklable_itertools.extras import equizip




class Encoder(Initializable):
    def __init__(self, dimension, input_size, rnn_type=None,
                 embed_input=False, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        if rnn_type is None:
            rnn_type = SimpleRecurrent
        if embed_input:
            self.embedder = LookupTable(input_size, dimension)
        else:
            self.embedder = Linear(input_size, dimension)
        encoder = Bidirectional(rnn_type(dim=dimension, activation=Tanh()))
        fork = Fork([name for name in encoder.prototype.apply.sequences
                     if name != 'mask'])
        fork.input_dim = dimension
        fork.output_dims = [dimension for _ in fork.input_names]

        self.fork = fork
        self.encoder = encoder
        self.children = [fork, encoder, self.embedder]

    @application
    def apply(self, input_, input_mask):
        input_ = self.embedder.apply(input_)
        return self.encoder.apply(
            **dict_union(
                self.fork.apply(input_, as_dict=True),
                mask=input_mask))


class AbstractEncoder(Initializable):
    def __init__(self, **kwargs):
        super(AbstractEncoder, self).__init__(**kwargs)

    @application
    def apply(self, input_, input_mask):
        input_ = self.embedder.apply(input_)
        input_ = self.fork.apply(input_, as_dict=True)
        encoded = self.encoder.apply(**dict_union(input_, mask=input_mask,
                                                  as_dict=True))
        return encoded['states']


class LSTMEncoder(AbstractEncoder):
    def __init__(self, dimension, input_size,
                 embed_input=False, **kwargs):
        super(LSTMEncoder, self).__init__(**kwargs)
        if embed_input:
            self.embedder = LookupTable(input_size, dimension)
        else:
            self.embedder = Linear(input_size, dimension)
        self.fork = Fork(['inputs'], dimension,
                         output_dims=[dimension],
                         prototype=Linear(dimension, 4 * dimension))
        encoder = Bidirectional(LSTM(dim=dimension, activation=Tanh()))

        self.encoder = encoder
        self.children = [encoder, self.embedder, self.fork]


class GRUEncoder(AbstractEncoder):
    def __init__(self, dimension, input_size,
                 embed_input=False, **kwargs):
        super(GRUEncoder, self).__init__(**kwargs)
        if embed_input:
            self.embedder = LookupTable(input_size, dimension)
        else:
            self.embedder = Linear(input_size, dimension)
        self.fork = Fork(['inputs', 'gate_inputs'],
                         dimension,
                         output_dims=[dimension, 2 * dimension],
                         prototype=Linear())
        encoder = Bidirectional(GatedRecurrent(dim=dimension,
                                               activation=Tanh()))

        self.encoder = encoder
        self.children = [encoder, self.embedder, self.fork]


class SimpleEncoder(AbstractEncoder):
    def __init__(self, dimension, input_size,
                 embed_input=False, **kwargs):
        super(SimpleEncoder, self).__init__(**kwargs)
        if embed_input:
            self.embedder = LookupTable(input_size, dimension)
        else:
            self.embedder = Linear(input_size, dimension)
        self.transform = MLP([Tanh()], [dimension, dimension])
        self.fork = Fork(['inputs'], dimension,
                         output_dims=[dimension],
                         prototype=Linear(dimension, dimension))
        encoder = Bidirectional(SimpleRecurrent(dim=dimension,
                                                activation=Tanh()))

        self.encoder = encoder
        self.children = [encoder, self.embedder, self.transform, self.fork]


class MultiLayerEncoder(Initializable):
    """Stacked Bidirectional RNN.
    Parameters
    ---------
    networks : a list of instance of :class:`BidirectionalGraves`
    dims: a list of dimensions from the first network state to the last one.
    """

    @lazy()
    def __init__(self, networks, dims, **kwargs):
        super(MultiLayerEncoder, self).__init__(**kwargs)
        self.dims = dims
        self.networks = networks

        self.hid_linear_trans_forw = [Fork([name for name in
                                            networks[i].prototype.apply.sequences
                                            if name != 'mask'],
                                           name='fork_forw_{}'.format(i),
                                           prototype=Linear(), **kwargs)
                                      for i in range(len(networks))]

        self.hid_linear_trans_back = [Fork([name for name in
                                            networks[i].prototype.apply.sequences
                                            if name != 'mask'],
                                           name='fork_back_{}'.format(i),
                                           prototype=Linear(), **kwargs)
                                      for i in range(len(networks))]

        self.out_linear_trans = Linear(name='out_linear', **kwargs)
        self.children = (networks +
                         self.hid_linear_trans_forw +
                         self.hid_linear_trans_back +
                         [self.out_linear_trans])
        self.num_layers = len(networks)

    @application
    def apply(self, x, input_mask, *args, **kwargs):
        # For each layer we first apply two transformations using fork for
        # forward and backward RNNs. Fork is because that it is possible
        # that an encoder needs more than one input and so needs more than
        # one linear transformation. We feed these two transformations to
        # Bidirectional RNN and the output will be the concatenation of
        # the states of the two RNN inside BidirectionalRNN (2 * h_dim).
        raw_states = x
        for i in range(self.num_layers):
            transformed_x_forw = self.hid_linear_trans_forw[i].apply(
                raw_states, as_dict=True)
            transformed_x_back = self.hid_linear_trans_back[i].apply(
                raw_states, as_dict=True)
            raw_states = self.networks[i].apply(transformed_x_forw,
                                                transformed_x_back,
                                                input_mask,
                                                *args, **kwargs)
        encoder_out = self.out_linear_trans.apply(raw_states)
        return encoder_out


class DropMultiLayerEncoder(Initializable):
    """Stacked Bidirectional RNN.
    Parameters
    ---------
    networks : a list of instance of :class:`BidirectionalGraves`
    dims: a list of dimensions from the first network state to the last one.
    """

    @lazy()
    def __init__(self, networks, dims, **kwargs):
        super(DropMultiLayerEncoder, self).__init__(**kwargs)
        self.dims = dims
        self.networks = networks
        self.use_bias = True

        self.hid_linear_trans_forw = [Fork([name for name in
                                            networks[i].prototype.apply.sequences
                                            if name not in ['mask', 'drops_states', 'drops_cells', 'drops_igates']],
                                           name='fork_forw_{}'.format(i),
                                           prototype=Linear(), **kwargs)
                                      for i in range(len(networks))]

        self.hid_linear_trans_back = [Fork([name for name in
                                            networks[i].prototype.apply.sequences
                                            if name not in ['mask', 'drops_states', 'drops_cells', 'drops_igates']],
                                           name='fork_back_{}'.format(i),
                                           prototype=Linear(), **kwargs)
                                      for i in range(len(networks))]

        self.out_linear_trans = Linear(name='out_linear', **kwargs)
        self.children = (networks +
                         self.hid_linear_trans_forw +
                         self.hid_linear_trans_back +
                         [self.out_linear_trans])
        self.num_layers = len(networks)

    @application
    def apply(self, x, drops_states, drops_cells, drops_igates, input_mask, *args, **kwargs):
        # For each layer we first apply two transformations using fork for
        # forward and backward RNNs. Fork is because that it is possible
        # that an encoder needs more than one input and so needs more than
        # one linear transformation. We feed these two transformations to
        # Bidirectional RNN and the output will be the concatenation of
        # the states of the two RNN inside BidirectionalRNN (2 * h_dim).

        # This is because all the layers have the same dim
        states_dim = self.dims[1]
        raw_states = x
        for i in range(self.num_layers):
            transformed_x_forw = self.hid_linear_trans_forw[i].apply(
                raw_states, as_dict=True)
            transformed_x_back = self.hid_linear_trans_back[i].apply(
                raw_states, as_dict=True)
            raw_states = self.networks[i].apply(transformed_x_forw,
                                                transformed_x_back,
                                                drops_states[0][:, :, i * states_dim:(i + 1)*states_dim],
                                                drops_cells[0][:, :, i * states_dim:(i + 1)*states_dim],
                                                drops_igates[0][:, :, i * states_dim:(i + 1)*states_dim],
                                                drops_states[1][:, :, i * states_dim:(i + 1)*states_dim],
                                                drops_cells[1][:, :, i * states_dim:(i + 1)*states_dim],
                                                drops_igates[1][:, :, i * states_dim:(i + 1)*states_dim],
                                                input_mask,
                                                *args, **kwargs)
        encoder_out = self.out_linear_trans.apply(raw_states)
        return encoder_out

    def _push_allocation_config(self):
        if not len(self.dims) - 2 == self.num_layers:
            raise ValueError

        # Input of the first layer is x so for both forward and
        # backward forks, input_dim = dims[0] and
        # output_dim = [dim of all inputs needed for the next layer].
        # But for other layers since the output is the concatenation,
        # forward and backward transformations have an input_dim
        # twice the actual dimension of layer.
        #
        self.hid_linear_trans_forw[0].input_dim = self.dims[0]
        self.hid_linear_trans_back[0].input_dim = self.dims[0]
        self.hid_linear_trans_forw[0].output_dims = \
            [self.networks[0].prototype.get_dim(name) for
             name in self.hid_linear_trans_forw[0].input_names]
        self.hid_linear_trans_back[0].output_dims = \
            [self.networks[0].prototype.get_dim(name) for
             name in self.hid_linear_trans_back[0].input_names]

        for network, input_dim, layer_forw, layer_back in \
                equizip(self.networks[1:],
                        self.dims[1: -2],
                        self.hid_linear_trans_forw[1:],
                        self.hid_linear_trans_back[1:]):
            layer_forw.input_dim = input_dim * 2
            layer_forw.output_dims = \
                [network.prototype.get_dim(name) for
                 name in layer_forw.input_names]
            layer_forw.use_bias = self.use_bias
            layer_back.input_dim = input_dim * 2
            layer_back.output_dims = \
                [network.prototype.get_dim(name) for
                 name in layer_back.input_names]
            layer_back.use_bias = self.use_bias
        self.out_linear_trans.input_dim = self.dims[-2] * 2
        self.out_linear_trans.output_dim = self.dims[-1]
        self.out_linear_trans.use_bias = self.use_bias
