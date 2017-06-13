import numpy as np
from fuel.datasets import IterableDataset
from fuel.streams import DataStream
import theano
from fuel.transformers import Transformer
import fuel
floatX = theano.config.floatX


class SampleDrops(Transformer):
    def __init__(self, data_stream, drop_prob, hidden_dim,
                 is_for_test, **kwargs):
        super(SampleDrops, self).__init__(
            data_stream, **kwargs)
        self.drop_prob = drop_prob
        self.hidden_dim = hidden_dim
        self.is_for_test = is_for_test
        self.produces_examples = False

    def get_data(self, request=None):
        data = next(self.child_epoch_iterator)
        transformed_data = []
        transformed_data.append(data[0])
        transformed_data.append(data[1])
        T, B, _ = data[1].shape
        if self.is_for_test:
            drops = np.ones((T, B, self.hidden_dim)) * self.drop_prob
        else:
            drops = np.random.binomial(n=1, p=self.drop_prob,
                                       size=(T, B, self.hidden_dim))
        transformed_data.append(drops.astype(floatX))
        return transformed_data


def get_seq_mnist_streams(hidden_dim, batch_size=100, drop_prob=0.5):
    permutation = np.random.randint(0, 784, size=(784,))

    train_set, valid_set, test_set = load_data('mnist.pkl.gz')
    train_x = train_set[0].reshape((50000 / batch_size, batch_size, 784))
    train_x = np.swapaxes(train_x, 2, 1)
    train_x = train_x[:, :, :, np.newaxis]
    # Now the dimension is num_batches x 784 x batch_size x 1

    train_y = (np.zeros(train_set[0].shape) - 1)
    # label for each time-step is -1 and for the last one is the real label
    train_y[:, -1] = train_set[1]
    train_y = train_y.reshape((50000 / batch_size, batch_size, 784))
    train_y = np.swapaxes(train_y, 2, 1)
    train_y = train_y[:, :, :, np.newaxis]
    # Now the dimension is num_batches x 784 x batch_size x 1

    valid_x = valid_set[0].reshape((10000 / batch_size, batch_size, 784))
    valid_x = np.swapaxes(valid_x, 2, 1)
    valid_x = valid_x[:, :, :, np.newaxis]
    # Now the dimension is num_batches x 784 x batch_size x 1

    valid_y = (np.zeros(valid_set[0].shape) - 1)
    # label for each time-step is -1 and for the last one is the real label
    valid_y[:, -1] = valid_set[1]
    valid_y = valid_y.reshape((10000 / batch_size, batch_size, 784))
    valid_y = np.swapaxes(valid_y, 2, 1)
    valid_y = valid_y[:, :, :, np.newaxis]
    # Now the dimension is num_batches x 784 x batch_size x 1

    train_x = train_x[:, permutation]
    valid_x = valid_x[:, permutation]

    train = IterableDataset({'x': train_x.astype(floatX),
                             'y': train_y[:, -1, :, 0].astype('int32')})
    train_stream = DataStream(train)
    train_stream = SampleDrops(train_stream, drop_prob, hidden_dim, False)
    train_stream.sources = ('y', 'x', 'drops')

    train_stream.get_epoch_iterator().next()

    valid = IterableDataset({'x': valid_x.astype(floatX),
                             'y': valid_y[:, -1, :, 0].astype('int32')})
    valid_stream = DataStream(valid)
    valid_stream = SampleDrops(valid_stream, drop_prob, hidden_dim, True)
    valid_stream.sources = ('y', 'x', 'drops')

    return train_stream, valid_stream


def get_dataset(which_set):
    MNIST = fuel.datasets.MNIST
    # jump through hoops to instantiate only once and only if needed
    _datasets = dict(
        train=MNIST(which_sets=["train"], subset=slice(None, 50000)),
        valid=MNIST(which_sets=["train"], subset=slice(50000, None)),
        test=MNIST(which_sets=["test"]))
    return _datasets[which_set]


def get_stream(which_set, batch_size, drop_prob,
               hidden_dim, for_evaluation, num_examples=None):
    np.random.seed(seed=1)
    permutation = np.random.randint(0, 784, size=(784,))
    dataset = get_dataset(which_set)
    if num_examples is None or num_examples > dataset.num_examples:
        num_examples = dataset.num_examples
    stream = fuel.streams.DataStream.default_stream(
        dataset,
        iteration_scheme=fuel.schemes.ShuffledScheme(num_examples, batch_size))
    ds = SampleDrops2(stream, drop_prob, hidden_dim,
                      for_evaluation, permutation)
    ds.sources = ('x', 'y', 'drops')
    return ds


class SampleDrops2(Transformer):
    def __init__(self, data_stream, drop_prob, hidden_dim,
                 is_for_test, permutation, **kwargs):
        super(SampleDrops2, self).__init__(
            data_stream, **kwargs)
        self.drop_prob = drop_prob
        self.hidden_dim = hidden_dim
        self.is_for_test = is_for_test
        self.produces_examples = False
        self.permutation = permutation

    def get_data(self, request=None):
        data = next(self.child_epoch_iterator)
        transformed_data = []
        transformed_data.append(
            np.swapaxes(data[0].reshape(data[0].shape[0], -1),
                        0, 1)[self.permutation, :, np.newaxis])
        transformed_data.append(data[1][:, 0])
        T, B, _ = transformed_data[0].shape
        if self.is_for_test:
            drops = np.ones((T, B, self.hidden_dim)) * self.drop_prob
        else:
            drops = np.random.binomial(n=1, p=self.drop_prob,
                                       size=(T, B, self.hidden_dim))
        transformed_data.append(drops.astype(floatX))
        return transformed_data


# PTB
_data_cache = dict()


def get_data(which_set):
    if which_set not in _data_cache:
        path = 'char_level_penntree.npz'
        data = np.load(path)
        # put the entire thing on GPU in one-hot (takes
        # len(self.vocab) * len(self.data) * sizeof(floatX) bytes
        # which is about 1G for the training set and less for the
        # other sets)
        #cudandarray = theano.sandbox.cuda.cuda_ndarray.cuda_ndarray.CudaNdarray
        # (doing it in numpy first because cudandarray doesn't accept
        # lists of indices)
        one_hot_data = np.eye(len(data["vocab"]), dtype=theano.config.floatX)[data[which_set]]
        #_data_cache[which_set] = cudandarray(one_hot_data)
    return one_hot_data#_data_cache[which_set]


class PTB(fuel.datasets.Dataset):
    provides_sources = ('features','targets')
    example_iteration_scheme = None

    def __init__(self, which_set, X_noise, Y_noise, rng, X_noise_type, Y_noise_type, percent_of_data, length, augment=False):
        self.which_set = which_set
        self.length = length
        self.augment = augment
        #self.data = get_data(which_set)
        self.X_noise = X_noise
        self.Y_noise = Y_noise
        self.rng = rng
        self.X_noise_type = X_noise_type 
        self.Y_noise_type = Y_noise_type 
        if self.augment:
            print "too much random! not doing augmentation."
            # -1 so we have one self.length worth of room for augmentation
            #self.num_examples -= 1


        # ADDING NOISE
        x = get_data(which_set)
        y = get_data(which_set)
        data_mean = x.mean()
        data_stdv = x.std()
        num_examples = int(len(x) / self.length)
        self.num_examples = int(num_examples * (float(percent_of_data)/100))
        x = x[:self.num_examples * self.length]
        y = y[:self.num_examples * self.length]
        data_len = len(x)
        print "data length: ", data_len 
        oh_dim = x.shape[1]
        if self.Y_noise > 0.0:
            rand_Y_one_hots = np.eye(oh_dim)[rng.randint(oh_dim, size=data_len)]
            if Y_noise_type == 'seq':
                y = y.reshape(self.num_examples, self.length, oh_dim)
                rand_Y_one_hots = rand_Y_one_hots.reshape(self.num_examples, self.length, oh_dim)
                num_y_noised = int(self.Y_noise*self.num_examples)
                y_noise_mask = np.zeros(self.num_examples)
                y_noise_mask[:num_y_noised] = 1
                y_noise_mask = y_noise_mask[rng.permutation(self.num_examples)].astype(bool)
                y[y_noise_mask] = rand_Y_one_hots[y_noise_mask]
            elif Y_noise_type == 'char':
                num_y_noised = int(self.Y_noise*data_len)
                y_noise_mask = np.zeros(data_len)
                y_noise_mask[:num_y_noised] = 1
                y_noise_mask = y_noise_mask[rng.permutation(data_len)].astype(bool)
                y[y_noise_mask] = rand_Y_one_hots[y_noise_mask]
                y = y.reshape(self.num_examples, self.length, oh_dim)
            else:
               pass
        else:
            y = y.reshape(self.num_examples, self.length, oh_dim)
        if self.X_noise > 0.0:
            rand_X_one_hots = np.eye(oh_dim)[rng.randint(oh_dim, size=data_len)]
            if X_noise_type == 'seq':
                x = x.reshape(self.num_examples, self.length, oh_dim)
                rand_X_one_hots = rand_X_one_hots.reshape(self.num_examples, self.length, oh_dim)
                num_x_noised = int(self.X_noise*self.num_examples)
                x_noise_mask = np.zeros(self.num_examples)
                x_noise_mask[:num_x_noised] = 1
                x_noise_mask = x_noise_mask[rng.permutation(self.num_examples)].astype(bool)
                x[x_noise_mask] = rand_X_one_hots[x_noise_mask]
            elif Y_noise_type == 'char':
                num_x_noised = int(self.X_noise*data_len)
                x_noise_mask = np.zeros(data_len)
                x_noise_mask[:num_x_noised] = 1
                x_noise_mask = x_noise_mask[rng.permutation(data_len)].astype(bool)
                x[x_noise_mask] = rand_X_one_hots[x_noise_mask]
                x = x.reshape(self.num_examples, self.length, oh_dim)
            else:
                print "derp"
        else:
            x = x.reshape(self.num_examples, self.length, oh_dim)
        self.data = (x,y)

        super(PTB, self).__init__()

    def open(self):
        # reshape to nonoverlapping examples
        #data = (data[:self.num_examples * self.length]
        #        .reshape((self.num_examples, self.length, self.data.shape[1])))
        # return the data so we will get it as the "state" argument to get_data
        #import ipdb; ipdb.set_trace()
        return self.data

    def get_data(self, state, request):
        #if isinstance(request, (tuple, list)):
        #    request = np.array(request, dtype=np.int64)
        #    import ipdb; ipdb.set_trace()
        #    return (state.take(request, 0),)
        #import ipdb; ipdb.set_trace()
        #return (state[request],)
        return (self.data[0][request], self.data[1][request])
        #return self.filter_sources(data)

class oldPTB(fuel.datasets.Dataset):
    provides_sources = ('features',)
    example_iteration_scheme = None

    def __init__(self, which_set, length, augment=True):
        self.which_set = which_set
        self.length = length
        self.augment = augment
        self.data = get_data(which_set) #(5059550,50)
        self.num_examples = int(len(self.data) / self.length)
        if self.augment:
            # -1 so we have one self.length worth of room for augmentation
            self.num_examples -= 1
        super(oldPTB, self).__init__()

    def open(self):
        offset = 0
        if self.augment:
            # choose an offset to get some data augmentation by
            # not always chopping the examples at the same point.
            offset = np.random.randint(self.length)
        # none of this should copy
        data = self.data[offset:]
        # reshape to nonoverlapping examples
        data = (data[:self.num_examples * self.length]
                .reshape((self.num_examples, self.length, self.data.shape[1])))
        # return the data so we will get it as the "state" argument to get_data
        return data #(50594,100,50)

    def get_data(self, state, request):
        if isinstance(request, (tuple, list)):
            request = np.array(request, dtype=np.int64)
            return (state.take(request, 0),)
        return (state[request],) #(32,100,50)

class SampleDropsPTB(Transformer):
    def __init__(self, data_stream, drop_prob_states, drop_prob_cells, drop_prob_igates, hidden_dim,
                 is_for_test, **kwargs):
        super(SampleDropsPTB, self).__init__(
            data_stream, **kwargs)
        self.drop_prob_states = drop_prob_states
        self.drop_prob_cells = drop_prob_cells
        self.drop_prob_igates = drop_prob_igates
        self.hidden_dim = hidden_dim
        self.is_for_test = is_for_test
        self.produces_examples = False

    def get_data(self, request=None):
        data = next(self.child_epoch_iterator)
        transformed_data = []
        # Now it is: T x B x F
        transformed_data.append(np.swapaxes(data[0], 0, 1))#[:-1])
        #transformed_data.append(np.swapaxes(data[0], 0, 1)[1:])
        T, B, _ = transformed_data[0].shape
        if self.is_for_test:
            drops_states = np.ones((T, B, self.hidden_dim)) * self.drop_prob_states
        else:
            drops_states = np.random.binomial(n=1, p=self.drop_prob_states,
                                       size=(T, B, self.hidden_dim))
        if self.is_for_test:
            drops_cells = np.ones((T, B, self.hidden_dim)) * self.drop_prob_cells
        else:
            drops_cells = np.random.binomial(n=1, p=self.drop_prob_cells,
                                       size=(T, B, self.hidden_dim))
        if self.is_for_test:
            drops_igates = np.ones((T, B, self.hidden_dim)) * self.drop_prob_igates
        else:
            drops_igates = np.random.binomial(n=1, p=self.drop_prob_igates,
                                       size=(T, B, self.hidden_dim))
        transformed_data.append(drops_states.astype(floatX))
        transformed_data.append(drops_cells.astype(floatX))
        transformed_data.append(drops_igates.astype(floatX))
        return transformed_data

class SampleDropsPTBnoised(Transformer):
    def __init__(self, data_stream, drop_prob_states, drop_prob_cells, drop_prob_igates, hidden_dim,
                 is_for_test, **kwargs):
        super(SampleDropsPTBnoised, self).__init__(
            data_stream, **kwargs)
        self.drop_prob_states = drop_prob_states
        self.drop_prob_cells = drop_prob_cells
        self.drop_prob_igates = drop_prob_igates
        self.hidden_dim = hidden_dim
        self.is_for_test = is_for_test
        self.produces_examples = False

    def get_data(self, request=None):
        data = next(self.child_epoch_iterator)
        transformed_data = []
        data_x = data[0]
        data_y = data[1]
        #import ipdb; ipdb.set_trace()
        # Now it is: T x B x F
        #transformed_data.append(np.swapaxes(data_x[0], 0, 1))#[:-1])
        #transformed_data.append(np.swapaxes(data_y[0], 0, 1))#[:-1])
        transformed_data.append(np.swapaxes(data_x, 0, 1))#[:-1])
        transformed_data.append(np.swapaxes(data_y, 0, 1))#[:-1])
        #transformed_data.append(np.swapaxes(data[0], 0, 1)[1:])
        T, B, _ = transformed_data[0].shape
        if self.is_for_test:
            drops_states = np.ones((T, B, self.hidden_dim)) * self.drop_prob_states
        else:
            drops_states = np.random.binomial(n=1, p=self.drop_prob_states,
                                       size=(T, B, self.hidden_dim))
        if self.is_for_test:
            drops_cells = np.ones((T, B, self.hidden_dim)) * self.drop_prob_cells
        else:
            drops_cells = np.random.binomial(n=1, p=self.drop_prob_cells,
                                       size=(T, B, self.hidden_dim))
        if self.is_for_test:
            drops_igates = np.ones((T, B, self.hidden_dim)) * self.drop_prob_igates
        else:
            drops_igates = np.random.binomial(n=1, p=self.drop_prob_igates,
                                       size=(T, B, self.hidden_dim))
        transformed_data.append(drops_states.astype(floatX))
        transformed_data.append(drops_cells.astype(floatX))
        transformed_data.append(drops_igates.astype(floatX))
        return transformed_data
    
class Sample_static_mask_DropsPTB(Transformer):
    def __init__(self, data_stream, drop_prob_states, drop_prob_cells, drop_prob_igates, hidden_dim,
                 is_for_test, seq_len, batch_size, **kwargs):
        super(Sample_static_mask_DropsPTB, self).__init__(
            data_stream, **kwargs)
        self.drop_prob_states = drop_prob_states
        self.drop_prob_cells = drop_prob_cells
        self.drop_prob_igates = drop_prob_igates
        self.hidden_dim = hidden_dim
        self.T = seq_len
        #self.B = batch_size
        self.is_for_test = is_for_test
        self.produces_examples = False
        #self.drops_states_mask = np.tile(np.random.binomial(n=1, p=self.drop_prob_states,
                                       #size=(self.T, 1, self.hidden_dim)), (1, self.B, 1))
        #self.drops_cells_mask  = np.tile(np.random.binomial(n=1, p=self.drop_prob_cells,
                                       #size=(self.T, 1, self.hidden_dim)), (1, self.B, 1))
        #self.drops_igates_mask = np.tile(np.random.binomial(n=1, p=self.drop_prob_igates,
                                       #size=(self.T, 1, self.hidden_dim)), (1, self.B, 1))
        self.drops_states_mask = np.random.binomial(n=1, p=self.drop_prob_states,
                                       size=(self.T, 1, self.hidden_dim))
        self.drops_cells_mask  = np.random.binomial(n=1, p=self.drop_prob_cells,
                                       size=(self.T, 1, self.hidden_dim))
        self.drops_igates_mask = np.random.binomial(n=1, p=self.drop_prob_igates,
                                       size=(self.T, 1, self.hidden_dim))

    def get_data(self, request=None):
        data = next(self.child_epoch_iterator)
        transformed_data = []
        # Now it is: T x B x F
        transformed_data.append(np.swapaxes(data[0], 0, 1))#[:-1])
        #transformed_data.append(np.swapaxes(data[0], 0, 1)[1:])
        _T, _B, _ = transformed_data[0].shape
        assert self.T == _T
        #assert self.B == _B
        drops_states =  np.tile(self.drops_states_mask, (1, _B, 1))
        drops_cells =  np.tile(self.drops_cells_mask, (1, _B, 1))
        drops_igates =  np.tile(self.drops_igates_mask, (1, _B, 1))
        #if self.is_for_test:
            #drops_states = np.ones((self.T, _B, self.hidden_dim)) * np.tile(self.drop_prob_states, (1, _B, 1))
        #else:
            #drops_states = self.drops_states_mask
        #if self.is_for_test:
            #drops_cells = np.ones((self.T, _B, self.hidden_dim)) * np.tile(self.drop_prob_cells, (1, _B, 1))
        #else:
            #drops_cells = self.drops_cells_mask
        #if self.is_for_test:
            #drops_igates = np.ones((self.T, _B, self.hidden_dim)) * np.tile(self.drop_prob_igates, (1, _B, 1))
        #else:
            #drops_igates = self.drops_igates_mask
        transformed_data.append(drops_states.astype(floatX))
        transformed_data.append(drops_cells.astype(floatX))
        transformed_data.append(drops_igates.astype(floatX))
        return transformed_data


def get_ptb_stream(which_set, batch_size, length, drop_prob_states, drop_prob_cells, drop_prob_igates,
                   hidden_dim, for_evaluation, num_examples=None,
                   augment=True):
    
    dataset = oldPTB(which_set, length=length, augment=augment)
    #import ipdb; ipdb.set_trace()
    if num_examples is None or num_examples > dataset.num_examples:
        num_examples = dataset.num_examples
    stream = fuel.streams.DataStream.default_stream(
        dataset,
        iteration_scheme=fuel.schemes.ShuffledScheme(num_examples, batch_size))
    ds = SampleDropsPTB(stream, drop_prob_states, drop_prob_cells, drop_prob_igates, hidden_dim,
                        for_evaluation)
    ds.sources = ('features', 'drops_states', 'drops_cells', 'drops_igates')#'outputs',
    return ds

def get_noised_stream(which_set, batch_size, length, drop_prob_states, drop_prob_cells, drop_prob_igates,
                      hidden_dim, X_noise, Y_noise, rng, X_noise_type, Y_noise_type, percent_of_data, for_evaluation, num_examples=None, augment=True):
    noised_dataset = PTB(which_set, X_noise, Y_noise, rng, X_noise_type, Y_noise_type, percent_of_data, length=length, augment=augment)
    if num_examples is None or num_examples > noised_dataset.num_examples:
        num_examples = noised_dataset.num_examples
    stream = fuel.streams.DataStream.default_stream(
        noised_dataset,
        iteration_scheme=fuel.schemes.ShuffledScheme(num_examples, batch_size))
    ds = SampleDropsPTBnoised(stream, drop_prob_states, drop_prob_cells, drop_prob_igates, hidden_dim, for_evaluation)
    ds.sources = ('features', 'targets', 'drops_states', 'drops_cells', 'drops_igates')
    return ds

def get_static_mask_ptb_stream(which_set, batch_size, length, drop_prob_states, drop_prob_cells, drop_prob_igates,
                   hidden_dim, for_evaluation, num_examples=None,
                   augment=True):
    
    dataset = PTB(which_set, length=length, augment=augment)
    if num_examples is None or num_examples > dataset.num_examples:
        num_examples = dataset.num_examples
    stream = fuel.streams.DataStream.default_stream(
        dataset,
        iteration_scheme=fuel.schemes.ShuffledScheme(num_examples, batch_size))
    ds = Sample_static_mask_DropsPTB(stream, drop_prob_states, drop_prob_cells, drop_prob_igates, hidden_dim,
                        for_evaluation, length, batch_size)
    ds.sources = ('features',  'drops_states', 'drops_cells', 'drops_igates')#'outputs',
    return ds
