'''
Build a tweet sentiment analyzer
'''

from __future__ import print_function
import six.moves.cPickle as pickle
import time
from collections import OrderedDict
import sys
import time
from sys import argv
import numpy
import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import imdb
from imdb import masklist

import scipy.io as sio
datanamee = 'sat'
datasets = {datanamee: imdb.load_data}

# Set the random number generators' seeds for consistency
SEED = 123
numpy.random.seed(SEED)


def numpy_floatX(data):
    # return numpy.asarray(data, dtype=config.floatX)
    return numpy.asarray(data, dtype=config.floatX)

def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])
    # print('minibatches', minibatches)
    return zip(range(len(minibatches)), minibatches)


def get_dataset(name):
    return datasets[name]


def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    for kk, vv in params.items():
        tparams[kk].set_value(vv)


def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.items():
        new_params[kk] = vv.get_value()
    return new_params


def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(use_noise,
                         (state_before *
                          trng.binomial(state_before.shape,
                                        p=0.5, n=1,
                                        dtype=state_before.dtype)),
                         state_before * 0.5)
    return proj


def _p(pp, name):
    return '%s_%s' % (pp, name)


def init_params(options):
    """
    Global (not LSTM) parameter. For the embeding and the classifier.
    """
    params = OrderedDict()
    # embedding
    # randn = numpy.random.rand(options['n_words'],
    #                          options['dim_proj'])
    # params['Wemb'] = (0.01 * randn).astype(config.floatX)
    params = get_layer(options['encoder'])[0](options,
                                              params,
                                              prefix=options['encoder'])
    # classifie
    for i in range(recyl_maxlen):
        # params for label prediction
        params['U_' + str(i)] =  (numpy.random.randn((i + 1) * options['dim_proj'],
                                                int(options['ydim'])).astype(config.floatX) /
                                                numpy.sqrt((i + 1) * options['dim_proj']))
        params['b_' + str(i)] = numpy.zeros((int(options['ydim']),)).astype(config.floatX)

        # params for modal prediction
        params['U_seq_' + str(i)] =  (numpy.random.randn( options['dim_proj'],
                                                int( options['maxlen'])).astype(config.floatX)
                                                / numpy.sqrt(options['dim_proj']))
        params['b_seq_' + str(i)] = numpy.zeros((int(options['maxlen']),)).astype(config.floatX)

    return params


def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.items():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params


def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.items():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


def get_layer(name):
    fns = layers[name]
    return fns


def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(config.floatX)


def param_init_lstm(options, params, prefix='lstm'):
    """
    Init the LSTM parameter:

    :see: init_params
    """
    W = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'W')] = W
    U = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'U')] = U
    b = numpy.zeros((4 * options['dim_proj'],))
    params[_p(prefix, 'b')] = b.astype(config.floatX)

    return params



def lstm_layer(tparams, state_below, options, prefix='lstm', mask=None,h_before=None,c_before=None):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step(m_, x_, h_, c_):
        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_

        i = tensor.nnet.sigmoid(_slice(preact, 0, options['dim_proj']))
        f = tensor.nnet.sigmoid(_slice(preact, 1, options['dim_proj']))
        o = tensor.nnet.sigmoid(_slice(preact, 2, options['dim_proj']))
        c = tensor.tanh(_slice(preact, 3, options['dim_proj']))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])

    dim_proj = options['dim_proj']
    [h,c] = _step(mask, state_below,h_before,c_before)
    return h,c


# ff: Feed Forward (normal neural net), only useful to put after lstm
#     before the classifier.
layers = {'lstm': (param_init_lstm, lstm_layer)}


def sgd(lr, tparams, grads, x, x3,y2, mask, y, cost,modal_cost, max_cost):
    """ Stochastic Gradient Descent

    :note: A more complicated version of sgd then needed.  This is
        done like that for adadelta and rmsprop.

    """
    # New set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.items()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    # f_grad_shared = theano.function([idxs, x, mask, y], cost, updates=gsup,
    #                                 name='sgd_f_grad_shared')
    f_grad_shared = theano.function([mask, y, x, x3,y2, modal_cost, max_cost], cost, updates=gsup,
                                    name='sgd_f_grad_shared')

    pup = [(p, p - lr * g) for p, g in zip(tparams.values(), gshared)]

    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=pup,
                               name='sgd_f_update')

    return f_grad_shared, f_update


def adadelta(lr, tparams, grads, x, x3,y2, mask, y, cost, modal_cost,max_cost):
    """
    An adaptive learning rate optimizer

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([mask, y, x, x3, y2, modal_cost, max_cost], cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared', on_unused_input='ignore', allow_input_downcast=True)
    # f_grad_shared = theano.function([idxs, x, mask, y], cost, updates=zgup + rg2up,
    #                                 name='adadelta_f_grad_shared')

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update', allow_input_downcast=True)

    return f_grad_shared, f_update


def rmsprop(lr, tparams, grads, x, mask, y, cost):
    """
    A variant of  SGD that scales the step size by running average of the
    recent step norms.

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [Hint2014]_.

    .. [Hint2014] Geoff Hinton, *Neural Networks for Machine Learning*,
       lecture 6a,
       http://cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    # f_grad_shared = theano.function([x, mask, y], cost,
    #                                 updates=zgup + rgup + rg2up,
    #                                 name='rmsprop_f_grad_shared')
    f_grad_shared = theano.function([mask, y, x], cost,
                                    updates=zgup + rgup + rg2up,
                                    name='rmsprop_f_grad_shared')

    updir = [theano.shared(p.get_value() * numpy_floatX(0.),
                           name='%s_updir' % k)
             for k, p in tparams.items()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(tparams.values(), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new + param_up,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')

    return f_grad_shared, f_update


def build_model(tparams, options):
    '''
    x: traing data
    y: traing label
    x3: neighbor data, datanum * neighbornum * featuredim
    y2: neighbor label
    '''
    trng = RandomStreams(SEED)
    # Used for dropout.
    use_noise = theano.shared(numpy_floatX(0.))
    maskl = tensor.matrix('maskl', dtype=config.floatX)
    y = tensor.vector('y', dtype='int32')
    x = tensor.matrix('x', dtype=config.floatX)
    n_samples = x.shape[0]
    dim_proj = x.shape[1]
    maxlen = options['maxlen']
    x3 = tensor.tensor3('x3', dtype=config.floatX)
    y2 = tensor.matrix('y2', dtype='int32')
    neigh_num = x3.shape[1]
    x_nerghbors = tensor.reshape(x3, [n_samples * neigh_num, dim_proj])
    modal_cost = tensor.vector('modal_cost', dtype=config.floatX)
    max_cost = tensor.scalar('max_cost', dtype=config.floatX)
    h = tensor.alloc(numpy_floatX(0.),n_samples,dim_proj)
    c = tensor.alloc(numpy_floatX(0.),n_samples,dim_proj)
    h_n = tensor.alloc(numpy_floatX(0.),n_samples * neigh_num,dim_proj)
    c_n = tensor.alloc(numpy_floatX(0.),n_samples * neigh_num,dim_proj)
    cost = 0
    cost1_mean = []
    cost2_mean = []
    cost3_mean = []
    next_mean = []
    mask = tensor.ones_like(x[:,0], dtype= config.floatX) # maks whether instance enter the  ith iter
    mask_n = tensor.ones_like(x_nerghbors[:,0], dtype= config.floatX)
    masks = []
    projs = []
    masks.append(mask)
    next_modal = tensor.zeros_like(x[:,0], dtype= 'int32')
    next_modal_n = tensor.zeros_like(x_nerghbors[:,0], dtype= 'int32')
    # cost_vector = tensor.alloc(numpy_floatX(0.),n_samples,1)
    cost_vector = tensor.alloc(numpy_floatX(0.),1, n_samples)
    f_pred_set = []
    f_pred_seq_set = []
    f_pred_seq_prob_set = []
    f_get_fea_set = []
    f_fea_other_set = []
    def get_other3(x, next_modal):
        fea_other = tensor.tile(x,(maxlen,1))
        fea_other = x.T
        fea_single = fea_other[:,next_modal]
        return fea_other,fea_single

    def get_other(x):
        # change the feature x from dim to the form of maxlen * dim
        fea_other = []
        for i in range(maxlen):
            fea_other.append(x * maskl[i])
        return tensor.stack(fea_other)
    def get_single(x,next_modal):
        # get the current modal' feature
        fea_single = x * maskl[next_modal]
        return fea_single


    def compute_dist(neighbor, pred_neighbor, fea_single, pred, mask, y, y2):
        '''
        minimize same label neighbor's distance, maximize different label neighbor's distance
        neighbor: neighbor's feature
        pred_neighbor: neighbor's netmodal's prediction
        fea_single: current instance's feature
        pred: current instance's prediction
        mask: whether current instance stops
        y: current instance's label
        y2: neighbor instance's label
        '''
        loss = 0
        if mask:
            ifsamelabel = -1
            for i in range(3):
                if y == y2[i]:
                    ifsamelabel = 1
                else:
                    ifsamelabel = -1
                dist = tensor.dot(get_other(neighbor[i]).T, pred_neighbor[i]) - tensor.dot(get_other(fea_single).T,pred)
                loss += ifsamelabel * tensor.dot(dist , dist.T)
        return loss/3


    costs = tensor.tile(modal_cost,(n_samples,1))
    xs = []
    for i in range(recyl_maxlen):
        # set high cost for modal that has been used to prevent predict same modal
        costs = tensor.set_subtensor(costs[tensor.arange(n_samples), next_modal], 1)
        feas, update = theano.scan(fn = get_single,
                            sequences=[x, next_modal],
                            )
        fea_single_n, update_n = theano.scan(fn = get_single,
                            sequences=[x_nerghbors, next_modal_n],
                            )
        fea_single = feas
        max_coefficients_supported = 10000

        xs.append(fea_single)

        [h,c] = get_layer(options['encoder'])[1](tparams, fea_single, options,
                                                prefix=options['encoder'],
                                                mask=mask,h_before = h,c_before = c)
        [h_n,c_n] = get_layer(options['encoder'])[1](tparams, fea_single_n, options,
                                                prefix=options['encoder'],
                                                mask=mask_n,h_before = h_n,c_before = c_n)
        proj = h
        proj_n = h_n
        projs.append(proj)
        projsmatrix = tensor.stack(projs)
        proj_pred = tensor.stack(projs) * tensor.stack(masks)[:, :, None]
        proj_pred = tensor.transpose(proj_pred,(1,0,2))
        proj_pred = tensor.reshape(proj_pred,[projsmatrix.shape[1],projsmatrix.shape[0] * projsmatrix.shape[2]])
        # print('h_n.shape', h_n.shape)
        if options['use_dropout']:
            proj_pred = dropout_layer(proj_pred, use_noise, trng)
        pred = tensor.nnet.softmax(tensor.dot(proj_pred, tparams['U_' + str(i)]) + tparams['b_' + str(i)])

        print('i', i)

        f_pred_prob = theano.function([ x, maskl,modal_cost, max_cost], pred,
                                                name='f_pred_prob',on_unused_input='ignore', allow_input_downcast=True)
        f_pred = theano.function([x, maskl, modal_cost, max_cost], pred.argmax(axis=1),
                                                name='f_pred',on_unused_input='ignore', allow_input_downcast=True)
        f_pred_set.append(f_pred)

        off = 1e-8
        if pred.dtype == 'float16':
            off = 1e-6


        pred_seq = tensor.nnet.softmax(tensor.dot(proj, tparams['U_seq_' + str(i)]) + tparams['b_seq_' + str(i)])
        pred_seq_n = tensor.nnet.softmax(tensor.dot(proj_n, tparams['U_seq_' + str(i)]) + tparams['b_seq_' + str(i)])

        f_pred_seq = theano.function([x, maskl, modal_cost,max_cost], pred_seq.argmax(axis=1),
                        name='f_pred_seq',on_unused_input='ignore',allow_input_downcast=True)


        f_pred_seq_set.append(f_pred_seq)

        pred_seq_index = pred_seq.argmax(axis=1)
        next_modal = pred_seq_index
        next_modal_n = pred_seq_n.argmax(axis=1)
        next_mean.append(next_modal)
        cost1_vector = tensor.log(pred[tensor.arange(n_samples), y] + off)
        cost1 = ( cost1_vector * mask).sum() / (mask.sum() + 1)

        pred_seq_n3 = tensor.reshape(pred_seq_n, [n_samples , neigh_num, maxlen])
        result_loss2, update = theano.scan(fn = compute_dist,
                            sequences=[x3, pred_seq_n3, x, pred_seq, mask, y, y2],
                            )
        cost2 = result_loss2.mean()
        cost3 = (costs * pred_seq).mean()
        cost1_mean.append(cost1)
        cost2_mean.append(cost2)
        cost3_mean.append(cost3)
        lamda1 = 0.001
        lamda2 = 0.1
        if i == recyl_maxlen - 1:
            lamda1 = 0.000000001
            lamda2 = 0.000000001
        cost += -cost1 +  lamda1 * cost2 + lamda2 * cost3
        # cost += -cost1
        # f_fea_other = theano.function([x, x3, y, maskl, modal_cost, max_cost],[nnext, D,cost1,cost2,cost3,mask.sum(),next_modal, fea_single, fea_other, fea_single3, fea_other3], on_unused_input='ignore')
        # f_fea_other_set.append(f_fea_other)
        result, update = theano.scan(lambda b,a: a[b],
                                sequences = pred_seq_index,
                                non_sequences = modal_cost )
        if i == 0:
            cost_vector = result
        else:
            cost_vector += result
        # mask the instance if its cost larger than max_cost
        choice = tensor.nonzero(tensor.gt(-cost_vector,-max_cost))[0]
        mask = tensor.zeros_like(x[:,0], dtype = config.floatX)
        mask = theano.tensor.set_subtensor(mask[choice],1.)
        masks.append(mask)
        if i < recyl_maxlen:
            cost -= (2 * (1-mask) * cost1_vector).sum() / (mask.sum() + 1)
        else:
            cost -= cost1
    f_fea_other = theano.function([x, x3,y2, y, maskl, modal_cost, max_cost],
            [tensor.stack(cost1_mean), tensor.stack(cost2_mean), tensor.stack(cost3_mean)], on_unused_input='ignore')

    return use_noise, x,x3, y2, maskl, y, cost,modal_cost,max_cost,f_pred_set,f_pred_seq_set, f_fea_other


def pred_error(maxlen, f_pred_set,f_pred_seq_set, data, iterator,maskl, model_len, modal_cost, max_cost,verbose=False):
    """
    Just compute the error
    f_pred: Theano fct computing the prediction
    prepare_data: usual prepare_data for that dataset.
    """
    valid_err = 0
    length = 0
    meancost = 0
    for _, valid_index in iterator:
        feax = [data[0][t] for t in valid_index]
        feax = numpy.array(feax)
        targets = [data[1][t] for t in valid_index]
        n_samples = len(valid_index)
        mask = numpy.ones_like(feax[:,0], dtype= 'int32')
        next_modal = numpy.zeros_like(feax[:,0], dtype= 'int32')
        cost_vector = numpy.zeros(n_samples)
        next_modals = numpy.zeros([n_samples,maxlen])
        mask_matrix = numpy.zeros([n_samples, maxlen])
        preds_s = numpy.zeros([n_samples, maxlen])
        preds = numpy.zeros(n_samples)
        costs = numpy.tile(modal_cost,(n_samples,1))
        for i in range(recyl_maxlen):
            cost_vector += costs[range(n_samples), next_modal]
            costs[range(n_samples), next_modal] = 0
            mask_matrix[:,i] = mask
            next_modal = f_pred_seq_set[i](feax, maskl, modal_cost, max_cost)
            next_modals[:,i] = next_modal
            mask = numpy.zeros_like(feax[:,0], dtype= 'int32')
            choice = numpy.greater(-cost_vector,-max_cost)
            mask[choice] = 1
            preds_s[:,i] = f_pred_set[i](feax, maskl, modal_cost, max_cost)
        meancost += cost_vector.sum()
        for i in range(n_samples):
            prem = False
            for j in range(maxlen):
                if mask_matrix[i,j] == 0:
                    prem = True
                    preds[i] = preds_s[i, j-1]
                    break
            if prem == False:
                preds[i] =  preds_s[i, -1]
        valid_err += (preds == targets).sum()
        length += len(preds)
    valid_err = 1. - numpy_floatX(valid_err) / length

    return valid_err, meancost / length


def get_neighbor(sam_traininx, neighinx, trainx, trainy):
    neighs = []
    neiy = []
    trainy = numpy.array(trainy)
    for i in range(len(sam_traininx)):
        ninx = neighinx[sam_traininx[i]]
        neighs.append(trainx[ninx,:])
        neiy.append(trainy[ninx])
    return numpy.array(neighs), numpy.array(neiy)

def train_lstm(
    dim_proj=36,  # word embeding dimension and LSTM number of hidden units.
    patience=15,  # Number of epoch to wait before early stop if no progress
    max_epochs=5000,  # The maximum number of epoch to run
    dispFreq=10,  # Display to stdout the training progress every N updates
    decay_c=0.,  # Weight decay for the classifier applied to the U weights.
    lrate=0.01,  # Learning rate for sgd (not used for adadelta and rmsprop)
    optimizer=adadelta,  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
    #  optimizer = sgd,
    encoder='lstm',  # TODO: can be removed must be lstm.
    saveto='lstm_model_best.npz',  # The best model will be saved there
    validFreq=100,  # Compute the validation error after this number of update.
    saveFreq=1110,  # Save the parameters after every saveFreq updates
    maxlen=4,  # Sequence longer then this get ignored
    batch_size=64,  # The batch size during training.
    valid_batch_size=256,  # The batch size used for validation/test set.
    dataset=datanamee,
    modal_costs = [0.1, 0.1, 0.1, 0.1], # the cost for each modal
    # model_lens = model_len,
    # Parameter for extra option
    noise_std=0.,
    use_dropout=True,  # if False slightly faster, but worst test error
                       # This frequently need a bigger model.
    reload_model=None,  # Path to a saved model we want to start from.
    test_size=-1,  # If >0, we keep only this number of test example.
    max_costs = 50, # max cost for each instance to use
):

    # Model options
    model_options = locals().copy()
    print("model options", model_options)

    load_data = get_dataset(dataset)

    print('Loading data')
    train, valid, test, mask, model_len, train_n3 = load_data()
    '''
    train: (train data, train label)
    valid: (valid data, valid label)
    test: (test data, test label)
    mask: mask each modal's feature
    model_len: each model's length
    train_n3: each traing data's 3 neighbor's index
    '''
    ydim = numpy.max(train[1]) + 1
    #
    model_options['ydim'] = ydim
    print('ydim', ydim)
    print('numpy.min(train[1])', numpy.min(train[1]))
    print('Building model')
    # This create the initial parameters as numpy ndarrays.
    # Dict name (string) -> numpy ndarray
    params = init_params(model_options)
    if reload_model:
        load_params('lstm_model.npz', params)

    # This create Theano Shared Variable from the parameters.
    # Dict name (string) -> Theano Tensor Shared Variable
    # params and tparams have different copy of the weights.
    tparams = init_tparams(params)
    (use_noise, x, x3, y2, mask, y, cost,
            modal_cost,max_cost,f_pred_set, f_pred_seq_set, f_fea_other) = build_model(tparams, model_options)
    if decay_c > 0.:
        decay_c = theano.shared(numpy_floatX(decay_c), name='decay_c')
        weight_decay = 0.
        weight_decay += (tparams['U'] ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay
    print('starting compute grad...')
    print(time.asctime( time.localtime(time.time()) ) )
    grads = tensor.grad(cost, wrt=list(tparams.values()))
    f_grad = theano.function([mask, y, x,x3, y2, modal_cost,max_cost], grads, name='f_grad',on_unused_input='ignore')
    lr = tensor.scalar(name='lr')
    print('starting optimizer')
    print(time.asctime( time.localtime(time.time()) ) )
    f_grad_shared, f_update = optimizer(lr, tparams, grads,
                                        x, x3, y2, mask, y, cost, modal_cost,max_cost)
    kf_valid = get_minibatches_idx(len(valid[0]), valid_batch_size)
    kf_test = get_minibatches_idx(len(test[0]), valid_batch_size)

    print("%d train examples" % len(train[0]))
    print("%d valid examples" % len(valid[0]))
    print("%d test examples" % len(test[0]))

    history_errs = []
    best_p = None
    bad_count = 0

    if validFreq == -1:
        validFreq = len(train[0]) // batch_size
    if saveFreq == -1:
        saveFreq = len(train[0]) // batch_size

    uidx = 0  # the number of update done
    estop = False  # early stop
    start_time = time.time()
    try:
        for eidx in range(max_epochs):
            n_samples = 0

            # Get new shuffled index for the training set.
            kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=False)

            for _, train_index in kf:
                feax3, feay2 = get_neighbor(train_index, train_n3, train[0], train[1])
                '''
                feax3: this mini-batch's data's feature
                feay2: this mini-batch's data's label
                '''
                uidx += 1
                use_noise.set_value(1.)

                # Select the random examples for this minibatch
                y = [train[1][t] for t in train_index]
                feax = [train[0][t] for t in train_index]
                n_samples += len(feax)
                maskl = masklist(model_len, dim_proj)

                cost = f_grad_shared(maskl, y, feax, feax3,feay2, modal_costs,max_costs)
                f_update(lrate)
                if numpy.isnan(cost) or numpy.isinf(cost):
                    print('bad cost detected: ', cost)
                    return 1., 1., 1.

                if numpy.mod(uidx, dispFreq) == 0:
                    print('Epoch ', eidx, 'Update ', uidx, 'Cost ', cost)

                if saveto and numpy.mod(uidx, saveFreq) == 0:
                    print('Saving...')

                    if best_p is not None:
                        params = best_p
                    else:
                        params = unzip(tparams)
                    numpy.savez(saveto, history_errs=history_errs, **params)
                    pickle.dump(model_options, open('%s.pkl' % saveto, 'wb'), -1)
                    print('Done')

                if numpy.mod(uidx, validFreq) == 0:
                    use_noise.set_value(0.)
                    maskls = masklist(model_len, dim_proj)
                    train_err, tracost = pred_error(maxlen, f_pred_set, f_pred_seq_set, train, kf,maskls, model_len, modal_costs, max_costs)
                    print('~~~~~~~~~~~train_err',train_err)
                    print('~~~~~~~~~~~tracost',tracost)
                    kf_valid = get_minibatches_idx(len(valid[0]), valid_batch_size)
                    kf_test = get_minibatches_idx(len(test[0]), valid_batch_size)
                    valid_err, valcost = pred_error(maxlen, f_pred_set, f_pred_seq_set, valid, kf_valid,maskls, model_len, modal_costs, max_costs)
                    test_err, tstcost = pred_error(maxlen, f_pred_set, f_pred_seq_set, test, kf_test,maskls, model_len, modal_costs, max_costs)
                    history_errs.append([valid_err, test_err, valcost, tstcost])
                    print('history_errs)', history_errs)
                    if (best_p is None or
                        valid_err <= numpy.array(history_errs)[:,
                                                               0].min()):

                        best_p = unzip(tparams)
                        bad_counter = 0

                    print( ('Train ', train_err, 'Valid ', valid_err,
                           'Test ', test_err) )
                    print( ('Traincost ', tracost, 'Validcost ', valcost,
                           'Testcost ', tstcost) )

                    if (len(history_errs) > patience and
                        valid_err >= numpy.array(history_errs)[:-patience,
                                                               0].min()):
                        bad_counter += 1
                        if bad_counter > patience:
                            print('Early Stop!')
                            estop = True
                            break

            print('Seen %d samples' % n_samples)

            if estop:
                break

    except KeyboardInterrupt:
        print("Training interupted")

    end_time = time.time()
    if best_p is not None:
        zipp(best_p, tparams)
    else:
        best_p = unzip(tparams)
    print('best valid',numpy.array(history_errs)[:,
                                           0].min())
    print('best valid',numpy.array(history_errs)[:,
                                           1].min())
    use_noise.set_value(0.)
    kf_train_sorted = get_minibatches_idx(len(train[0]), batch_size)
    train_err, tracost = pred_error(maxlen, f_pred_set, f_pred_seq_set, train, kf_train_sorted,maskls, model_len, modal_costs, max_costs)

    kf_valid = get_minibatches_idx(len(valid[0]), valid_batch_size)
    kf_test = get_minibatches_idx(len(test[0]), valid_batch_size)
    valid_err, valcost = pred_error(maxlen, f_pred_set, f_pred_seq_set, valid, kf_valid,maskls, model_len, modal_costs, max_costs)

    test_err,tstcost = pred_error(maxlen, f_pred_set, f_pred_seq_set, test, kf_test,maskls, model_len, modal_costs, max_costs)

    print( 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err )
    print( ('Traincost ', tracost, 'Validcost ', valcost,
           'Testcost ', tstcost) )
    if saveto:
        numpy.savez(saveto, train_err=train_err,
                    valid_err=valid_err, test_err=test_err,
                    history_errs=history_errs, **best_p)
    print('The code run for %d epochs, with %f sec/epochs' % (
        (eidx + 1), (end_time - start_time) / (1. * (eidx + 1))))
    print( ('Training took %.1fs' %
            (end_time - start_time)), file=sys.stderr)
    return train_err, valid_err, test_err

if __name__ == '__main__':
    # See function train for all possible parameter and there definition.
    recyl_maxlen = int(argv[1]) # the max madal can been used for one instance
    train_lstm(
        max_epochs=100,
        test_size=500,
    )
