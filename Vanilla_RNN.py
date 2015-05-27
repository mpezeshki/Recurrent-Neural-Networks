import numpy as np
import theano
import theano.tensor as T
import time
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ion()

mode = theano.Mode(linker='cvm')
floatX = theano.config.floatX


@theano.compile.ops.as_op(itypes=[T.fmatrix],
                          otypes=[T.fmatrix, T.fmatrix, T.fmatrix])
def svd(x):
    P, D, Q = np.linalg.svd(x, full_matrices=False)
    return P, D, Q


class RNN(object):
    def __init__(self, n_u, n_h, n_y, activation, output_type,
                 learning_rate, learning_rate_decay, l1_reg, l2_reg,
                 initial_momentum, final_momentum, momentum_switchover,
                 n_epochs):

        self.n_u = int(n_u)
        self.n_h = int(n_h)
        self.n_y = int(n_y)

        if activation == 'linear':
            self.activation = lambda x: x
        elif activation == 'tanh':
            self.activation = T.tanh
        elif activation == 'sigmoid':
            self.activation = T.nnet.sigmoid
        elif activation == 'relu':
            self.activation = lambda x: x * (x > 0)
        else:
            raise NotImplementedError

        self.output_type = output_type
        self.learning_rate = float(learning_rate)
        self.learning_rate_decay = float(learning_rate_decay)
        self.l1_reg = float(l1_reg)
        self.l2_reg = float(l2_reg)
        self.initial_momentum = float(initial_momentum)
        self.final_momentum = float(final_momentum)
        self.momentum_switchover = int(momentum_switchover)
        self.n_epochs = int(n_epochs)

        # input which is `x`
        self.x = T.matrix()

        # weights are initialized from an uniform distribution
        self.W_uh = theano.shared(
            value=np.asarray(np.random.uniform(
                size=(n_u, n_h),
                low=-.01, high=.01),
                dtype=floatX),
            name = 'W_uh')

        self.W_hh = theano.shared(
            value=np.asarray(
                np.random.uniform(
                    size=(n_h, n_h),
                    low=-.01, high=.01),
                dtype=floatX),
            name = 'W_hh')

        self.W_hy = theano.shared(
            value=np.asarray(
                np.random.uniform(
                    size=(n_h, n_y),
                    low=-.01, high=.01),
                dtype=floatX),
            name = 'W_hy')

        # initial value of hidden layer units are set to zero
        self.h0 = theano.shared(
            value=np.zeros((n_h, ), dtype=floatX),
            name='h0')

        # biases are initialized to zeros
        self.b_h = theano.shared(
            value=np.zeros((n_h, ), dtype=floatX),
            name='b_h')

        self.b_y = theano.shared(
            value=np.zeros((n_y, ), dtype=floatX),
            name='b_y')

        self.params = [self.W_uh, self.W_hh, self.W_hy, self.h0,
                       self.b_h, self.b_y]

        # Initial value for updates is zero matrix.
        self.updates = {}
        for param in self.params:
            self.updates[param] = theano.shared(
                value=np.zeros(param.get_value(
                    borrow=True).shape,
                    dtype=floatX),
                name='updates')

        # h_t = g(W_uh * u_t + W_hh * h_tm1 + b_h)
        # y_t = W_yh * h_t + b_y
        def recurrent_fn(u_t, h_tm1):
            h_t = (self.activation(T.dot(u_t, self.W_uh) +
                                   T.dot(h_tm1, self.W_hh) +
                                   self.b_h))
            y_t = T.dot(h_t, self.W_hy) + self.b_y
            return h_t, y_t

        [self.h, self.y_pred], _ = theano.scan(recurrent_fn,
                                               sequences=self.x,
                                               outputs_info=[self.h0, None])

        # L1 norm
        self.L1 = (abs(self.W_uh.sum()) +
                   abs(self.W_hh.sum()) +
                   abs(self.W_hy.sum()))

        # square of L2 norm
        self.L2_sqr = ((self.W_uh ** 2).sum() +
                       (self.W_hh ** 2).sum() +
                       (self.W_hy ** 2).sum())

        if self.output_type == 'real':
            self.y = T.matrix(name='y', dtype=floatX)
            self.loss = lambda y: self.mse(y)
            self.predict = theano.function(inputs=[self.x, ],
                                           outputs=self.y_pred,
                                           mode=mode,
                                           allow_input_downcast=True)

        elif self.output_type == 'binary':
            self.y = T.matrix(name='y', dtype='int32')
            self.p_y_given_x = T.nnet.sigmoid(self.y_pred)
            self.y_out = T.round(self.p_y_given_x)  # round to {0,1}
            self.loss = lambda y: self.nll_binary(y)
            self.predict_proba = theano.function(inputs=[self.x, ],
                                                 outputs=self.p_y_given_x,
                                                 mode=mode)
            self.predict = theano.function(inputs=[self.x, ],
                                           outputs=T.round(self.p_y_given_x),
                                           mode=mode)

        elif self.output_type == 'softmax':
            self.y = T.vector(name='y', dtype='int32')
            self.p_y_given_x = T.nnet.softmax(self.y_pred)
            self.y_out = T.argmax(self.p_y_given_x, axis=-1)
            self.loss = lambda y: self.nll_multiclass(y)
            self.predict_proba = theano.function(inputs=[self.x, ],
                                                 outputs=self.p_y_given_x,
                                                 mode=mode)
            self.predict = theano.function(inputs=[self.x, ],
                                           outputs=self.y_out,
                                           mode=mode)
        else:
            raise NotImplementedError

        # Just for tracking training error for Graph 3
        self.errors = []
        self.monitor = []

    def mse(self, y):
        # mean is because of minibatch
        return T.mean((self.y_pred - y) ** 2)

    def nll_binary(self, y):
        # negative log likelihood here is cross entropy
        return T.mean(T.nnet.binary_crossentropy(self.p_y_given_x, y))

    def nll_multiclass(self, y):
        # notice to [  T.arange(y.shape[0])  ,  y  ]
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def build_trian(self, x_train, y_train, x_test=None, y_test=None):
        train_set_x = theano.shared(np.asarray(x_train, dtype=floatX))
        train_set_y = theano.shared(np.asarray(y_train, dtype=floatX))
        if self.output_type in ('binary', 'softmax'):
            train_set_y = T.cast(train_set_y, 'int32')

        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print 'Buiding model ...'

        index = T.lscalar('index')    # index to a case
        # learning rate (may change)
        lr = T.scalar('lr', dtype=floatX)
        mom = T.scalar('mom', dtype=floatX)  # momentum

        cost = self.loss(self.y) \
            + self.l1_reg * self.L1 \
            + self.l2_reg * self.L2_sqr

        compute_train_error = theano.function(inputs=[index, ],
                                              outputs=self.loss(self.y),
                                              givens={
                                                  self.x: train_set_x[index],
                                                  self.y: train_set_y[index]},
                                              mode=mode)

        gparams = []
        for param in self.params:
            gparams.append(T.grad(cost, param))

        # zip just concatenate two lists
        updates = {}
        for param, gparam in zip(self.params, gparams):
            weight_update = self.updates[param]
            upd = mom * weight_update - lr * gparam
            updates[weight_update] = upd
            updates[param] = param + upd

        train_model = theano.function(inputs=[index, lr, mom],
                                      outputs=cost,
                                      updates=updates,
                                      givens={
                                          self.x: train_set_x[index],
                                          self.y: train_set_y[index]},
                                      mode=mode)

        monitor_svd = theano.function(inputs=[],
                                      outputs=svd(self.W_hh))

        ###############
        # TRAIN MODEL #
        ###############
        print 'Training model ...'
        epoch = 0
        n_train = train_set_x.get_value(borrow=True).shape[0]

        while (epoch < self.n_epochs):
            epoch = epoch + 1
            for idx in xrange(n_train):
                effective_momentum = self.final_momentum \
                    if epoch > self.momentum_switchover \
                    else self.initial_momentum
                train_model(idx,
                            self.learning_rate,
                            effective_momentum)

            # compute loss on training set
            train_losses = [compute_train_error(i)
                            for i in xrange(n_train)]
            this_train_loss = np.mean(train_losses)
            self.errors.append(this_train_loss)

            print('epoch %i, train loss %f ''lr: %f' %
                  (epoch, this_train_loss, self.learning_rate))
            self.monitor.append(monitor_svd()[1])

            self.learning_rate *= self.learning_rate_decay


def test_real():
    print 'Testing model with real outputs'
    n_u = 3
    n_h = 4
    n_y = 3
    time_steps = 15
    n_seq = 100

    np.random.seed(0)

    # generating random sequences
    seq = np.random.randn(n_seq, time_steps, n_u)
    targets = np.zeros((n_seq, time_steps, n_y))

    targets[:, 1:, 0] = seq[:, :-1, 0]
    targets[:, 2:, 1] = seq[:, :-2, 1]
    targets[:, 3:, 2] = seq[:, :-3, 2]

    targets += 0.01 * np.random.standard_normal(targets.shape)

    model = RNN(n_u=n_u, n_h=n_h, n_y=n_y,
                activation='linear', output_type='real',
                learning_rate=0.001, learning_rate_decay=0.999,
                l1_reg=0, l2_reg=0,
                initial_momentum=0.5, final_momentum=0.9,
                momentum_switchover=5,
                n_epochs=4000)

    model.build_trian(seq, targets)

    plt.close('all')
    plt.figure()

    # Graph 1
    ax1 = plt.subplot(411)
    plt.plot(seq[0])
    plt.grid()
    ax1.set_title('Input sequence')

    # Graph 2
    ax2 = plt.subplot(412)
    true_targets = plt.plot(targets[0])

    guess = model.predict(seq[0])
    guessed_targets = plt.plot(guess, linestyle='--')
    plt.grid()
    for i, x in enumerate(guessed_targets):
        x.set_color(true_targets[i].get_color())
    ax2.set_title('solid: true output, dashed: model output')

    # Graph 3
    ax3 = plt.subplot(413)
    plt.plot(model.errors)
    plt.grid()
    ax3.set_title('Training error')

    # Graph 4
    ax3 = plt.subplot(414)
    concat = np.vstack([D for D in model.monitor]).T
    D = concat
    DD = np.zeros((D.shape[0] * 5, D.shape[1]))
    for i in range(D.shape[0]):
        DD[5 * i, :] = D[i, :]
        DD[5 * i + 1, :] = D[i, :]
        DD[5 * i + 2, :] = D[i, :]
        DD[5 * i + 3, :] = D[i, :]
        DD[5 * i + 4, :] = D[i, :]
    plt.imshow(DD, interpolation='nearest')

    # Save as a file
    plt.savefig('real2.png')


if __name__ == "__main__":
    t0 = time.time()
    test_real()
    print "Elapsed time: %f" % (time.time() - t0)
