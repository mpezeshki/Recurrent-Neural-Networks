"""

Gated recurrent units (Long Short-Term Memory)
Code provided by Mohammad Pezeshki - Nov. 2014 - Universite de Montreal
This code is distributed without any warranty, express or implied.

"""

import numpy as np
import theano
import theano.tensor as T

class LSTM(object):
    def __init__(self, n_u, n_h):

        self.n_u = int(n_u)
        self.n_h = int(n_h)

        # Here we define 8 weight matrices and 4 bias vector.
        # To determine the dimension of weight matrices and bias vectors,
        # we just need to have the following numbers: n_u, n_h
        # (You can check it using the equations).
        #
        # Thus:
        # <var>: <dimention>
        #
        # W_xi : n_h  x  n_u
        # W_hi : n_h  x  n_h
        # W_ci : n_h  x  n_h
        # W_xf : n_h  x  n_u
        # W_hf : n_h  x  n_h
        # W_cf : n_h  x  n_h
        # W_xc : n_h  x  n_u
        # W_hc : n_h  x  n_h
        #
        # b_i : n_h  x  1 #Note: do not confuse b_i as the bias of input! It's the bias of input GATE
        # b_f : n_h  x  1
        # b_o : n_h  x  1
        # b_c : n_h  x  1

        # Input gate weights
        self.W_xi = theano.shared(value = np.asarray(
                                              np.random.uniform(
                                                  size = (n_h, n_u),
                                                  low = -.01, high = .01),
                                              dtype = theano.config.floatX),
                                  name = 'W_xi')
        self.W_hi = theano.shared(value = np.asarray(
                                              np.random.uniform(
                                                  size = (n_h, n_h),
                                                  low = -.01, high = .01),
                                              dtype = theano.config.floatX),
                                  name = 'W_hi')
        self.W_ci = theano.shared(value = np.asarray(
                                              np.random.uniform(
                                                  size = (n_h, n_h),
                                                  low = -.01, high = .01),
                                              dtype = theano.config.floatX),
                                  name = 'W_ci')

        # Forget gate weights
        self.W_xf = theano.shared(value = np.asarray(
                                              np.random.uniform(
                                                  size = (n_h, n_u),
                                                  low = -.01, high = .01),
                                              dtype = theano.config.floatX),
                                  name = 'W_xf')
        self.W_hf = theano.shared(value = np.asarray(
                                              np.random.uniform(
                                                  size = (n_h, n_h),
                                                  low = -.01, high = .01),
                                              dtype = theano.config.floatX),
                                  name = 'W_hf')
        self.W_cf = theano.shared(value = np.asarray(
                                              np.random.uniform(
                                                  size = (n_h, n_h),
                                                  low = -.01, high = .01),
                                              dtype = theano.config.floatX),
                                  name = 'W_cf')

        # Output gate weights
        self.W_xo = theano.shared(value = np.asarray(
                                              np.random.uniform(
                                                  size = (n_h, n_u),
                                                  low = -.01, high = .01),
                                              dtype = theano.config.floatX),
                                  name = 'W_xo')
        self.W_ho = theano.shared(value = np.asarray(
                                              np.random.uniform(
                                                  size = (n_h, n_h),
                                                  low = -.01, high = .01),
                                              dtype = theano.config.floatX),
                                  name = 'W_ho')
        self.W_co = theano.shared(value = np.asarray(
                                              np.random.uniform(
                                                  size = (n_h, n_h),
                                                  low = -.01, high = .01),
                                              dtype = theano.config.floatX),
                                  name = 'W_co')

        # Cell weights
        self.W_xc = theano.shared(value = np.asarray(
                                              np.random.uniform(
                                                  size = (n_h, n_u),
                                                  low = -.01, high = .01),
                                              dtype = theano.config.floatX),
                                  name = 'W_xc')
        self.W_hc = theano.shared(value = np.asarray(
                                              np.random.uniform(
                                                  size = (n_h, n_h),
                                                  low = -.01, high = .01),
                                              dtype = theano.config.floatX),
                                  name = 'W_hc')

        # Input gate bias
        self.b_i = theano.shared(value = np.zeros(
                                             (n_h, ),
                                             dtype = theano.config.floatX),
                                 name = 'b_i')

        # Forget gate bias
        self.b_f = theano.shared(value = np.zeros(
                                             (n_h, ),
                                             dtype = theano.config.floatX),
                                 name = 'b_f')

        # Output gate bias
        self.b_o = theano.shared(value = np.zeros(
                                             (n_h, ),
                                             dtype = theano.config.floatX),
                                 name = 'b_o')

        # cell bias
        self.b_c = theano.shared(value = np.zeros(
                                             (n_h, ),
                                             dtype = theano.config.floatX),
                                 name = 'b_c')

        self.params = [self.W_xi, self.W_hi, self.W_ci,
                          self.W_xf, self.W_hf, self.W_cf,
                          self.W_xo, self.W_ho, self.W_co,
                          self.W_xc, self.W_hc,
                          self.b_i, self.b_f, self.b_o,
                          self.b_c]


    def lstm_as_activation_function(self, x_t, h_tm1, c_tm1):
        #print self.W_xi.get_value(borrow = True)
        i_t = T.nnet.sigmoid(T.dot(self.W_xi, x_t) + \
                             T.dot(self.W_hi, h_tm1) + \
                             T.dot(self.W_ci, c_tm1) + \
                             self.b_i)
        f_t = T.nnet.sigmoid(T.dot(self.W_xf, x_t) + \
                             T.dot(self.W_hf, h_tm1) + \
                             T.dot(self.W_cf, c_tm1) + \
                             self.b_f)
        c_t = f_t * c_tm1 + i_t * \
                  T.tanh(T.dot(self.W_xc, x_t) + \
                         T.dot(self.W_hc, h_tm1) + \
                         self.b_c)
        o_t = T.nnet.sigmoid(T.dot(self.W_xo, x_t) + \
                        T.dot(self.W_ho, h_tm1) + \
                        T.dot(self.W_co, c_t) + \
                        self.b_o)
        h_t = o_t * T.tanh(c_t)

        return h_t, c_t

# Gated Recurrent Unit
# Recently proposed: 
# K. Cho, B. van Merrienboer,
# D. Bahdanau, and Y. Bengio.
# On the properties of neural 
# machine translation: Encoder-decoder approaches. 
class GRU(object):
    def __init__(self, n_u, n_h):

        self.n_u = int(n_u)
        self.n_h = int(n_h)

        # Here we define 6 weight matrices and 3 bias vector.
        # To determine the dimention of weight matrices and bias vectors,
        # we just need to have the following numbers: n_u, n_h
        # (You can check it using the equations).
        #
        # Thus:
        # <var>: <dimention>
        #
        # W_xz : n_h  x  n_u
        # W_hz : n_h  x  n_h
        # W_xr : n_h  x  n_u
        # W_hr : n_h  x  n_h
        # W_xh : n_h  x  n_h
        # W_hh : n_h  x  n_h
        #
        # b_z : n_h  x  1
        # b_r : n_h  x  1
        # b_h : n_h  x  1

        # Update gate weights
        self.W_xz = theano.shared(value = np.asarray(
                                              np.random.uniform(
                                                  size = (n_h, n_u),
                                                  low = -.01, high = .01),
                                              dtype = theano.config.floatX),
                                  name = 'W_xz')
        self.W_hz = theano.shared(value = np.asarray(
                                              np.random.uniform(
                                                  size = (n_h, n_h),
                                                  low = -.01, high = .01),
                                              dtype = theano.config.floatX),
                                  name = 'W_hz')

        # Reset gate weights
        self.W_xr = theano.shared(value = np.asarray(
                                              np.random.uniform(
                                                  size = (n_h, n_u),
                                                  low = -.01, high = .01),
                                              dtype = theano.config.floatX),
                                  name = 'W_xr')
        self.W_hr = theano.shared(value = np.asarray(
                                              np.random.uniform(
                                                  size = (n_h, n_h),
                                                  low = -.01, high = .01),
                                              dtype = theano.config.floatX),
                                  name = 'W_hr')

        # Other weights :-)
        self.W_xh = theano.shared(value = np.asarray(
                                              np.random.uniform(
                                                  size = (n_h, n_u),
                                                  low = -.01, high = .01),
                                              dtype = theano.config.floatX),
                                  name = 'W_xh')
        self.W_hh = theano.shared(value = np.asarray(
                                              np.random.uniform(
                                                  size = (n_h, n_h),
                                                  low = -.01, high = .01),
                                              dtype = theano.config.floatX),
                                  name = 'W_hh')

        # Update gate bias
        self.b_z = theano.shared(value = np.zeros(
                                             (n_h, ),
                                             dtype = theano.config.floatX),
                                 name = 'b_z')

        # Reset gate bias
        self.b_r = theano.shared(value = np.zeros(
                                             (n_h, ),
                                             dtype = theano.config.floatX),
                                 name = 'b_r')

        # Hidden layer bias
        self.b_h = theano.shared(value = np.zeros(
                                             (n_h, ),
                                             dtype = theano.config.floatX),
                                 name = 'b_h')

        self.params = [self.W_xz, self.W_hz, self.W_xr, self.W_hr, 
                          self.W_xh, self.W_hh, self.b_z, self.b_r, 
                          self.b_h]


    def gru_as_activation_function(self, x_t, h_tm1):
        # update gate
        z_t = T.nnet.sigmoid(T.dot(self.W_xz, x_t) + \
                             T.dot(self.W_hz, h_tm1) + \
                             self.b_z)
        # reset gate
        r_t = T.nnet.sigmoid(T.dot(self.W_xr, x_t) + \
                             T.dot(self.W_hr, h_tm1) + \
                             self.b_r)
        # candidate h_t
        can_h_t = T.tanh(T.dot(self.W_xh, x_t) + \
                         r_t * T.dot(self.W_hh, h_tm1) + \
                         self.b_h)
        # h_t
        h_t = (1 - z_t) * h_tm1 + z_t * can_h_t

        return h_t
