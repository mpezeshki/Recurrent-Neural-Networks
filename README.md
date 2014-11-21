Recurrent-Neural-Networks
=========================

Here's an RNN which is used for three kinds of output: real-valued, binary, and softmax.
And for five kinds of activation function: sigmoid, tanh, relu, lstm, gru.
To run the code you need to have Theano libary in your PYTHONPATH:

https://github.com/Theano/Theano

The results are then saved under *.png files: <test>_<activation function used>_<number of epochs>.png
To test the model with different hyper-parameters, you need to modify any of testing function.

Running code with default parameters takes around 5 minutes on CPU.

Related resources
=================
Graham Taylor's implementation:

https://github.com/gwtaylor/theano-rnn


Razvan Pascanu's implementation:

https://github.com/pascanur/trainingRNNs


Alex Grave's paper with a nice description of RNNs:

http://arxiv.org/pdf/1308.0850v5.pdf


Yoshua Bengio, Aaron Courville, and Ian Goodfellow book:

Deep Learning - Chapter 12

Notice
======
This code is distributed without any warranty, express or implied.
