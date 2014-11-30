# -*-coding:Utf-8 -*

r"""
The theano version of the Auto-Encoder.

Theano is a Python library that allows you to define, optimize, and evaluate mathematical expressions involving multi-dimensional arrays efficiently. Theano features:
    - Tight integration with NumPy: Use numpy.ndarray in Theano-compiled
    functions.
    - Transparent use of a GPU: Perform data-intensive calculations up to 140x
    faster than with CPU.(float32 only)
    - Efficient symbolic differentiation: Theano does your derivatives for
    function with one or many inputs.
    - Speed and stability optimizations: Get the right answer for log(1+x) even
    when x is really tiny.
    - Dynamic C code generation: Evaluate expressions faster.
    - Extensive unit-testing and self-verification: Detect and diagnose many
    types of mistake.

This script define a class 'AutoEncoder'. See below for further description of this class

This script also include a test script performed on MNIST demonstrating how the 'AutoEncoder' object has to be used.

-------------
DEPENDENCIES:
-------------
    Libraries:
        - theano: Download at http://deeplearning.net/software/theano/#download
        - numpy
        - scipy
        - PIL

    Scripts:
        - Tokenizer: Load the dataset
        - Support: Several support function found on DeepLearning.net to work with
        theano
"""

import cPickle
import os
import sys
import time
import copy

import numpy as np
import scipy.optimize

import warnings
warnings.filterwarnings("ignore")

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

import PIL.Image

import Tokenizer as tokenizer

from Support import tile_raster_images


class AutoEncoder(object):
    r"""
    Auto-Encoder class (AutoEncoder) -- Version using Theano
                                        (and GPU if available)

    An  auto-encoder tries to reconstruct the input by projecting it first
    in a latent space (hidden layer) and reprojecting it afterwards back in the
    input space.

    An auto-encoder is a neural network with:
        - an input layer 'inputs'
        - an hidden layer 'hidden_layer_value'
        - a output layer 'reconstruction_layer_value' (reconstruction of 'inputs')

    To avoid learning the identity function -trivial but useless solution- one
    can add constraints on the network. Several regulations methods can be used.
    One can chose to apply:
        - a weight decay constraint: Limit the value of the weights
        - a sparsity constraint: Limit the number of active neurons for each
            example
        - a denoising behavior: try to reconstruct the input from an noisy version
            of it.
        - a contractive constraint: use the jacobian norm as a constraint
        - tied weights: force the network to respect: W2 = transpose(W1)

    The neural net can be trained:
        - Using a first order gradient descent algorithm:
            This is fully implemented using theano and GPU enable computation. It
            is thus much faster than the other solution
        - Using a second order gradient descent algorithm:
            This use the algorithms implemented in the scipy.optimize library.
            Those learning algorithm are supposed to be more efficient in order to
            find the global minimum.
            However this is not fully implemented using theano and GPU computation
            and it is thus slower than the first order methods.


    Please refer to Andrew Ng CS294 Lecture notes "Sparse autoencoder" for more
    details.
    """
    def __init__(self, n_visible, n_hidden, input= None, tied_weight= True,
            encoder_activation_function= 'sigmoid',
            decoder_activation_function= 'sigmoid',
            W1= None, W2= None, b1= None, b2= None,
            np_seed= None, theano_seed= None):
        r"""
        Initialize the AutoEncoder class by specifying the number of
        visible units (ie: dimension of the input), the number of hidden units
        (ie: the dimension of the latent or hidden space ) and the desired average
        activation of the hidden neurons.
        The initial weights of the network are optional

        ----------
        Parameters:
        ----------
        n_visible: int
            The number of neuron in the input layer.
            (ie: the dimension of the input)

        n_hidden: int
            The number of neuron in the hidden layer.
            (ie: the dimension in which you want to project the input)

        input: (optional) theano.tensor.TensorType
            This is a symbolic variable decribing the input of the neural net.
            (ie: one minibatch)

        tied_weight: (optional) Boolean
            If set to true then W2 = W1.T (np.transpose(W1))

        encoder_activation_function: (optional) String
            What is the function used to activate the neurons of the hidden layer
            At the time the options available are:
            'sigmoid', 'tanh' , 'softplus' , 'softmax', 'rectifiedLinear'

        decoder_activation_function: (optional) String
            What is the function used to activate the neurons of the
            reconstruction layer
            At the time the options available are:
            'sigmoid', 'softplus', 'softmax', 'linear'

        W1, W2, b1, b2: (optional) theano.tensor.TensorType
            Weight matrixs to initialize the network

        np_seed: (optional) int or array_like
            Random seed initializing the pseudo-random number generator

        theano_seed: (optional) int or array_like
            ?

        ----------
        Attributs:
        ----------
        self.__initargs__: Pickle object
            Set up the cPickle serialization

        self.n_visible: int
            Initialized with the value of the parameter 'n_visible'

        self.n_hidden: int
            Initialized with the value of the parameter 'n_hidden'

        self tied_weight: Boolean
            Initialized with the value of the parameter 'tied_weight'

        self.encoder_activation_function
            Initialized with the value of the parameter
            'encoder_activation_function'

        self.decoder_activation_function
            Initialized with the value of the parameter
            'decoder_activation_function'

        self.np_rng: numpy.Random.RandomState object
            Initialized with the seed 'np_seed'

        self.theano_rng: theano.Random.RandomState object
            Initialized with the random state generator 'self.np_rng'

        self.W1, W2, b1, b2: theano.tensor.TensorType
            Initialized with W1, W2, b1 and b2 if given.
            If nothing is given
                - W1: W1 is initialized with `initial_W1` which is uniformely
                sampled from -4*sqrt(6./(n_visible+n_hidden)) and
                4*sqrt(6./(n_hidden+n_visible))
                The output of uniform is converted using asarray to dtype
                theano.config.floatX so that the code is runable on GPU
                - W2:
                    - is sampled in the same way as W1 is if self.tied_weight =
                    False
                    - ElseW2 = transpose(W1)
                - b1 and b2: vectors of zeros

        self.inputs: theano.tensor.TensorType
            Initiazed with the given inputs if any or  asa symbolic variable
            decribing the input of the neural net.

        sefl.theta= list of theano.tensor.TensorType
            theta = [W1, W2, b1, b2]
        """
        # Use of cPickle serialization
        self.__initargs__ = copy.copy(locals())
        del self.__initargs__['self']

        # Initialize the parameters of the auto-encoder object
        self.n_visible      = n_visible
        self.n_hidden       = n_hidden
        self.tied_weight    = tied_weight

        # Check if correct activation functions have been used:
        assert encoder_activation_function in set(['sigmoid', 'tanh' ,
            'softplus' , 'softmax', 'rectifiedLinear'])
        assert decoder_activation_function in set(['sigmoid', 'softplus',
            'softmax', 'linear'])
        self.encoder_activation_function = encoder_activation_function
        self.decoder_activation_function = decoder_activation_function

        # Creation of a numpy random generator
        # This will be used to add noise to the input (if required)
        if not np_seed:
            np_rng       = np.random.RandomState(89677)
            self.np_seed = 89677
        else:
            self.np.seed = np_seed #useless self?
            np_rng       = np.random.RandomState(np_seed)
        self.np_rng = np_rng

        # Create a Theano random generator that gives symbolic random values
        # This will be used to add noise to the input (if required)
        if not theano_seed:
            theano_rng = RandomStreams(self.np_rng.randint(2**30))
            self.theano_seed = 2**30 #useless self?
        else:
            self.theano_seed = theano_seed
            theano_rng = RandomStreams(self.np_rng.randint(theano_seed))
        self.theano_rng = theano_rng

        if W1 == None:
        # W1 is initialized with `W1_init` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # The output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
            W1_init = np.asarray(np_rng.uniform(
                            low= -np.sqrt(6.) / np.sqrt(n_visible + n_hidden),
                            high= np.sqrt(6) / np.sqrt(n_visible + n_hidden),
                            size= (n_visible, n_hidden)),
                        dtype= theano.config.floatX)
            if self.encoder_activation_function == 'sigmoid':
                W1_init *= 4

            W1 = theano.shared(value= W1_init, name= 'W1', borrow= True)

        if self.tied_weight == False:
            if W2 == None:
                W2_init = np.asarray(np_rng.uniform(
                            low= -np.sqrt(6.) / np.sqrt(n_visible + n_hidden),
                            high= np.sqrt(6) / np.sqrt(n_visible + n_hidden),
                            size= (n_hidden, n_visible)),
                            dtype= theano.config.floatX)
                if self.decoder_activation_function == 'sigmoid':
                    W2_init *= 4

                W2 = theano.shared(value= W2_init, name= 'W2', borrow= True)

        if b1 == None:
            b1 = theano.shared(value= np.zeros(n_hidden,
                                    dtype=theano.config.floatX),
                                borrow= True)
        if b2 == None:
            b2 = theano.shared(value= np.zeros(n_visible,
                                    dtype=theano.config.floatX),
                                borrow= True)

        self.W1 = W1

        if self.tied_weight == True:
            self.W2 = self.W1.T
        else:
            self.W2 = W2

        self.b1 = b1
        self.b2 = b2

        if input == None:
            # We use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.inputs = T.fmatrix(name='input')
        else:
            self.inputs = input

        if self.tied_weight == True:
            self.theta = [self.W1, self.b1, self.b2]
        else:
            self.theta = [self.W1, self.W2, self.b1, self.b2]

    ###########################################################
    # Secondary functions used for training the Auto encoder: #
    ###########################################################
    def corrupte_input(self, input, corruption_level, noise= 'masking'):
        r"""
        Corrupte input method
        This method keeps `1-corruption_level` entries of the inputs the same
        and zero-out randomly selected subset of size `coruption_level`

        ----------
        Inputs:
        ----------
        input: theano.tensor.TensorType
            This is a symbolic variable decribing the input of the neural net.
            (ie: one minibatch)

        corruption_level: float
            Level of noise to be applied to the input

        noise: string
            Type of noise to be applied to the input. ('masking' or 'whiteNoise')

        ----------
        Outputs:
        ----------
        corrupted_input: theano.tensor.TensorType
            This is a symbolic variable decribing the new input of the neural net.
            (ie: one minibatch)
        """
        if noise == 'masking':
            return self.theano_rng.binomial(size= input.shape, n= 1,
                    p= 1- corruption_level, dtype=theano.config.floatX) * input
        elif noise == 'white':
            return input + self.theano_rng.normal(size = input.shape, avg= 0,
                    std= corruption_level, dtype= theano.config.floatX)
        else:
            raise NotImplementedError("This noise {0}".format(noise), " is not implemented")


    def encoding_pass(self, input):
        """
        Compute the value of the hidden layer given an input.

        ----------
        Inputs:
        ----------
        input: theano.tensor.TensorType
            This is a symbolic variable decribing the input of the neural net.
            (ie: one minibatch)

        ----------
        Outputs:
        ----------
        hidden_layer_value: theano.tensor.TensorType
            This is a symbolic variable decribing the value of the neuron in the
            hidden layer of the neural net given an input.
        """

        if self.encoder_activation_function == 'sigmoid':
            return T.nnet.sigmoid(T.dot(input, self.W1) + self.b1)

        elif self.encoder_activation_function == 'tanh':
            return T.tanh(T.dot(input, self.W1) + self.b1)

        elif self.encoder_activation_function == 'softplus':
            return T.nnet.softplus(T.dot(input, self.W1) + self.b1)

        elif self.encoder_activation_function == 'softmax':
            return T.nnet.softmax(T.dot(input, self.W1) + self.b1)

        elif self.encoder_activation_function == 'rectifiedLinear':
            def rectifiedLinear(x):
                return x * (x>0)
            return rectifiedLinear(T.dot(input, self.W1) + self.b1)

        else:
            raise NotImplementedError("The encoding function {0}".format(encoder_activation_function), " is not implemented")


    def decoding_pass(self, hidden_value):
        """
        Compute the value of the reconstruction layer given an hidden layer.

        ----------
        Inputs:
        ----------
        hidden_value: theano.tensor.TensorType
            This is a symbolic variable decribing the hidden value of the
            neural net.

        ----------
        Outputs:
        ----------
        reconstructed_layer_value: theano.tensor.TensorType
            This is a symbolic variable decribing the value of the neuron in the
            output layer of the neural net given an input.
        """
        if self.decoder_activation_function == 'sigmoid':
            return T.nnet.sigmoid(T.dot(hidden_value, self.W2) + self.b2)

        elif self.decoder_activation_function == 'softplus':
            return T.nnet.softplus(T.dot(hidden_value, self.W2) + self.b2)

        elif self.decoder_activation_function == 'softmax':
            return T.nnet.softmax(T.dot(hidden_value, self.W2) + self.b2)

        elif self.decoder_activation_function == 'linear':
             return T.dot(hidden_value, self.W2) + self.b2
        else:
            raise NotImplementedError("The decoding function {0}".format(self.decoder_activation_function), " is not implemented")


    def jacobian_computation(self, hidden_layer_value, batch_size, W):
        r"""
        Computes the jacobian of the hidden layer with respect to
        the input, reshapes are necessary for broadcasting the
        element-wise product on the right axis

        ----------
        Inputs:
        ----------
        hidden_ayer_value: theano.tensor.TensorType
            This is a symbolic variable decribing the hidden value of the
            neural net.

        batch_size: int
            Size of the minibatch

        W: theano.tensor.TensorType
            The jacobian is computed with regard to this weight matrix

        ----------
        Outputs:
        ----------
        frob_norm_jacobian_matrix: theano.tensor.TensorType
            The frobenius norm of the jacobian matrix of the hidden layer value
            wrt to W
            The general expression of this is:
            frob_norm(J) = sum_i(f'(a_i) * sum_j(W_ij))
        """
        if self.encoder_activation_function == 'sigmoid':
            # f_{sig}' = f * (1 - f)
            return T.reshape(hidden_layer_value * (1 - hidden_layer_value),
                (batch_size, 1 , self.n_hidden)) \
                     * T.reshape(W, (1, self.n_visible, self.n_hidden))

        elif self.encoder_activation_function == 'tanh':
            # f_{tanh}' = (1 - f²)
            return T.reshape((1 - hidden_layer_value ** 2),
                (batch_size, 1 , self.n_hidden)) \
                     * T.reshape(W, (1, self.n_visible, self.n_hidden))

        elif self.encoder_activation_function == 'softplus':
            # f_{softplus}' = f_{sigmoid}
            return T.reshape(T.nnet.sigmoid(hidden_layer_value),
                (batch_size, 1 , self.n_hidden)) \
                     * T.reshape(W, (1, self.n_visible, self.n_hidden))

        elif self.encoder_activation_function == 'softmax':
            # f_{softmax}' = f(x_j) * (1 - f(x_i)) if i = j
            #                -f(x_i) * f(x_j) if i != j
            return  (-hidden_layer_value * hidden_layer_value) * \
                        (T.ones_like(hidden_layer_value) \
                            - T.identity_like(hidden_layer_value)) \
                      + (hidden_layer_value * (1 - hidden_layer_value)) * \
                                   T.identity_like(hidden_layer_value)

        elif self.encoder_activation_function == 'rectifiedLinear':
            # f_{RL}' = 1 if x_i > 0, 0 otherwise
            return T.gt(hidden_layer_value, 0)

        else:
            raise NotImplementedError("The contrative option for the encoding function {0}".format(encoder_activation_function), " is not implemented.")


    def to_vector(self, theta):
        r"""
        Transform the list of shared variable theta into a np.array. This will be
        used for the scipy optimization.

        ----------
        Inputs:
        ----------
        theta: list(theano.tensor.TensorType)
            List of tensor to be transformed

        ----------
        Outputs:
        ----------
        np_theta: np.array
            Theta flatten under the np format
        """
        theta_vector = []
        for thet in theta:
            theta_vector.append(thet.get_value(borrow=True).flatten())
        return np.concatenate(theta_vector).astype(dtype= theano.config.floatX)


    def to_share_value(self, theta_value):
        r"""
        Transform an np.array theta_value into a list of shared variable. This
        will be used for the scipy optimization.

        ----------
        Inputs:
        ----------
        np_theta: np.array
            Flatten np.array with the correct dimension (can be used to reform a
            self.theta like list)

        ----------
        Outputs:
        ----------
        theta: list(theano.tensor.TensorType)
            List of self.theta like
        """

        if self.tied_weight == True:
            # Check if the input vector as the good dimension to recreate theta
            assert(theta_value.shape[0] == (self.n_visible + 1) * \
                                                (self.n_hidden + 1) -1)

            limit_0 = 0
            limit_1 = self.n_visible * self.n_hidden
            limit_2 = self.n_visible * self.n_hidden + self.n_hidden
            limit_3 = (self.n_visible + 1) * (self.n_hidden + 1) - 1

            self.W1.set_value(theta_value[limit_0:limit_1].reshape(
                              self.n_visible, self.n_hidden).astype(
                                                    dtype= theano.config.floatX),
                              borrow=True)

            self.b1.set_value(theta_value[limit_1:limit_2].reshape(
                              self.n_hidden).astype(dtype= theano.config.floatX),
                              borrow=True)

            self.b2.set_value(theta_value[limit_2:limit_3].reshape(
                              self.n_visible).astype(dtype= theano.config.floatX),
                              borrow=True)

            self.theta = [self.W1, self.b1, self.b2]

        else:
            # Check if the input vector as the good dimension to recreate theta
            assert(theta_value.shape[0] == 2 * self.n_visible * self.n_hidden \
                                           + self.n_hidden + self.n_visible)

            limit_0 = 0
            limit_1 = self.n_visible * self.n_hidden
            limit_2 = 2 * self.n_visible * self.n_hidden
            limit_3 = 2 * self.n_visible * self.n_hidden + self.n_hidden
            limit_4 = 2*self.n_visible*self.n_hidden+self.n_hidden+self.n_visible

            self.W1.set_value(theta_value[limit_0:limit_1].reshape(
                              self.n_visible, self.n_hidden).astype(
                                        dtype= theano.config.floatX),
                              borrow=True)
            self.W2.set_value(theta_value[limit_1:limit_2].reshape(
                              self.n_hidden, self.n_visible).astype(
                                        dtype= theano.config.floatX),
                              borrow=True)
            self.b1.set_value(theta_value[limit_2:limit_3].reshape(
                              self.n_hidden).astype(dtype= theano.config.floatX),
                              borrow=True)
            self.b2.set_value(theta_value[limit_3:limit_4].reshape(
                              self.n_visible).astype(dtype= theano.config.floatX),
                              borrow=True)

            self.theta = [self.W1, self.W2, self.b1, self.b2]


    ################################################
    # Training function: Unsupervised pre-training #
    ################################################
    def backpropagation(self, **kwargs):
        r"""
        Back-propagation over the auto-encoder method
        Perform a backpropagtion to update the weight of the network given
        the network, the state of its parameters and input values.

        ----------
        Inputs:
        ----------
        learning_rate: float
            What is the size of the step done for each step of the gradient
            descent

        batch_size: int
            Size of the minibatch

        ---- DENOISING ----
        corruption_level: (optional) float
            Level of noise to be applied to the input

        noise: (optional) string
            Type of noise to be applied to the input. ('masking' or 'whiteNoise')

        ---- ERROR FUNCTION ----
        error_function: string ('squared' or 'CE')
            Whether you want to use a square error or the cross-entropy loss

        ---- REGULARIZATION ----
        lamda: float
            The weight decay parameter (will influence the cost function).
            The weigh decay parameter lamda control the relative importance of
            the regularization term over the others in the final cost.

        ---- SPARSITY ----
        beta: float
            The weight sparsity parameter (will influence the cost function).
            It control the relative importance of the sparsity penalty term in
            the cost function over the other in the final cost.

        rho: float
            The desired average activation of the hidden units (will influence
            the cost function).
            Typically rho is a small value close to zero (say rho = 0.05).

        ---- CONTRASTIVE ----
        delta: float
            The weight of the contractive parameter (will influence the cost
            function).
            The contractive parameter deta control the relative importance of
            the contractive term over the others in the final cost.

        ---- SCIPY USAGE ----                   ---------------------
        scipy_opt: (optional) Boolean           ----- NOT READY -----
            Wheter you want to use a classique gradient descent (set to 'False')
            or a more robust optimization method ('L-BFGS-B') implemented in Scipy
            WARNING: Using Scipy imply to transfer data from GPU to CPU and could
            lead to a slow down in computation

        ----------
        Outputs:
        ----------
        cost: float
            What is the cost of the reconstruction for this input

        if scipy_opt == True:
            gradient:
                Gradient of the cost function
        else:
            updates:
                Updated parameters
        """
        #############################
        # BACKPROPAGATION ALGORITHM #
        #############################
        # to calculate the partial derivative
        # (see Andrew Ng CS294A sparse autoencoder p9)

        # 1) Perform the feed-forward pass (on the noised input)
        if kwargs['corruption_level'] != 0.:
            input_corrupted = self.corrupte_input(self.inputs,
                    corruption_level= kwargs['corruption_level'],
                            noise= kwargs['noise'])
            hidden_layer_value = self.encoding_pass(input_corrupted)
        else:
            hidden_layer_value = self.encoding_pass(self.inputs)

        reconstructed_layer_value = self.decoding_pass(hidden_layer_value)

        # 1-bis) Compute the cost function J:
        if kwargs['error_function'] == 'CE':
            if self.encoder_activation_function == 'tanh':
                # Reshaping tanh from [-1;1] to [0;1]
                self.loss = - T.sum(
                        ((self.inputs + 1) / 2) * T.log(reconstructed_layer_value)
                        +(1 -((self.inputs + 1)/2)) \
                            * T.log(1-reconstructed_layer_value),
                        axis=1)
            else:
                self.loss = - T.sum(
                        self.inputs * T.log(reconstructed_layer_value) +
                        (1 - self.inputs) * T.log(1 - reconstructed_layer_value),
                        axis=1)

        elif kwargs['error_function'] == 'squared':
            self.loss = T.sum((self.inputs - reconstructed_layer_value)**2,
                    axis=1)
        else:
            raise NotImplementedError("This cost function {0}".format(kwargs['error_function']), " is not implemented")

        J = self.loss

        # ---- Regularization term
        if kwargs['lamda'] != 0.:
            lamda = kwargs['lamda']

            self.weight_decay = 0.5 * lamda * (T.sum(self.W1 ** 2)
                    + T.sum(self.W2 **2))
            # Update of J for the regularized auto encoder:
            J += self.weight_decay # see Andrew Ng CS294A sparse autoencoder p15
        else:
            self.weight_decay = 0.

        # ---- Sparsity term
        if kwargs['beta'] != 0:
            # Average activation of hidden unit
            beta = kwargs['beta']
            rho = kwargs['rho']

            rho_hat = T.sum(hidden_layer_value, axis= 1) / self.inputs.shape[1]

            self.KL_divergence = beta * T.sum(rho * T.log(rho / rho_hat) \
                    + (1 - rho) * T.log((1 - rho)/(1 - rho_hat)))
                        # see Andrew Ng CS294A sparse autoencoder p15

            # Update of J for the sparse auto encoder:
            J += self.KL_divergence # see Andrew Ng CS294A sparse autoencoder p15
        else:
            self.KL_divergence = 0.

        # ---- Contractive term
        if kwargs['delta'] != 0:
            delta = kwargs['delta']

            jacobian_matrix = self.jacobian_computation(hidden_layer_value,
                    kwargs['batch_size'], self.W1)
            self.contractive_reg = delta * T.sum(jacobian_matrix**2) / \
                                        kwargs['batch_size']

            # Update of J for the contractive auto encoder:
            J += self.contractive_reg
        else:
            self.contractive_reg = 0.

        # At the end: J = reconstruction_error + weight_decay
        #        + KL_divergence + contractive_penalty
        # J is a vector where each element is the cost of the reconstruction
        # of an example of the minibatch.
        # Computing the average on those bach give us the cost of the minibatch
        cost = T.mean(J)


        # Using theano gradient method allow us to skip step 2 and 3 (compute
        # the erro term for the output layer and for the hidden layer)
        # see Andrew Ng CS294A sparse autoencoder p9

        if kwargs['scipy_opt'] == True:
            return cost

        else:
            # 4) Compute the gradient of theta
            grad_theta = T.grad(cost, self.theta)

            # Generate the list of updates
            updates = []
            for thet, grad_thet in zip(self.theta, grad_theta):
                updates.append((thet, thet - kwargs['learning_rate'] * grad_thet))

            return (cost, updates)


    def train_AE(self, **kwargs):
        r"""
        Training an Auto-Encoder method
        Perform a gradient descent on the cost of the reconstruction with regard
        to theta.

        ----------
        Inputs:
        ----------
        test_set:

        learning_rate: float
            What is the size of the step done for each step of the gradient
            descent

        tau_learning: (Optional) int or None
            If tau is not 'None' then we use a decreasing learning rate schedule.
            We then have:
                    learning_rae = initial_learning_rate x tau_learning rate \
                                    / max(turn^(alpha_learning),tau_learning_rate)

        alpha_learning: float
            Speed of decrease of the learning rate

        epochs: int
            How many time does we go through the train_set

        batch_size: int
            Size of the minibatch

        ---- DENOISING ----
        corruption_level: (optional) float
            Level of noise to be applied to the input

        noise: (optional) string
            Type of noise to be applied to the input. ('masking' or 'whiteNoise')

        ---- ERROR FUNCTION ----
        error_function: string ('squared' or 'CE')
            Whether you want to use a square error or the cross-entropy loss

        ---- REGULARIZATION ----
        lamda: float
            The weight decay parameter (will influence the cost function).
            The weigh decay parameter lamda control the relative importance of
            the regularization term over the others in the final cost.

        ---- SPARSITY ----
        beta: float
            The weight sparsity parameter (will influence the cost function).
            It control the relative importance of the sparsity penalty term in
            the cost function over the other in the final cost.

        rho: float
            The desired average activation of the hidden units (will influence
            the cost function).
            Typically rho is a small value close to zero (say rho = 0.05).

        ---- CONTRASTIVE ----
        delta: float
            The weight of the contractive parameter (will influence the cost
            function).
            The contractive parameter deta control the relative importance of
            the contractive term over the others in the final cost.
        """
        ###########################
        # Simple gradient descent #
        ###########################
        if kwargs['lr_dic']['scipy_opt'] == False:
            # Compute number of minibatches for training, validation and testing
            batch_size = kwargs['lr_dic']['batch_size']

            n_batches = kwargs['train_set'].get_value(borrow=True).shape[0] / \
                            batch_size
            # Allocate symbolic variables for the data
            index = T.iscalar()    # index to a [mini]batch
            inputs = T.matrix('inputs')  # data presented as rasterized images

            # Allocate a symbolic variable for the parameters
            # This will allow to make them vary during the learning
            # Learning rate:
            lr_rate = T.fscalar("lr_rate")
            # Corruption level:
            cor_lev = T.fscalar("cor_lev")
            # L2 regularisation:
            lamda_param = T.fscalar("lamda_param")
            # Sparsity:
            beta_param = T.fscalar("beta_param")
            # Contractive:
            delta_param = T.fscalar("delta_param")

            batch_cost, updates = self.backpropagation(
                                scipy_opt= kwargs['lr_dic']['scipy_opt'],
                                learning_rate= lr_rate,
                                batch_size= batch_size,
                                corruption_level= cor_lev ,
                                noise= kwargs['reg_dic']['noise'],
                                error_function= kwargs['error_function'],
                                lamda= lamda_param, beta= beta_param,
                                rho= kwargs['reg_dic']['rho'],
                                delta= delta_param)

            # Create the input list
            inputs = [ index, theano.Param(lr_rate, default= 0.1),
                          theano.Param(cor_lev, default= 0.0),
                          theano.Param(lamda_param, default= 0.0),
                          theano.Param(beta_param, default= 0.0),
                          theano.Param(delta_param, default= 0.0)]

            # Set the parameters value:
            pretrain_lr     = kwargs['lr_dic']['learning_rate']
            pretrain_cor    = kwargs['reg_dic']['corruption_level']
            pretrain_lamda  = kwargs['reg_dic']['lamda']
            pretrain_beta   = kwargs['reg_dic']['beta']
            pretrain_delta  = kwargs['reg_dic']['delta']

            # Create the output list:
            output_list = [T.mean(self.loss)]
            if pretrain_lamda != 0.:
                output_list.append(self.weight_decay)
            if pretrain_beta != 0.:
                output_list.append(self.KL_divergence)
            if pretrain_delta !=0.:
                output_list.append(self.contractive_reg)

            train_AE = theano.function(
                            inputs= inputs,
                            outputs= output_list,   #[batch_cost],
                            updates= updates,
                            givens= {self.inputs:
                            kwargs['train_set'][index * batch_size:(index+1) \
                                            * batch_size]},
                            name= 'train_AE',
                            on_unused_input='ignore')

            start = time.clock()

            # TRAINING:
            cost = []
            if __name__ == '__main__':
                print("Training the autoencoder...")
            # Go through training epochs
            for epoch in xrange(kwargs['lr_dic']['epochs']):
                tic = time.clock()
                # Go through the training set
                cost_vector = []
                for batch_index in xrange(int(n_batches)):

                    # Decreasing learning rate:
                    if kwargs['lr_dic']['tau_learning'] != None:
                        pretrain_lr = kwargs['lr_dic']['learning_rate'] * \
                                        kwargs['lr_dic']['tau_learning'] \
                                        / max(pow(batch_index,
                                            kwargs['lr_dic']['alpha_learning']),
                                        kwargs['lr_dic']['tau_learning'])

                    cost_vector.append(train_AE(batch_index,
                                                lr_rate= pretrain_lr,
                                                cor_lev= pretrain_cor,
                                                lamda_param= pretrain_lamda,
                                                beta_param= pretrain_beta,
                                                delta_param= pretrain_delta))


                # Average cost on a batch:
                cost.append(np.mean(cost_vector)) #cost_vector[0]
                toc = time.clock()
                print("Epoch %d: training time %.2fm, Reconstruction cost: %.2f"\
                        %(epoch, (toc - tic)/60., cost[-1]))

            end = time.clock()

            training_time = (end - start)

            if __name__ == '__main__':
                print ('The code for file ' + os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((training_time) / 60.))
        ##################################
        # END of simple gradient descent #
        ##################################


        ######################
        # Scipy optimization #
        ######################
        else:
            # Compute number of minibatches for training, validation and testing
            batch_size = kwargs['lr_dic']['batch_size']

            n_batches = kwargs['train_set'].get_value(borrow=True).shape[0] / \
                        batch_size

            # Allocate symbolic variables for the data
            index = T.iscalar()    # index to a [mini]batch
            inputs = T.matrix('inputs')  # data presented as rasterized images

            # Allocate a symbolic variable for the parameters
            # This will allow to make them vary during the learning
            # Learning rate:
            lr_rate = T.fscalar("lr_rate")
            # Corruption level:
            cor_lev = T.fscalar("cor_lev")
            # L2 regularisation:
            lamda_param = T.fscalar("lamda_param")
            # Sparsity:
            beta_param = T.fscalar("beta_param")
            # Contractive:
            delta_param = T.fscalar("delta_param")


            cost = self.backpropagation(
                                scipy_opt= kwargs['lr_dic']['scipy_opt'],
                                batch_size= batch_size,
                                corruption_level= cor_lev ,
                                noise= kwargs['reg_dic']['noise'],
                                error_function= kwargs['error_function'],
                                lamda= lamda_param, beta= beta_param,
                                rho= kwargs['reg_dic']['rho'],
                                delta= delta_param)

            # Create the input list
            inputs = [ index, theano.Param(cor_lev, default= 0.0),
                          theano.Param(lamda_param, default= 0.0),
                          theano.Param(beta_param, default= 0.0),
                          theano.Param(delta_param, default= 0.0)]

            # Set the parameters value:
            pretrain_cor    = kwargs['reg_dic']['corruption_level']
            pretrain_lamda  = kwargs['reg_dic']['lamda']
            pretrain_beta   = kwargs['reg_dic']['beta']
            pretrain_delta  = kwargs['reg_dic']['delta']

            # Create the output list:
            output = T.mean(self.loss)
            if pretrain_lamda != 0.:
                output += self.weight_decay
            if pretrain_beta != 0.:
                output += self.KL_divergence
            if pretrain_delta !=0.:
                output += self.contractive_reg

            # Compile a thenao function that returns the cost of a minibatch
            batch_cost = theano.function(inputs= inputs,
                                         outputs= output,
                                         givens= {self.inputs:
                                         kwargs['train_set'][index * batch_size:\
                                                 (index+1) * batch_size]},
                                         name= 'batch_cost',
                                         on_unused_input='ignore')

            # Compile the theano function that returns the gradient of a minibatch
            # with respect to theta
            batch_grad = theano.function(inputs= inputs,
                                         outputs= T.grad(cost, self.theta),
                                         givens= {self.inputs:
                                         kwargs['train_set'][index * batch_size:\
                                                 (index+1) * batch_size]},
                                         name= 'batch_grad',
                                         on_unused_input='ignore')

            # Define a python function that compute the cost and the gradient for
            # given theta (under numpy format)
            def train_fn(theta_value):
                r"""
                This function take as an input a self.theta like variable
                flattened using the function 'to_vecor' and compute the average
                cost of the reconstruction for this theta and the average
                gradient of the cost regarding to each element of theta.
                All the output are converted into numpy.array in order to be used
                by scipy.optimize.minimize()

                ----------
                Inputs:
                ----------
                theta_value: np.array
                Variable self.theta like flattened using the 'to_vector' function

                ----------
                Outputs:
                ----------
                cost: np.float64
                Cost of he reconstruction

                gradient: np.array
                Flatten np.array describing the gradient of the cost regarding to
                theta
                """
                hip = time.clock()

                # Update self.theta
                self.to_share_value(theta_value)

                # Learning parameters:
                pretrain_cor    = kwargs['reg_dic']['corruption_level']
                pretrain_lamda  = kwargs['reg_dic']['lamda']
                pretrain_beta   = kwargs['reg_dic']['beta']
                pretrain_delta  = kwargs['reg_dic']['delta']

                # Compute the cost and the gradient for every batch
                cost_vect = []
                for batch_index in xrange(int(n_batches)):

                    # Cost of the minibatch
                    cost = batch_cost(batch_index,
                                  cor_lev= pretrain_cor,
                                  lamda_param= pretrain_lamda,
                                  beta_param= pretrain_beta,
                                  delta_param= pretrain_delta)

                    cost_vect.append(cost)

                    # Gradient of the minibatch
                    gradient = batch_grad(batch_index,
                                          cor_lev= pretrain_cor,
                                          lamda_param= pretrain_lamda,
                                          beta_param= pretrain_beta,
                                          delta_param= pretrain_delta)

                    # Converting gradient and its elements into np.arrays for
                    # compatibilities
                    gradient = np.asarray(gradient)

                    for i, elmt_1 in enumerate(gradient):
                        elmt_1 = np.asarray(elmt_1).flatten()
                        gradient[i] = elmt_1

                    if batch_index == 0:
                        grad_sum = np.array(np.concatenate(gradient),
                                                            dtype= np.float64)
                    else:
                        grad_sum += np.concatenate(gradient)

                # Average over the minibatch:
                cost = np.float64(np.mean(np.asarray(cost_vect)))
                grad_theta = np.array(grad_sum / n_batches)

                hop = time.clock()

                print("  Minimize: Loop over the test_set (%.2fm - cost: %.2f)") \
                                    %((hop-hip)/60., cost)

                return cost, grad_theta

            start = time.clock()

            # TRAINING:
            if __name__ == '__main__':
                print("Training the autoencoder...")

            for epoch in xrange(kwargs['lr_dic']['epochs']):
                tic = time.clock()

                opt_solution = scipy.optimize.minimize(train_fn,
                    self.to_vector(self.theta),
                    args = (),
                    method = kwargs['lr_dic']['optim_method'],
                    jac = True,
                    options = {'maxiter': kwargs['lr_dic']['max_iter']})

                if __name__ == '__main__':
                    print opt_solution.message

                cost = opt_solution.fun

                toc = time.clock()

                print("Epoch %d: training time %.2fm, Reconstruction cost: %.2f"\
                        %(epoch, (toc - tic)/60., cost))
                print("Number of iteration over the train_set: %i") \
                                %(opt_solution.nfev)

        #############################
        # End of scipy optimization #
        #############################

            end = time.clock()
            training_time = (end - start)
            if __name__ == '__main__':
                print ('The code for file ' + os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((training_time) / 60.))

    #############################
    # Post-processig functions: #
    #############################
    def vizualize_learning(self, name= 'AE_filters'):
        image = PIL.Image.fromarray(tile_raster_images(
            X=self.W1.get_value(borrow=True).T,
            img_shape=(28, 28), tile_shape=(10, 10),
            tile_spacing=(1, 1)))

        image.save(name + '.png')


    def reconstruct(self, input):
        r"""
        Reconstruction of input method
        Given an input and a network computes the output

        ----------
        Inputs:
        ----------
        input: theano.tensor.TensorType
            This is a symbolic variable decribing the input of the neural net.

        ----------
        Outputs:
        ----------
        reconstructed_layer: theano.tensor.TensorType
            The value of the reconstruction for all the input

        recconstructed_error: float
        """
        # Perform the feed-forward pass for testing:
        reconstructed_layer_value = self.decoding_pass(
                            self.encoding_pass(input))

        n_row_input = input.get_value(borrow=True).shape[0]
        n_column_input = input.get_value(borrow=True).shape[1]

        # If the input is a column
        if n_row_input == 0:
            error = T.sum(T.sum(abs(reconstructed_layer_value - input)))\
                    / (n_column_input) * 100
        # If the input is a row
        elif n_column_input == 0:
            error = T.sum(T.sum(abs(reconstructed_layer_value - input)))\
                    / (n_row_input) * 100
        # If the input is a matrix
        else:
            error = T.sum(T.sum(abs(reconstructed_layer_value - input)))\
                    / (n_row_input * n_column_input) * 100

        return reconstructed_layer_value, error


#########
# Test: #
#########
if __name__ == '__main__':

    ############################################
    # SETTING the parameters of the experience #
    ############################################
    print(" ")
    print "-----------------PARAMETERS---------------------"

    print "---- THEANO AUTO-ENCODER: "
    print "-------- Geometry of the network: "
    n_visible = 28 * 28
    print "     number of visible units :       {0}".format(n_visible)
    hidden_row = 50                 # Number of line in the visualization
    hidden_col = 10                 # Number of column in the visualization
    n_hidden = hidden_row * hidden_col
    print "     number of hidden units :        {0}".format(n_hidden)
    print "-------- Activation functions: "
    encoding_acti_fun = 'sigmoid'   # sigmoid, tanh, softmax, softplus,
                                    # rectifiedLinear
    print "     encoding_acti_fun :             {0}".format(encoding_acti_fun)
    decoding_acti_fun = 'sigmoid'   # sigmoid, softplus, softmax, linear
    print "     decoding_acti_fun :             {0}".format(decoding_acti_fun)
    print "-------- Weights: "
    tied_weight = True              # Set to true for W2 = W1^T
    print "     tied_weight :                   {0}".format(tied_weight)

    print(" ")
    print "---- LEARNING: "
    batch_size = 10
    print "    batch_size :                     {0}".format(batch_size)
    # Scipy optimisation?
    scipy_opt = True
    print "    scipy_opt :                      {0}".format(scipy_opt)

    # Parameters requiered for using scipy optimization:
    if scipy_opt == True:
        optim_method = 'CG'         # L-BFGS-B, CG, Newton-CG... (see scipy doc)
        print "    optim_method :                   {0}".format(optim_method)
        max_iter = 400              # Maximum number of evaluation of the cost
                                    # function
        print "    max_iter :                       {0}".format(max_iter)
        epochs = 1                  # No real need for epochs in this case
        print "    epochs :                         {0}".format(epochs)

    # Parameters requiered for using the simple theano optimization
    else:
        epochs = 20
        print "    epochs :                         {0}".format(epochs)
        learning_rate = 0.1         # Initial learning rate
        print "    learning_rate :                  {0}".format(learning_rate)
        tau_learning = None         # Set it to an int if you want the learning
                                    # rate to decrease over time
        print "    tau_learning :                   {0}".format(tau_learning)
        alpha_learning =    1.5     # Speed of decrease
        print "    alpha_learning :                 {0}".format(alpha_learning)

    print(" ")
    print "---- COST: "
    error_function = 'squared'      # CE, squared
    print "    error_function:                  {0}".format(error_function)
    print(" ")
    print "---- REGULARIZATION: "
    print "-------- Noise: "
    noise = 'masking'               # masking, white
    print "    noise :                          {0}".format(noise)
    corruption_level = 0.4
    print "    corruption_level :               {0}".format(corruption_level)
    print "-------- L2 regularisation: "
    lamda = 0.01
    print "    lamda :                          {0}".format(lamda)
    print "-------- Sparsity: "
    beta = 0.3
    print "    beta :                           {0}".format(beta)
    rho = 0.1
    print "    rho :                            {0}".format(rho)
    print "-------- Contractive: "
    delta = 0.3
    print "    delta :                          {0}".format(delta)

    print(" ")
    print "---- PLOTTING:"
    plot_name= 'test_sig_sig_CG'
    print "        name :                       {0}".format(plot_name)

    print(" ")

    # Argument dictionaries:
    # Creating the appropriate 'learning dictionary' requiered for creating
    # a SAE

    if scipy_opt == True:
        lr_dic= {'scipy_opt': scipy_opt, 'optim_method': optim_method,
                 'batch_size': batch_size, 'epochs': epochs, 'max_iter': max_iter}
    else:
        lr_dic= {'scipy_opt': scipy_opt, 'batch_size': batch_size,
                       'epochs': epochs, 'learning_rate': learning_rate,
                       'tau_learning': tau_learning,
                       'alpha_learning': alpha_learning}

    reg_dic = {'corruption_level': corruption_level, 'noise': noise,
               'lamda': lamda, 'beta': beta, 'rho': rho, 'delta': delta}
    ################################################
    # END setting the parameters of the experience #
    ################################################

    ###########################
    # Using the Auto-Encoder: #
    ###########################
    # Load the dataset
    print("Load dataset...")
    datasets = tokenizer.load_mnist()
    train_set_x, train_set_y = datasets[0]
    test_set_x, test_set_y = datasets[2]

    # Instantiation of an autoEncoder object:
    theano_AE= AutoEncoder(n_visible= n_visible, n_hidden= n_hidden,
            input= None, tied_weight= tied_weight,
            encoder_activation_function= encoding_acti_fun,
            decoder_activation_function= decoding_acti_fun,
            W1= None, W2= None, b1= None, b2= None,
            np_seed= None, theano_seed= None)


    # Training the autoEncoder
    theano_AE.train_AE(train_set= train_set_x, error_function= error_function,
                       lr_dic= lr_dic, reg_dic= reg_dic)

    # Plotting the results:
    print("Plotting...")
    theano_AE.vizualize_learning(plot_name)

    reconstructed_layer_value, error = theano_AE.reconstruct(test_set_x)

    print("The error of reconstruction is:  {0}".format(error.eval()), "%")












