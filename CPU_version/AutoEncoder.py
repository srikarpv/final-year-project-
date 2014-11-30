# -*-coding:Utf-8 -*

"""
 This script is a first implementation of auto-encoders (AE) using standard python library.
"""

import numpy as np
import math
import scipy.optimize
import time
import matplotlib.pyplot as plt
import time
import os

import ActivationFunction as AF

class AutoEncoder(object):
    """
    Auto-Encoder class (AutoEncoder) -- Version using CPU only

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

    The neural net is train using a gradient descent like algorithm.

    Please refer to Andrew Ng CS294 Lecture notes "Sparse autoencoder" for more
    details.
    """

    def __init__(self, n_visible, n_hidden, acti_fun= 'Sigmoid', tied_weight= False, W1= None, W2= None, b1= None, b2= None):
        """
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

        acti_fun: (optional) String
            What is the function used to activate the neurons
            Those functions are defined in the module 'activationFunction.py'

        tied_weight: (optional) Boolean
            If set to true then W2 = W1.T (np.transpose(W1))

        W1, W2, b1, b2: (optional) matrix
            Weight matrixs to initialize the network

        ----------
        Attributs:
        ----------
        self.n_visible: int
            Initialized with the value of the parameter 'n_visible'

        self.n_hidden: int
            Initialized with the value of the parameter 'n_hidden'

        self tied_weight: Boolean
            Initialized with the value of the parameter 'tied_weight'

        self.limitK: int (K=0,1,2,3,4)
            This parameter is used to split the vector of state theta into
            W1, W2,b1, b2.

        self.theta: np.array(float)
            This vector concatenates the matrix of weights of the network
            W1, W2,b1, b2.
            It is stored as a single vector for the optimization algorithm.

        ----------
        Possible improvements:
        ----------
        - add the possibility to choose your activation function:
            To do so you will have to implement a method to select the correct
            expression to express the derivate (or make it a parameter)

        - add the possibility to choose your optimization method (in the method
        'full_gradient_descent'

        """
        # Initialize the parameters of the auto-encoder object
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        self.tied_weight = tied_weight

        # Initialize the neural network weights randomly
        # W1, W2 values are uniformly sampled within a range [-bound, bound]
        if (W1 == None or W2 == None or b1 == None or b2 == None):
            # W1 is initialized with `W1_init` which is uniformely sampled
            # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
            # for tanh activation function
            # Note : optimal initialization of weights is dependent on the
            #        activation function used (among other things).
            #        For example, results presented in [Xavier10] suggest that you
            #        should use 4 times larger initial weights for sigmoid
            #        compared to tanh
            #        We have no info for other function, so we use the same as
            #        tanh.
            bound = 4 * math.sqrt(6) / math.sqrt(n_visible + n_hidden)
            rand = np.random.RandomState(int(time.time()))

        if W1 == None:
            W1 = np.asarray(rand.uniform(low= -bound, high= bound, size=(n_hidden,n_visible)))

        if W2 == None:
            if self.tied_weight == True:
                W2 = np.transpose(W1)
            else:
                W2 = np.asarray(rand.uniform(low= -bound, high= bound, size=(n_visible, n_hidden)))

        # Biais are initialized to zero
        if b1 == None:
            b1 = np.zeros((n_hidden,1))
        if b2 == None:
            b2 = np.zeros((n_visible,1))

        # Set limits for accessing 'theta' values.
        # theta is a vector containing the parameters of the neural net.
        # (ie: vector( W1, b1, W2, b2) )
        # Note: we need to store it as a vector in order to use as an input
        # in the 'full_gradient_descent' method.
        self.limit0 = 0
        self.limit1 = n_hidden * n_visible
        self.limit2 = 2 * n_hidden * n_visible
        self.limit3 = 2 * n_hidden * n_visible + n_hidden
        self.limit4 = 2 * n_hidden * n_visible + n_hidden + n_visible

        # Create 'theta': (stored as a single vector for the optimization
        # algorithm)
        self.theta = np.concatenate((W1.flatten(), W2.flatten(),b1.flatten(), b2.flatten()))

        # Instantiate the correct activatin function
        if type(acti_fun) == str:
            self.activation_function = eval("AF." + acti_fun)()
        else:
            self.activation_function = eval("AF." + acti_fun[0]
                    + "(" + str(acti_fun[1:])+ ")")

    def corrupte_input(self, input, corruption_level):
        """
        corrupte input method
            This method keeps `1-corruption_level` entries of the inputs the same
            and zero-out randomly selected subset of size `coruption_level`

        ----------
        Inputs:
        ----------
        input: np.array((n_input, n_example), type: float)
            Matrix of the input to be learnt

        corruption_level: float
            Level of noise

        ----------
        Outputs:
        ----------
        corrupted_input: np.array((n_input, n_example),type: float)
            Matrix of the corrupted input to be learnt
        """
        noise = np.random.binomial(size=input.shape, n=1, p=1 - corruption_level)
        return input * noise

    def forward_pass(self, input, W, b):
        """
        Possibility to ad activation_func
        """
        return self.activation_function.function(np.dot(W, input) + b)

    def split_theta(self, theta):
        """
        Method taking theta and spliting it into W1, W2, b1 and b2

        ----------
        Inputs:
        ----------
        theta: np.array(float)
            This vector concatenates the matrix of weights of the network
            W1, W2,b1, b2.

        ----------
        Outputs:
        ----------
        W1, W2, b1, b2
        """
        W1 = theta[self.limit0 : self.limit1].reshape(self.n_hidden,
                                                      self.n_visible)
        if self.tied_weight == True:
            W2 = np.transpose(W1)
        else:
            W2 = theta[self.limit1 : self.limit2].reshape(self.n_visible,
                                                          self.n_hidden)

        b1 = theta[self.limit2 : self.limit3].reshape(self.n_hidden, 1)
        b2 = theta[self.limit3 : self.limit4].reshape(self.n_visible, 1)

        return W1, W2, b1, b2


    def autoencoder_backpropagation(self, theta, input,
                                    error_function = 'Squared', lamda= 0.0001,
                                    beta= 0.1, rho= 0.01, corruption_level= 0.01,
                                    grad_check= False):
        """
        Back propagation over the auto encoder method
        Perform a backpropagtion to update the weight of the network given
        the network, the state of its parameters (theta) and input values.

        ----------
        Inputs:
       ----------
        theta: np.array(type: float)
            This vector concatenate the matrix of weights of the network W1, W2,
            b1, b2.
            Note: theta is declared as an input eventhough its not necessary
            (we could access it as self.theta) to run the optimization algorithm

        input:  np.array((n_input, n_example),type: float)
            Matrix of the input to be learnt

        ---- ERROR FUNCTION ----
        error_function: string ('squared' or 'CE')
            Whether you want to use a square error or the cross-entropy loss

        ---- REGULARIZATION ----
        lamba: float
            The weight decay parameter (will influence the cost function).
            The cost function associated to the network will be made of three
            terms:
                - the average sum-of-squares error
                - the regularization term
                - the sparsity penalty term
            The weigh decay parameter lamba control the relative importance of
            the regularization term over the others.

        ---- SPARSITY ----
        beta: float
            The weight sparsity parameter (will influence the cost function).
            It control the relative importance of the sparsity penalty term in
            the cost function over the two other terms.

        rho: float
            The desired average activation of the hidden units (will influence
            the cost function).
            Typically rho is a small value close to zero (say rho = 0.05).

        ---- DENOISING ----
        corruption_level: float
            The desired noise level to be added to the input before the
            reconstruction of the non-noisy version

        ---- CONTRASTIVE ----

        ----------
        Outputs:
        ----------
        J: float
            Value of the cost associated to this network with those weights and
            this batch of inputs.

        grad_theta: np.array(theta, type: float)
            It is a vector of the same shape as 'theta' describing the
            modification to be made on 'theta'
        """
        # Rebuild W1, W2, b1 and b2 from theta
        W1 , W2, b1, b2 = self.split_theta(theta)

        #############################
        # BACKPROPAGATION ALGORITHM #
        #############################
        # to calculate the partial derivatives
        # (see Andrew Ng CS294A sparse autoencoder p9)

        # 1) Perform the feed-forward pass (on the noised input)
        # ---- Denoising
        if corruption_level != 0:
            # ----> Denoising Auto Encoder
            corrupted_input = self.corrupte_input(input, corruption_level)

            hidden_layer_value = self.forward_pass(corrupted_input, W1, b1)
        # ---- Normal
        else:
            hidden_layer_value = self.forward_pass(input, W1, b1)

        reconstructed_layer_value = self.forward_pass(hidden_layer_value, W2, b2)

        # 1-bis) Compute the cost function J:
        # ---- Error between the 'input' and the reconstruction made using
        # 'corrupted_input'
        error = reconstructed_layer_value - input

        # ---- Reconstruction error:
        # -------- Squared error:
        if error_function == 'Squared':
            reconstruction_error = 0.5 * np.sum(np.multiply(error, error)) \
                    / input.shape[1]
            # see Andrew Ng CS294A sparse autoencoder p6

        # -------- Cross entropy:
        elif error_function == 'CE':
            eps = 1e-10
            reconstruction_error = - np.sum((input * \
                                        np.log(reconstructed_layer_value + eps)\
                                                + (1.-input) * np.log(1.- \
                                                reconstructed_layer_value + eps)))
        else:
            print("### ERROR ### : Unknown error function")

        J = reconstruction_error    # see Andrew Ng CS294A sparse autoencoder p15

        # ---- Regularization term
        if lamda != 0:
            weight_decay = 0.5 * lamda * ( np.sum(np.multiply(W1,W1)) \
                                            + np.sum(np.multiply(W2,W2)))

            # Update of J for the regularized auto encoder:
            J += weight_decay # see Andrew Ng CS294A sparse autoencoder p15

        # ---- Sparsity term
        if beta != 0:
            # Average activation of hidden unit
            rho_hat = np.sum(hidden_layer_value, axis= 1) / input.shape[1]

            KL_divergence = beta * np.sum(rho * np.log(rho/rho_hat)
                                    +(1-rho) * np.log((1-rho)/(1-rho_hat)))
                # see Andrew Ng CS294A sparse autoencoder p15

            # Update of J for the sparse auto encoder:
            J += KL_divergence # see Andrew Ng CS294A sparse autoencoder p15

        # At the end:
        # J = reconstruction_error + weight_decay + KL_divergence
        #       + contractive_penalty

        # 2) Error term for the output layer
        if error_function == 'Squared':
            delta_reconstruction = np.matrix(np.multiply(
                error,
                self.activation_function.derivate( reconstructed_layer_value)))
                # see Andrew Ng CS294A sparse autoencoder p16 and p9
        elif error_function == 'CE':
            delta_reconstruction = error
        else:
            print("### ERROR ### : Unknown error function")

        # 3) Error term for the hidden layer
        if error_function == 'Squared':
            if beta != 0:
                # Trick to include the KL into the back propagation
                KL_gradient = beta * (-(rho/rho_hat) + ((1-rho)/(1-rho_hat)))
                    # see Andrew Ng CS294A sparse autoencoder p16

                delta_hidden = np.multiply(
                        np.dot(np.transpose(W2),delta_reconstruction)
                            + np.transpose(np.matrix(KL_gradient)),
                        self.activation_function.derivate(hidden_layer_value))
                    # see Andrew Ng CS294A sparse autoencoder p16 and p9
            else:
                delta_hidden = np.multiply(
                        np.dot(np.transpose(W2),delta_reconstruction),
                        self.activation_function.derivate(hidden_layer_value))

        elif error_function == 'CE':
            if beta != 0:
                # Trick to include the KL into the back propagation
                KL_gradient = beta * (-(rho/rho_hat) + ((1-rho)/(1-rho_hat)))
                    # see Andrew Ng CS294A sparse autoencoder p16

                delta_hidden = np.dot(np.transpose(W2),delta_reconstruction)\
                                + np.transpose(np.matrix(KL_gradient))
                    # see Andrew Ng CS294A sparse autoencoder p16 and p9
            else:
                delta_hidden = np.dot(np.transpose(W2),delta_reconstruction)

        # 4) Compute the gradient values by averaging partial derivatives
        # Partial derivatives are averages over all training examples
        grad_J_W1 = (np.dot(delta_hidden, np.transpose(input))
                        / input.shape[1]) + lamda * W1
            # see Andrew Ng CS294A sparse autoencoder p12
        grad_J_W2 = (np.dot(delta_reconstruction,
                     np.transpose(hidden_layer_value)) / input.shape[1])\
                    + lamda * W2
            # see Andrew Ng CS294A sparse autoencoder p12

        grad_J_b1 = np.sum(delta_hidden, axis=1) / input.shape[1]
            # see Andrew Ng CS294A sparse autoencoder p12

        grad_J_b2 = np.sum(delta_reconstruction, axis=1) / input.shape[1]
            # see Andrew Ng CS294A sparse autoencoder p12

        # Retransform the matrix into array
        grad_J_W1 = np.array(grad_J_W1)
        grad_J_W2 = np.array(grad_J_W2)
        grad_J_b1 = np.array(grad_J_b1)
        grad_J_b2 = np.array(grad_J_b2)

        # Return the grad_theta in correct shape:
        grad_theta = np.concatenate((grad_J_W1.flatten(),
                                     grad_J_W2.flatten(),
                                     grad_J_b1.flatten(),
                                     grad_J_b2.flatten()))

        # Numerical check of the gradient:
            # see Andrew Ng CS294A sparse autoencoder p10
        if grad_check == True:
            self.gradient_checking(input, grad_theta, rho, lamda, beta)

        # END OF THE BACKPROPAGATION ALGORITHM

        return [J, grad_theta]
        # Ng says p12: need to be able to compute J(theta) -cost- and grad_theta
        # to feed an optimizer better than grad descent

    def full_gradient_descent(self,input, error_function = 'Squared',
                              lamda= 0.0001, beta= 0.1, rho= 0.01,
                              corruption_level= 0.01, grad_check= False,
                              max_iter= 400):
        """
        Full gradient descent 'like' method
        Given an input and an number of iteration this method call an optimization
        algorithm to minimize the cost of the net.
        The optimizatin algorithm used is L-BFGS (as recommand by Ng CS294A sparse
        autoencoder p12). This algorithm is implemented in the scipy library for
        python.

        ----------
        Inputs:
        ----------
        input: np.array((n_input,n_example), type: float)
            Matrix of the input to be learnt

        max_iter: int
            Maximum number of call to the function to optimize

        ----------
        Outputs:
        ----------
        opt_solution:
        The optimization result represented as a OptimizeResult object.
        Important attributes are:
            x the solution array
            success a Boolean flag indicating if the optimizer exited
            successfully
            message which describes the cause of the termination.
            See OptimizeResult for a description of other attributes.
        """
        opt_solution = scipy.optimize.minimize(self.autoencoder_backpropagation,
                self.theta,
                args = (input, error_function, lamda, beta, rho,
                            corruption_level, grad_check),
                method = 'L-BFGS-B',
                jac = True, options = {'maxiter': max_iter})

        return opt_solution

    def optimize_network(self, input, epoch= 1, error_function='Squared',
                         lamda= 0.0001, beta= 0.01, rho= 0.01,
                         corruption_level= 0.01, grad_check= False,
                         max_iter= 400):
        """
        This method runs a complete optimzation of the neural net

        ----------
        Inputs:
        ----------
        input: np.array((n_input,n_example), type: float)
            Matrix of the input to be learnt

        epoch: int
            Number of epoch of training

        ----------
        Outputs:
        ----------
        None
        """
        print("Learning...")

        for epo in range(0,epoch):
            start = time.time()
            opt_theta = self.full_gradient_descent(input,
                                                error_function= error_function,
                                                lamda= lamda, beta= beta,
                                                rho= rho, grad_check= grad_check,
                                                max_iter= max_iter).x

            # Update theta with opt_theta
            self.theta = opt_theta

            end = time.time()
            print("Epoch %.0f has been trained in %.3f seconds.") %(epo+1,
                                                                    end-start)

    def reconstruct(self, input, threshold= None):
        """
        Reconstruction of input method
        Given an input and a network computes the output

        ----------
        Inputs:
        ----------
        input: np.array((n_input,n_example), type: float)
            Matrix of the input to be learnt

        threshold: (optional) float
            Float between 0 and 1 used to threshold the value of the
            reconstruction

        ----------
        Outputs:
        ----------
        reconstructed_layer: np.array((n_input,n_example), type: float)
            Matrix with the value of the reconstruction for all the input

        recconstructed_error: float
        """
        # Reconstruct W1, W2, b1 and b2 from theta
        W1 , W2, b1, b2 = self.split_theta(self.theta)

        # Perform the feed-forward pass for testing:
        hidden_layer_value = self.forward_pass(input, W1, b1)
        reconstructed_layer_value = self.forward_pass(hidden_layer_value, W2, b2)

        if threshold != None:
            reconstructed_layer_value[reconstructed_layer_value > threshold] = 1
            reconstructed_layer_value[reconstructed_layer_value <= threshold] = 0

        # If the input is a column
        if input.shape[0] == 0:
            error = sum(sum(abs(reconstructed_layer_value - input)))\
                    / (input.shape[1]) * 100
        # If the input is a line
        elif input.shape[1] == 0:
            error = sum(sum(abs(reconstructed_layer_value - input)))\
                    / (input.shape[0]) * 100
        # If the input is a matrix
        else:
            error = sum(sum(abs(reconstructed_layer_value - input)))\
                    / (input.shape[0] * input.shape[1]) * 100

        return reconstructed_layer_value, error

    def visualize_learning(self, visible_row, visible_column, hidden_row,
                           hidden_column):
        """
        Plot the matrix of weights W1 to see what has been learnt.

        ----------
        Inputs:
        ----------
        None

        ----------
        Outputs:
        ----------
        None
        """
        # Reconstruct W1 from theta
        W1 = self.theta[self.limit0 : self.limit1].reshape(self.n_hidden,
                                                           self.n_visible)
        W1 = (W1 - W1.min()) / (W1.max() - W1.min())

        # Check if the input and hidden layers are suitable to be plotted
        if math.sqrt(self.n_visible).is_integer():
            # Add the weights as a matrix of images
            fig, axis = plt.subplots(nrows = hidden_row, ncols = hidden_column)
            index = 0
            for axis in axis.flat:
                # Add row of weights as an image to the plot
                img = axis.imshow(W1[index, :].reshape(visible_row,
                    visible_column),cmap = 'gray', interpolation = 'nearest')
                axis.set_frame_on(False)
                axis.set_axis_off()
                index += 1

            plt.show()

        else:
            print("No representation possible")

    def save_parameters(self, save_dir, save_filename = 'model.csv'):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        np.savetxt(os.path.join(save_dir, save_filename), self.theta ,
                delimiter= ',')

    def load_parameters(self, load_dir, load_filename):

        self.theta = np.loadtxt(os.path.join(load_dir, load_filename),
                delimiter=',')

    def gradient_checking(self, input, grad_theta, rho, lamda, beta):
        """
        Numerical method to perform a sanity check on the gradient calculated
        during the back propagation.
        see Andrew Ng CS294A sparse autoencoder p10 for more details
        ----------
        Inputs:
        ----------
        input: np.array((n_input,n_example), type: float)
        Matrix of the input to be learnt

        grad_theta: np.array(type: float)
        Gradient calculated during the backpropagation

        rho, lamba, beta, corruption_level: float
        Parameters used during the backpropagation
        ----------
        Outputs:
        ----------
        None
        """
        grad_error = False
        compt_error = 0

        # Create theta+ and theta-
        for i in range(self.theta.size):
            epsilon = np.zeros(self.theta.shape)
            epsilon[i] = pow(10.,-4)

            theta_plus = self.theta + epsilon
            theta_minus = self.theta - epsilon

            # Rebuild W1_plus, W2_plus, b1_plus, b2_plus from theta_plus
            # (respectively _minus from theta_minus)
            W1_plus , W2_plus, b1_plus, b2_plus = self.split_theta(theta_plus)

            W1_minus , W2_minus, b1_minus, b2_minus = \
                                                 self.split_theta(theta_minus)

            # Perform the feed-forward pass for the two sets of weights
            hidden_layer_value_plus = self.forward_pass(input, W1_plus, b1_plus)
            reconstructed_layer_value_plus = self.forward_pass(
                                                        hidden_layer_value_plus,
                                                        W2_plus, b2_plus)

            hidden_layer_value_minus = self.forward_pass(input, W1_minus,
                                                         b1_minus)
            reconstructed_layer_value_minus = self.forward_pass(
                                                        hidden_layer_value_minus,
                                                        W2_minus, b2_minus)

            # Average activation of hidden unit
            rho_hat_plus = np.sum(hidden_layer_value_plus, axis= 1) \
                                    / input.shape[1]
            rho_hat_minus = np.sum(hidden_layer_value_minus, axis= 1) \
                                    / input.shape[1]

            # Compute the sparse cost function J_plus and J_minus respectivelly
            # for theta_plus and theta_minus:
            error_plus = reconstructed_layer_value_plus - input
            sum_of_square_error_plus = 0.5 * np.sum(np.multiply(error_plus,
                                                                error_plus)) \
                                            / input.shape[1]
                # see Andrew Ng CS294A sparse autoencoder
            weight_decay_plus = 0.5 * lamda * (np.sum(np.multiply(W1_plus,
                                                                  W1_plus))
                                                + np.sum(np.multiply(W2_plus,
                                                                     W2_plus)))

            KL_divergence_plus = beta * np.sum(rho*np.log(rho/rho_hat_plus)
                                                + (1 - rho) * np.log((1 - rho) \
                                                    /(1 - rho_hat_plus)))
                # see Andrew Ng CS294A sparse autoencoder p15

            J_plus = sum_of_square_error_plus + weight_decay_plus \
                        + KL_divergence_plus
                # see Andrew Ng CS294A sparse autoencoder p15

            error_minus = reconstructed_layer_value_minus - input
            sum_of_square_error_minus = 0.5 * np.sum(np.multiply(error_minus, error_minus)) / input.shape[1]    # see Andrew Ng CS294A sparse autoencoder p6
            weight_decay_minus = 0.5 * lamda * ( np.sum(np.multiply(W1_minus,W1_minus))
                                                + np.sum(np.multiply(W2_minus,W2_minus)))
            KL_divergence_minus = beta * np.sum( rho * np.log(rho / rho_hat_minus)
                                                + (1 - rho) * np.log((1 - rho)/(1 - rho_hat_minus)))    # see Andrew Ng CS294A sparse autoencoder p15
            J_minus = sum_of_square_error_minus + weight_decay_minus + KL_divergence_minus                # see Andrew Ng CS294A sparse autoencoder p15

            grad_check = (J_plus -J_minus) / 2 * epsilon

            if np.linalg.norm(grad_theta - grad_check) > pow(10,-2):
                grad_error == True
                compt_error += 1

        if grad_error == True:
            print("    Possible error on the gradient")
            print("    ", compt_error, " coordinates where wrong")
        else:
            print ("    Gradient ok")


def test_autoencoder():
    # Define the parameters of the Autoencoder
    rho             = 0.01      # desired average activation of hidden units
    lamda           = 0.0001    # weight decay parameter
    beta            = 0.        # weight of sparsity penalty term
    n_example       = 500       # number of training examples
    n_hidden        = 10
    n_input         = 5         # dimension of the input
    corruption_level = 0.1
    max_iter        = 50        # number of optimization iterations
    tied_weight     = True
    acti_fun        = 'Sigmoid' # Sigmoid, Tanh
    error_function  = 'Squared' # Squared, CE
    epoch           = 20

    # Generating random inputs:
    rand = np.random.RandomState(int(time.time()))
    inputs = np.asarray(rand.randint(low= 0, high= 2, size=(n_input,n_example)))

    # Initialize the Autoencoder with the above parameters
    encoder = AutoEncoder(n_input, n_hidden,
                          acti_fun= acti_fun,
                          tied_weight= tied_weight)

    # Run the L-BFGS algorithm to get the optimal parameter values
    encoder.optimize_network(inputs, epoch = epoch,
                             error_function= error_function,
                             lamda= lamda, beta= beta, rho= rho,
                             corruption_level= corruption_level,
                             grad_check= False,
                             max_iter= max_iter)

    print(" ")

    # Manual check for the tied weights:
    if tied_weight == True:
        W1 = encoder.theta[encoder.limit0 : encoder.limit1].reshape(
                                            encoder.n_hidden, encoder.n_visible)
        W2 = encoder.theta[encoder.limit1 : encoder.limit2].reshape(
                                            encoder.n_visible, encoder.n_hidden)

        if (W2 == np.transpose(W1)).all() :
            print("WARNING:")
            print("W2 == np.transpose(W1)----> Tied weight correctly implemented")
            print("The network behaved like if there where no tied weight")

    # Inference:
    print(" ")
    print("Inference:")

    entry = rand.randint(low= 0, high= 2, size=(n_input,5))
    threshold = 0.5

    recons_infer, recons_error = encoder.reconstruct(entry,threshold)

    # Saving the model learnt:
    save_dir = "Saving/Test_AE"
    filename = "model.csv"
    encoder.save_parameters(save_dir, save_filename = filename)

    # Creating another AE and loading the model:
    encoder2 = AutoEncoder(n_input, n_hidden, acti_fun= acti_fun,
                           tied_weight= tied_weight)
    encoder2.load_parameters(load_dir= save_dir, load_filename = filename)

    if encoder.theta.all() == encoder2.theta.all():
        print ("Load function is working properly")
    else:
        print ("Bug in the load function")


    print("The error of reconstruction is:  {0}".format(recons_error), "%")

    for col in range(0,entry.shape[1]):
        print ("    entry      -    results    ")
        for line in range(0,entry.shape[0]):
            if (entry[line,col] == recons_infer[line,col]):
                print("%.10f    -    %.10f") %(entry[line,col],
                                                recons_infer[line,col])
            else:
                print("%.10f    -    %.10f  --> error") %(entry[line,col],
                                                          recons_infer[line,col])
        print(" ")

# TEST:
if __name__ == "__main__":
    test_autoencoder()
