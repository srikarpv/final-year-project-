# -*-coding:Utf-8 -*
r"""
The theano version of the Stack Auto-Encoder.

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

This script define a class 'StackAutoEncoder'. See below for further description of this class

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
        - AutoEncoder_theano: Defines the auto-encoder class
        - Logistic: logistic regression using Theano and stochastic gradient
        descent found on DeepLearning.net
"""

import os
import sys
import time
import csv

import numpy as np

import warnings
warnings.filterwarnings("ignore")

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

import PIL.Image

import AutoEncoder_theano as AE
import Tokenizer as tokenizer

from Logistic import LogisticRegression
from Support import tile_raster_images

class StackAutoEncoder(object):
    r"""
    Stact Auto-Encoder class (StackAutoEncoder) -- Version using Theano
                                                   (and GPU if available)

    A stack auto-encoder produce a new representation (ie: feature generation) of
    the input data by stacking several auto-encoders.

    The representation learned can then be used for classification by adding a
    logistic classifier on the top of the SAE and fine tuning the whole network's
    parameters with a supervized learning phase.

    For testing purpose the SAE can also be used as a pure feature generator.
    Meaning that instead of fine-tuning the full SAE's parameters, one can choose
    to simply perform a classification using the representation learned during
    the unsupervised phase as input data.


    The neural net is train using a gradient descent like algorithm.

    Please refer to Andrew Ng CS294 Lecture notes "Sparse autoencoder" for more
    details.
    """
    def __init__(self, architecture):
        r"""
        Initialize the stack AutoEncoder class by specifying the number of
        visible units (ie: dimension of the input), the number of output units
        (ie: the dimension of the final layer), the architecture of the hidden
        autoencoderthe and -optional- their weights.

        ----------
        Parameters:
        ----------
        architecture: np.array(number of machines,
                        type: dic(n_visible, n_hidden, weights, rho, lamda, beta,
                            corruption_level, max_iter))
            An array containing the different parameters requiered to construct
            and train the various autoencoders

        ----------
        Attributs:
        ----------
        self.inputs: theano.tensor.TensorType
            Initiazed with the given inputs if any or  asa symbolic variable
            decribing the input of the neural net.

        self.architecture: np.array(number of machines,
                        type: dic(n_visible, n_hidden, weights, rho, lamda, beta,
                            corruption_level, max_iter))
            An array containing the different parameters requiered to construct
            and train the various autoencoders

        self.stack_machine: list(AE.autoencoder)
            List of autoEncoder composing the SAE

        self.encoder: list(tuple(W1,b1))
            List of weights to perform the encoding

        self.decoder: list(tuple(W2,b2))
            List of weights to perform the decoding

        ----------
        Possible improvements:
        ----------
        """
        # We use a matrix because we expect a minibatch of several
        # examples, each example being a row
        self.inputs = T.fmatrix(name='input')

        self.architecture = architecture

        # Let's create a np.array of AutoEncoders:
        if architecture.shape[0] != 0:
            # Create an empty list which will contain the AutoEncoders
            self.stack_machine = []

            # Build the encoder and the decoder:
            # encoder: weights from input to output
            # decoder: weights from output to reconstruction
            self.encoder = []
            self.decoder = []

            # Machine: n_visible - n_hidden
            for i  in range(self.architecture.shape[0]):
                self.stack_machine.append(AE.AutoEncoder(
                        self.architecture[i]['n_visible'],
                        self.architecture[i]['n_hidden'],
                        input = self.architecture[i]['input'],
                        tied_weight= self.architecture[i]['tied_weight'],
                        encoder_activation_function = \
                            self.architecture[i]['encoder_activation_function'],
                        decoder_activation_function = \
                            self.architecture[i]['decoder_activation_function'],
                        W1= self.architecture[i]['W1'],
                        W2= self.architecture[i]['W2'],
                        b1= self.architecture[i]['b1'],
                        b2= self.architecture[i]['b2'],
                        np_seed= self.architecture[i]['np_seed'],
                        theano_seed= self.architecture[i]['theano_seed']))

                # Fill the list of weights describing the encoder and decoder
                # parts:
                # Each element of an encoder (or decoder) is a tuple (W,b)
                self.encoder.append(
                        (self.stack_machine[i].W1,self.stack_machine[i].b1))
                # Decoder built backward
                self.decoder.insert(0,
                        (self.stack_machine[i].W2,self.stack_machine[i].b2))


    ################################################################
    # Secondary functions used for training the stack-autoEncoder: #
    ################################################################
    def forward_encoding(self, input, encod_start, encod_end):
        r"""
        Given an input and a layer number perform the forward
        propagation up to this layer

        ----------
        Inputs:
        ----------
        input: theano.tensor.TensorType
            This is a symbolic variable decribing the input of the neural net.
            (ie: one minibatch)

        encod_start: int
            Encoder from where to start the encodage

        encod_end: int
            Encoder from where to end the encodage

        ----------
        Outputs:
        ----------
        hidden_layer_value: theano.tensor.TensorType
            This is a symbolic variable decribing the value of the neuron in the
            hidden layer of the neural net given an input.

        """
        for i in range(encod_start, encod_end):
            if i == encod_start:
                hidden_layer_value = self.stack_machine[i].encoding_pass(input)
            else:
                hidden_layer_value = self.stack_machine[i].\
                                    encoding_pass(hidden_layer_value)

        return hidden_layer_value


    def forward_decoding(self, input, decod_start, decod_end):
        r"""
        Given an input and a layer number perform the forward
        propagation up to this layer

        ----------
        Inputs:
        ----------
        input: theano.tensor.TensorType
            This is a symbolic variable decribing the input of the neural net.
            (ie: one minibatch)

        decod_start: int
            Decoder from where to start the decodage

        decod_end: int
            Decoder from where to end the decodage

        ----------
        Outputs:
        ----------
        reconstructed_layer_value: theano.tensor.TensorType
            This is a symbolic variable decribing the value of the neuron in the
            reconstructed layer of the neural net given an input.
        """
        for i in (range(decod_start, decod_end)):
            if i == decod_start:
                reconstructed_layer_value = self.stack_machine[-(i+1)].\
                                                decoding_pass(input)

            else:
                reconstructed_layer_value = self.stack_machine[-(i+1)].\
                        decoding_pass(reconstructed_layer_value)

        return reconstructed_layer_value

    ################################################
    # Training function: Unsupervised pre-training #
    ################################################
    def train_autoencoder(self, train_set, i):
        """
        Method performing the training of one auto-encoder among the stacked one.

        ----------
        Inputs:
        ----------
        dataset: theano.tensor.TensorType
            This is a symbolic variable decribing the input of the neural net.

        i: int
            Position of the layer to be trained

        epochs: (optional) int
            Number of loop over the dataset

        batch_size: (optional) int
            How many examples are to be included into the batch to be learned

        ----------
        Outputs:
        ----------
        None but the layer is trained

        ----------
        Argument dictionaries:
        ----------
        if scipy_opt == True:
            lr_dic= {'scipy_opt': scipy_opt, 'batch_size': batch_size,
                       'epochs': epochs, 'max_iter': max_iter}
        else:
            lr_dic= {'scipy_opt': scipy_opt, 'batch_size': batch_size,
                       'epochs': epochs, 'learning_rate': learning_rate,
                       'tau_learning': tau_learning,
                       'alpha_learning': alpha_learning}

        reg_dic = {'corruption_level': corruption_level, 'noise': noise,
                   'lamda': lamda, 'beta': beta, 'rho': rho, 'delta': delta}

        # Training the autoEncoder
        theano_AE.train_AE(train_set= train_set_x, error_function= error_function,
                       lr_dic= lr_dic, reg_dic= reg_dic)
        """
        print("Training encoder {0}.".format(i))
        if i == 0:
            # Then the input for learing is the one provided
            # Training:
            self.stack_machine[i].train_AE(train_set= train_set,
                        error_function= self.architecture[i]['error_function'],
                        lr_dic= self.architecture[i]['lr_dic'],
                        reg_dic= self.architecture[i]['reg_dic'])

        else:
            # Then we create 'input_interm' as the propagation of 'input'
            # through the network
            train_set_interm = theano.shared(
                    value= self.forward_encoding(train_set,0,i).eval(),
                    name= 'dataset_interm',
                    borrow= True)

            # Training:
            self.stack_machine[i].train_AE(train_set= train_set_interm,
                        error_function= self.architecture[i]['error_function'],
                        lr_dic= self.architecture[i]['lr_dic'],
                        reg_dic= self.architecture[i]['reg_dic'])


    def unsupervised_pre_training(self, train_set):
        r"""
        Method performing the training of all the auto-encoder
        of the stack auto-encoder.

        ----------
        Inputs:
        ----------
        dataset: theano.tensor.TensorType
            This is a symbolic variable decribing the input of the neural net.

        epochs: (optional) int
            Number of loop over the dataset

        batch_size: (optional) int
            How many examples are to be included into the batch to be learned

        ----------
        Outputs:
        ----------
        None but the layer is trained
        """
        start = time.clock()

        for i in range(len(self.stack_machine)):
            self.train_autoencoder(train_set, i)

        end = time.clock()

        print("The stack auto-encoder was trained in {0} min."\
                .format((end-start)/60))

    #############################################
    # Training function: Supervised fine-tuning #
    #############################################
    def supervised_finetuning(self, datasets, n_labels = 10, batch_size= 1,
                                    learning_rate= 0.001,
                                    n_patience= 10, patience_increase=2.,
                                    improvement_threshold = 0.995,
                                    training_epochs = 1000):
        r"""
        This method perform the supervised finetuning of the stack auto-encoder
        object to create a classifier.
        It add a classifier on top of the stacked machines and perform the
        training of the whole network (stacked machines + classifier unit) by
        backpropagation of the error.

        ----------
        Inputs:
        ----------
        datasets: list(theano.tensor.TensorType)
            List of the different part of the dataset:
            - Training
            - Validation
            - Test

        n_labels: int
            How many classes are we trying to learn?

        batch_size: int
            What is the size of a batch
            A size of 1 mean that each batch will be of the size of the train set.
            A size greater than 1 will lead to batches of a size close to:
                train_set_size / batch_size

        learning rate: float
            Size of the step of the gradient descent

        n_patience: int
            How long do we allow ourself not to find a better solution

        patience_increase: float
            By how much is increased the patience when a better solution is found

        improvement_threshold: float
            By how much the score must be improved for this state to be considered
            as an effective improvement

        training_epochs: int
            Maximum training epoch

        ----------
        Outputs:
        ----------
        None
        """
        # Add a classifier unit on top of our network:
        deconstructed_layer_value = self.forward_encoding (self.inputs,
                0, self.architecture.shape[0])

        classifier_layer = LogisticRegression(
                         input= deconstructed_layer_value,
                         n_in= int(self.architecture[-1]['n_hidden']),
                         n_out= n_labels)

        # Create a list of parameter to be optimized:
        # 1) Parameters from the stack auto-encoders
        theta = []
        for couple in self.encoder:
            theta.append(couple[0])
            theta.append(couple[1])
        # 2) Parameters from the classifier
        for elmt in classifier_layer.params:
            theta.append(elmt)

        #the labels are presented as 1D vector of [int] labels
        labels = T.ivector('labels')

        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set_x.get_value(borrow=True).shape[0]
        n_train_batches /= batch_size
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches /= batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches /= batch_size

        index = T.lscalar('index')  # index to a [mini]batch

        # Compute the cost of the finetuning classification:
        finetune_cost = classifier_layer.negative_log_likelihood(labels)

        # compute the gradients with respect to the model parameters
        grad_theta = T.grad(finetune_cost, theta)

        # Error
        errors = classifier_layer.errors(labels)

        # Compute list of fine-tuning updates
        updates = []
        for thet, grad_thet in zip(theta, grad_theta):
            updates.append((thet, thet - learning_rate * grad_thet))

        train_fn = theano.function(inputs=[index],
              outputs= finetune_cost,
              updates= updates,
              givens= {
                self.inputs: train_set_x[index * batch_size:
                                    (index + 1) * batch_size],
                labels: train_set_y[index * batch_size:
                                    (index + 1) * batch_size]},
              name= 'train')

        test_score_i = theano.function([index], errors,
                givens= {
                   self.inputs: test_set_x[index * batch_size:
                                      (index + 1) * batch_size],
                   labels: test_set_y[index * batch_size:
                                      (index + 1) * batch_size]},
                name= 'test')

        valid_score_i = theano.function([index], errors,
                givens= {
                    self.inputs: valid_set_x[index * batch_size:
                                     (index + 1) * batch_size],
                    labels: valid_set_y[index * batch_size:
                                     (index + 1) * batch_size]},
                name= 'valid')

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in xrange(n_test_batches)]

        print("Finetuning the model...")

        # Early-stopping parameters
        # Look as this many examples regardless the results
        patience = n_patience * n_train_batches

        # Wait this much longer when a new best is found
        patience_increase = float(patience_increase)

        # A relative improvement of this much is considered significant:
        improvement_threshold = improvement_threshold

        # Go through this many minibatches before checking the network
        # on the validation set; in this case we check every epoch
        validation_frequency = min(n_train_batches, patience / 2)

        best_theta = None
        best_validation_loss = np.inf
        test_scor = 0.
        start_time = time.clock()

        done_looping = False
        epoch = 0

        training_epochs = training_epochs

        while (epoch < training_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in xrange(n_train_batches):
                mimibatch_avg_cost = train_fn(minibatch_index)
                iter = (epoch - 1) * n_train_batches + minibatch_index

                if (iter +1) % validation_frequency == 0:
                    validation_losses = valid_score()
                    this_validation_loss = np.mean(validation_losses)
                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                        (epoch, minibatch_index + 1, n_train_batches,
                        this_validation_loss * 100.))

                    # If we got a the best validation score so far:
                    if this_validation_loss < best_validation_loss:
                        # Increase patience if the loss improvement is good enough
                        if (this_validation_loss <
                            best_validation_loss * improvement_threshold):
                            patience = max(patience, iter * patience_increase)

                        # Save best validation score and iteration number:
                        best_validation_loss = this_validation_loss
                        best_iter = iter

                        # Test it on the test set:
                        test_losses = test_score()
                        test_scor = np.mean(test_losses)
                        print(('epoch %i, minibatch %i/%i, test error '
                               'best model %f %%') %
                               (epoch, minibatch_index + 1, n_train_batches,
                               test_scor * 100.))

                if patience <= iter:
                    done_looping = True
                    break

        end_time = time.clock()
        print(('Optimization complete with best validation score of %f %%,'
               'with test performance %f %%') %
               (best_validation_loss * 100., test_scor * 100.))

        print >> sys.stderr, ('The training code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))


    def supervised_classification(self, input, label,
            classification_method= 'RandomForestClassifier'):

        assert classification_method in set(['KNeighborsClassifier',
            'SVC', 'DecisionTreeClassifier', 'RandomForestClassifier',
            'AdaBoostClassifier', 'GaussianNB', 'LDA', 'QDA'])

        # Generate the clasifier:
        if classification_method == 'KNeighborsClassifier':
            from sklearn.neighbors import KNeighborsClassifier
            classifier = KNeighborsClassifier(n_neighbors= 10)
        elif classification_method == 'SVC':
            from sklearn.svm import SVC
            classifier = SVC(gamma=2, C=1)
        elif classification_method == 'DecisionTreeClassifier':
            from sklearn.tree import DecisionTreeClassifier
            classifier = DecisionTreeClassifier(max_depth=5)
        elif classification_method == 'AdaBoostClassifier':
            from sklearn.ensemble import AdaBoostClassifier
            classifier = AdaBoostClassifier()
        elif classification_method == 'GaussianNB':
            from sklearn.naive_bayes import GaussianNB
            classifier = GaussianNB()
        elif classification_method == 'LDA':
            from sklearn.lda import LDA
            classifier = LDA()
        elif classification_method == 'QDA':
            from sklearn.qda import QDA
            classifier = QDA()
        else:
            from sklearn.ensemble import RandomForestClassifier
            classifier = RandomForestClassifier(max_depth=5,
                    n_estimators=10, max_features=1)

        # Train classifier
        classifier.fit(input, label)

        return classifier

    #############################
    # Post-processig functions: #
    #############################
    def reconstruct(self, input):
        r"""
        Reconstruction of input method
        Given an input and a network computes the reconstruction

        ----------
        Inputs:
        ----------
        input: np.array((n_input,n_example), type: float)
            Matrix of the input to be learnt

        ----------
        Outputs:
        ----------
        reconstructed_layer: np.array((n_input,n_example), type: float)
            Matrix with the value of the reconstruction for all the input

        recconstructed_error: float
        """
        # Perform the feed-forward pass for testing:
        deconstructed_layer_value = self.forward_encoding(input, 0,
                                                       self.architecture.shape[0])
        reconstructed_layer_value = self.forward_decoding(
                                                    deconstructed_layer_value,
                                                    0, self.architecture.shape[0])

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


    def vizualize_learning(self, layer, n_input_length, n_input_width,
                           name= 'AE_filters'):
        r"""
        Plot the matrix of weights W1 to see what has been learnt.

        ----------
        Inputs:
        ----------
        layer: int
            Print the required layer

        name: (optional) str
            Name of the saved png

        ----------
        Outputs:
        ----------
        None
        """
        # Printing the first layer is strait forward:
        if layer==0:

            image = PIL.Image.fromarray(tile_raster_images(
                X=self.stack_machine[layer].W1.get_value(borrow=True).T,
                img_shape=(n_input_length, n_input_width), tile_shape=(10, 10),
                tile_spacing=(1, 1)))

        # To print any other layer we first need to project it back into the input
        # space:
        else:
            # Projection:
            weight = theano.shared(
                        value= self.stack_machine[layer]\
                                .W1.get_value(borrow=True).T,
                        name= 'weight',
                        borrow= True)

            for i in range(0,layer):
                weight = theano.shared(
                        value= T.dot(
                            weight.get_value(borrow=True),
                            self.stack_machine[layer-(i+1)].\
                                    W1.get_value(borrow=True).T).eval(),
                        name= 'weight',
                        borrow= True)

            # Plotting:
            image = PIL.Image.fromarray(tile_raster_images(
                X=weight.get_value(borrow=True),
                img_shape=(28, 28), tile_shape=(10, 10),
                tile_spacing=(1, 1)))

        image.save(name + '_layer_'+ str(layer) + '.png')


    def save_experiment(self, save_dir, n_input_length, n_input_width):
        r"""
        Save the learning into a folder.
        It saves:
            - the vizualisation of each layer
            - the architecture (csv file)
            - the weights of each auto-encoder (folder containing csvs)

        ----------
        Inputs:
        ----------
        save_dir: str
            Directory to save he results in

        ----------
        Outputs:
        ----------
        None
        """
        # Creating the folder
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        # Saving the visualization of the weights
        save_viz_name = "AE_filter"

        for layer in range(len(self.architecture)):
            self.vizualize_learning(layer, n_input_length, n_input_width,
                                    name= os.path.join(save_dir, save_viz_name))

        # Saving the parameter of the experiments:
        save_archi_name = "architecture.csv"

        with open(os.path.join(save_dir, save_archi_name), 'w') as outfile:
            fp = csv.DictWriter(outfile, self.architecture[0].keys())
            fp.writeheader()
            fp.writerows(self.architecture)

        # Saving the weights of the network:
        save_dir += "/weight"
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        for i,machine in enumerate(self.stack_machine):
            save_name = "machine" + str(i) + "_"
            # Save W1
            with open(os.path.join(save_dir, save_name + "W1" + ".csv"),
                    'w') as outfile:
                np.savetxt(outfile,
                    np.asarray(machine.W1.get_value(borrow=True),
                            dtype= np.float32),
                        delimiter=';')
            # Save b1
            with open(os.path.join(save_dir, save_name + "b1" + ".csv"),
                    'w') as outfile:
                np.savetxt(outfile,
                        np.asarray(machine.b1.get_value(borrow=True),
                            dtype= np.float32),
                        delimiter=';')
            # Save W2
            if machine.tied_weight == False:
                with open(os.path.join(save_dir, save_name + "W2" + ".csv"),
                        'w') as outfile:
                    np.savetxt(outfile,
                        np.asarray(machine.W2.get_value(borrow=True),
                            dtype= np.float32),
                        delimiter=';')
            else:
                with open(os.path.join(save_dir, save_name + "W2" + ".csv"),
                        'w') as outfile:
                    np.savetxt(outfile,
                        np.asarray(machine.W1.get_value(borrow=True).T,
                            dtype= np.float32),
                        delimiter=';')

            # Save b2
            with open(os.path.join(save_dir, save_name + "b2" + ".csv"),
                    'w') as outfile:
                np.savetxt(outfile,
                        np.asarray(machine.b2.get_value(borrow=True),
                            dtype= np.float32),
                        delimiter=';')

    #####################################
    # End of the class StackAutoEncoder #
    #####################################

##################
# Load function: #
##################
def load(load_dir, archi_csv_name = "architecture.csv", weight_dir= "weight"):
    r"""
    Load a stack auto-ecoder from a folder.
    It load:
        - the architecture (csv file)
        - the weights of each auto-encoder (folder containing csvs)

    ----------
    Inputs:
    ----------
    load_dir: str
        Directory to load the stack auto-encoder from

    archi_csv_name: (optional) str
        Name of the csv file containing the architecture requiered for this stack
        auto-encoder

    weight_dir: (optional) str
        Name of the folder containing the files with the saved weights for the
        stack auto-encoder

    ----------
    Outputs:
    ----------
    stack_AE: StackAutoEncoder object
        Loaded stack auto-encoder
    """
    # Load the architecture
    print("Loading the architecture and generating the stack auto-encoder...")
    archi_csv_name = "architecture.csv"
    input_file = csv.DictReader(open(os.path.join(load_dir, archi_csv_name)))

    architecture = []
    for row in input_file:
        architecture.append(dict(row))

    # Reshaping the dictionary and loading the weight
    load_dir += "/" + weight_dir

    for i, dictionary in enumerate(architecture):
        for elmt in dictionary.values():
            if len(str(elmt)) == 0:
                elmt = 'None'

        machine_name = "machine" + str(i) + "_"

        dictionary['W1'] = theano.shared(value= np.genfromtxt(
                            os.path.join(load_dir, machine_name + "W1" + ".csv"),
                            dtype=np.float32,
                            delimiter=';'),
                             name= 'W1', borrow= True)

        dictionary['b1'] = theano.shared(value= np.genfromtxt(
                            os.path.join(load_dir, machine_name + "b1" + ".csv"),
                            dtype=np.float32,
                            delimiter=';'),
                             name= 'b1', borrow= True)

        if dictionary['tied_weight'] == False:
            dictionary['W2'] = theano.shared(value= np.genfromtxt(
                            os.path.join(load_dir, machine_name + "W2" + ".csv"),
                            dtype=np.float32,
                            delimiter=';'),
                             name= 'W2', borrow= True)
        else:

            dictionary['W2'] = dictionary['W1'].T

        dictionary['b2'] = theano.shared(value= np.genfromtxt(
                            os.path.join(load_dir, machine_name + "b2" + ".csv"),
                            dtype=np.float32,
                            delimiter=';'),
                             name= 'b2', borrow= True)

    architecture = np.array(architecture)

    stack_AE = StackAutoEncoder(architecture)

    return stack_AE


##########
# Tests: #
##########
def test_stack_machine():
    ############################################
    # SETTING the parameters of the experience #
    ############################################
    print "----------------------------- DATASET ----------------------------"
    data = 'mnist.pkl.gz'
    print "Dataset studied :                {0}".format(data)
    print(" ")

    print "--------------------------- PARAMETERS ---------------------------"
    print "-------- Geometry of the network: "
    n_input_width = 28
    n_input_length = 28
    n_input             = n_input_width * n_input_length
    print "    Number of visible units :        {0}".format(n_input)
    print "---- LEARNING: "
    epochs              = 50
    print "    Number of epochs :               {0}".format(epochs)
    batch_size          = 10
    print "    Size of batchs :                 {0}".format(batch_size)
    # Scipy optimisation?
    scipy_opt = True
    print "    Scipy_opt :                      {0}".format(scipy_opt)

    # Parameters requiered for using scipy optimization:
    if scipy_opt == True:
        optim_method = 'CG'         # L-BFGS-B, CG... (see scipy doc)
        print "    optim_method :                   {0}".format(optim_method)
        max_iter = 400              # Maximum number of evaluation of the
                                    # cost function
        print "    max_iter :                       {0}".format(max_iter)
        epochs = 1                  # No real need for epochs in this case
        print "    epochs :                         {0}".format(epochs)

        # Creating the appropriate 'learning dictionary' requiered for creating a
        # SAE
        lr_dic = {'scipy_opt': scipy_opt, 'optim_method': optim_method,
                  'batch_size': batch_size, 'epochs': epochs,
                  'max_iter': max_iter}

    # Parameters requiered for using the simple theano optimization
    else:
        epochs = 20
        print "    epochs :                         {0}".format(epochs)
        learning_rate = 0.1         # Initial learning rate
        print "    learning_rate :                  {0}".format(learning_rate)
        tau_learning = None         # Set it to an int if you want the
                                    # learning rate to decrease over time
        print "    tau_learning :                   {0}".format(tau_learning)
        alpha_learning =    1.5     #
        print "    alpha_learning :                 {0}".format(alpha_learning)

        # Creating the appropriate 'learning dictionary' requiered for creating
        # a SAE
        lr_dic = {'scipy_opt': scipy_opt, 'batch_size': batch_size,
                  'epochs': epochs, 'learning_rate': learning_rate,
                  'tau_learning': tau_learning,
                  'alpha_learning': alpha_learning}

    print(" ")
    print "-------------------------- ARCHITECTURE --------------------------"
    # Parameters for layer 1:
    error_function_1    = 'squared'
    corruption_level_1  = 0.4
    noise_1             = 'masking'
    lamda_1             = 0.01
    beta_1              = 0.3
    rho_1               = 0.1
    delta_1             = 0.3
    learning_rate_1     = 0.001

    reg_dic_1 = {'corruption_level': corruption_level_1, 'noise': noise_1,
                 'lamda': lamda_1, 'beta': beta_1, 'rho': rho_1,
                 'delta': delta_1}

    machine_1 = {'n_visible': n_input, 'n_hidden': 1000,
                 'input': None, 'tied_weight': True,
                 'encoder_activation_function': 'sigmoid',
                 'decoder_activation_function': 'sigmoid',
                 'W1': None, 'W2': None,
                 'b1': None, 'b2': None,
                 'np_seed': None, 'theano_seed': None,
                 'error_function': error_function_1,
                 'lr_dic': lr_dic, 'reg_dic': reg_dic_1}

    # Parameters for layer 2:
    error_function_2    = 'squared'
    corruption_level_2  = 0.4
    noise_2             = 'masking'
    lamda_2             = 0.01
    beta_2              = 0.3
    rho_2               = 0.1
    delta_2             = 0.3
    learning_rate_2     = 0.001 #0.5

    reg_dic_2 = {'corruption_level': corruption_level_2, 'noise': noise_2,
                 'lamda': lamda_2, 'beta': beta_2, 'rho': rho_2,
                 'delta': delta_2}

    machine_2 = {'n_visible': machine_1['n_hidden'], 'n_hidden': 500,
                 'input': None, 'tied_weight': True,
                 'encoder_activation_function': 'sigmoid',
                 'decoder_activation_function': 'sigmoid',
                 'W1': None, 'W2': None,
                 'b1': None, 'b2': None,
                 'np_seed': None, 'theano_seed': None,
                 'error_function': error_function_2,
                 'lr_dic': lr_dic, 'reg_dic': reg_dic_2}

    # Parameters for layer 3:
    error_function_3    = 'squared'
    corruption_level_3  = 0.4
    noise_3             = 'masking'
    lamda_3             = 0.01 #0.0001
    beta_3              = 0.3 #0.1
    rho_3               = 0.1
    delta_3             = 0.3
    learning_rate_3     = 0.001

    reg_dic_3 = {'corruption_level': corruption_level_3, 'noise': noise_3,
                 'lamda': lamda_3, 'beta': beta_3, 'rho': rho_3,
                 'delta': delta_3}

    machine_3 = {'n_visible': machine_2['n_hidden'], 'n_hidden': 100,
                 'input': None, 'tied_weight': True,
                 'encoder_activation_function': 'sigmoid',
                 'decoder_activation_function': 'sigmoid',
                 'W1': None, 'W2': None,
                 'b1': None, 'b2': None,
                 'np_seed': None, 'theano_seed': None,
                 'error_function': error_function_3,
                 'lr_dic': lr_dic, 'reg_dic': reg_dic_3}

    # Parameters for layer 4:
    error_function_4    = 'squared'
    corruption_level_4  = 0.3
    noise_4             = 'masking'
    lamda_4             = 0 #0.0001
    beta_4              = 0 #0.1
    rho_4               = 0 #0.1
    delta_4             = 0.1
    learning_rate_4     = 0.001

    reg_dic_4 = {'corruption_level': corruption_level_4, 'noise': noise_4,
                 'lamda': lamda_4, 'beta': beta_4, 'rho': rho_4,
                 'delta': delta_4}

    machine_4 = {'n_visible': machine_3['n_hidden'], 'n_hidden': 20,
                 'input': None, 'tied_weight': True,
                 'encoder_activation_function': 'sigmoid',
                 'decoder_activation_function': 'sigmoid',
                 'W1': None, 'W2': None,
                 'b1': None, 'b2': None,
                 'np_seed': None, 'theano_seed': None,
                 'error_function': error_function_4,
                 'lr_dic': lr_dic, 'reg_dic': reg_dic_4}

    # Add as much layer as requiered (or as possible considering the memory
    # available...)

    # Create the 'architecture' array:
    architecture = np.array([machine_1, machine_2, machine_3])#, machine_4])

    for i, machine in enumerate(architecture):
        print "   -Machine %.0f:" %(i+1)
        print machine
        print " "

    print "----------------------------- SAVING -----------------------------"
    save_path = "Saving/Stack_AE_theano"
    print "Path to the saving directory :          {0}".format(save_path)
    save_dir = "0707_scipyCG_all"
    print "Saving directory name :                 {0}".format(save_dir)

    print(" ")
    print "---------------------------- CLASSIFIER --------------------------"
    # Do you want to do a full fine-tuning or a on top classifier?
    finetuning = True

    if finetuning == True:
        print "The network will be fintetuned using a supervised backpropagation."
        # classifier = 'AdaBoostClassifier'    ----> To be implemented
        #print "Classifier used on the top :            {0}".format(classifier)
        n_label = 10
        print "Number of label:                     {0}".format(n_label)
        ft_batch_size = 1
        print "Size of the batch (1= dataset size): {0}".format(ft_batch_size)
        ft_learning_rate = 0.001
        print "Finetuning learning rate:            {0}".format(ft_learning_rate)
        n_patience = 10
        print "n_patience:                          {0}".format(n_patience)
        patience_increase = 2.
        print "patience_increase:                   {0}".format(patience_increase)
        imprvmt_threshold = 0.995
        print "improvement_threshold:               {0}".format(imprvmt_threshold)
        ft_epochs = 1000
        print "Finetuning epochs:                   {0}".format(ft_epochs)

    else:
        classifier = 'AdaBoostClassifier'
        print "Classifier used on the top :            {0}".format(classifier)

    ################################################
    # END setting the parameters of the experience #
    ################################################

    #################################
    # Using the Stack Auto-Encoder: #
    #################################
    print(" ")
    # Load the dataset:
    print("Loading dataset...")
    datasets = tokenizer.load_mnist()
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    print(" ")
    # Generating the stack autoencoder:
    print("Generating the stack auto-encoder...")
    stack_AE = StackAutoEncoder(architecture)

    print(" ")
    # Training:
    print("Training the stack auto-encoder...")
    stack_AE.unsupervised_pre_training(train_set_x)

    # Reconstruction error:
    reconstructed_layer_value, error = stack_AE.reconstruct(test_set_x)

    print(" ")
    print("The error of reconstruction is:  {0}".format(error.eval()), "%")

    # Save learning:
    sub_folder = 'unsupervised'
    stack_AE.save_experiment(os.path.join(save_path,save_dir,sub_folder),
                              n_input_length, n_input_width)

    #if load_only == True:
    #    # Load learning:
    #    print(" ")
    #    sub_folder = 'unsupervised'
    #    load_dir = os.path.join(save_path,save_dir,sub_folder)
    #    stack_AE = load(load_dir)

    #    reconstructed_layer_value, error = stack_AE.reconstruct(test_set_x)
    #    print("The error of the loaded network reconstruction is:  {0}".format(error.eval()), "%")

    ###############
    # Classifier: #
    ###############
    print(" ")
    # Finetuning:
    if finetuning == True:
        stack_AE.supervised_finetuning(datasets, n_labels = n_label,
                                    batch_size= ft_batch_size,
                                    learning_rate= ft_learning_rate,
                                    n_patience= n_patience,
                                    patience_increase= patience_increase,
                                    improvement_threshold = imprvmt_threshold,
                                    training_epochs = ft_epochs)

        # Performance assessment is include in the learning phase

        # Save fine tuned model:
        sub_folder = 'supervised'
        stack_AE.save_experiment(os.path.join(save_path,save_dir,sub_folder),
                                 n_input_length, n_input_width)

    # Classification:
    else:
        # Perform the feed-forward pass for train & testing sets:
        train_deconstructed_layer_value = stack_AE.forward_encoding (train_set_x,
            0, stack_AE.architecture.shape[0])
        train_reconstructed_layer_value = stack_AE.forward_decoding(
            train_deconstructed_layer_value, 0, stack_AE.architecture.shape[0])

        test_deconstructed_layer_value = stack_AE.forward_encoding (test_set_x,0,
            stack_AE.architecture.shape[0])
        test_reconstructed_layer_value = stack_AE.forward_decoding(
            test_deconstructed_layer_value, 0, stack_AE.architecture.shape[0])

        # Classifiers:
        print("Classifier used: ", classifier)
        print ("Learning the logistic regression without stack...")
        logReg_withoutStack = stack_AE.supervised_classification(
                train_set_x.eval(), train_set_y.eval(),
                classification_method= classifier)
        print ("Learning the logistic regression with stack...")
        logReg_afterStack = stack_AE.supervised_classification(
            train_reconstructed_layer_value.eval(), train_set_y.eval(),
            classification_method= classifier)

        # Performances:
        print("Without Stack_AE:")
        print ("Accuracy training set:", logReg_withoutStack.score(
                                                            train_set_x.eval(),
                                                            train_set_y.eval()))
        print ("Accuracy test set:", logReg_withoutStack.score(test_set_x.eval(),
                                                          test_set_y.eval()))

        print("With Stack_AE:")
        print ("Accuracy training set:", logReg_afterStack.score(
                                        train_reconstructed_layer_value.eval(),
                                        train_set_y.eval()))

        print ("Accuracy test set:", logReg_afterStack.score(
                                        test_reconstructed_layer_value.eval(),
                                        test_set_y.eval()))

    return stack_AE

#############
# Run test: #
#############
if __name__ == "__main__":
    stack_AE = test_stack_machine()

