# -*-coding:Utf-8 -*

import AutoEncoder as AE
import numpy as np
import time
import math
import matplotlib.pyplot as plt
import os
import csv


class stack_AutoEncoder(object):

    def __init__(self, architecture):
        """
        Initialize the stack AutoEncoder class by specifying the number of
        visible units (ie: dimension of the input), the number of output units
        (ie: the dimension of the final layer), the architecture of the hidden
        autoencoderthe and -optional- their weights.

        ----------
        Parameters:
        ----------
        n_input: int
            The number of neuron in the input layer. (ie: the dimension of the
            input)

        n_output: int
            The number of neuron in the output layer.
            (ie: the dimension in which you want to project the input)

        architecture: np.array(number of machines,
                        type: dic(n_visible, n_hidden, weights, rho, lamda, beta,
                            corruption_level, max_iter))
            An array containing the different parameters requiered to construct
            and train the various autoencoders

        ----------
        Attributs:
        ----------

        ----------
        Possible improvements:
        ----------
        """

        self.architecture = architecture

        # Let's create a np.array of AutoEncoders:
        if architecture.shape[0] != 0:
            # Create an non initialized np.array of AutoEncoders of the lenght
            # of the stacked network
            self.stack_machine = np.empty(architecture.shape[0],
                                          dtype= AE.AutoEncoder, order='C')

            # Build the encoder and the decoder:
            # encoder: weights from input to output
            # decoder: weights from output to recosntruction
            self.encoder = []
            self.decoder = []

            # Machine: n_visible - n_hidden
            for i  in range(self.architecture.shape[0]):
                self.stack_machine[i] = AE.AutoEncoder(
                        self.architecture[i]['n_visible'],
                        self.architecture[i]['n_hidden'],
                        acti_fun= self.architecture[i]['acti_fun'],
                        tied_weight = self.architecture[i]['tied_weight'],
                        W1= self.architecture[i]['W1'],
                        W2= self.architecture[i]['W2'],
                        b1= self.architecture[i]['b1'],
                        b2= self.architecture[i]['b2'])

                # Retrieving the weights to fill encoder and decoder
                # If they are not precised as an input they are attributs of
                # the class AE.AutoEncoder
                if (self.architecture[i]['W1'] == None or
                        self.architecture[i]['W2'] == None or
                        self.architecture[i]['b1'] == None or
                        self.architecture[i]['b2'] == None ):

                    W1, W2, b1, b2 = self.stack_machine[i].split_theta(
                                                    self.stack_machine[i].theta)
                else:
                    W1 = self.architecture[i]['W1']
                    W2 = self.architecture[i]['W2']
                    b1 = self.architecture[i]['b1']
                    b2 = self.architecture[i]['b2']

                # Fill encoder and decoder:
                self.encoder.append((W1,b1))
                self.decoder.insert(0,(W2,b2)) # Decoder built backward

    def update_encoder_decoder(self, layer= 'all'):

        if layer == 'all':
            for i  in range(self.stack_machine.shape[0]):
                W1, W2, b1, b2 = self.stack_machine[i].split_theta(
                                                    self.stack_machine[i].theta)

                self.encoder[i] = (W1,b1)
                self.decoder[len(self.decoder)-(i+1)] = (W2, b2)

        else: # Only update the requiered layer
            W1, W2, b1, b2 = self.stack_machine[layer].split_theta(
                                                self.stack_machine[layer].theta)


    def forward_encoding(self, input, encod_start= 0, encod_end= 1):
        """
        Given an input and a layer number perform the forward
        propagation up to this layer
        """
        for i in range(encod_start, encod_end):
            if i == encod_start:
                hidden_layer_value = self.stack_machine[i].forward_pass(input,
                                                            self.encoder[i][0],
                                                            self.encoder[i][1])
            else:
                hidden_layer_value = self.stack_machine[i].forward_pass(
                                                            hidden_layer_value,
                                                            self.encoder[i][0],
                                                            self.encoder[i][1])

        return hidden_layer_value

    def forward_decoding(self, input, decod_start= 0, decod_end= 1):
        """
        Given an input and a layer number perform the forward
        propagation up to this layer
        """

        for i in range(decod_start, decod_end):
            if i == decod_start:
                reconstructed_layer_value = self.stack_machine[i].forward_pass(
                                                            input,
                                                            self.decoder[i][0],
                                                            self.decoder[i][1])

            else:
                reconstructed_layer_value = self.stack_machine[i].forward_pass(
                                                       reconstructed_layer_value,
                                                       self.decoder[i][0],
                                                       self.decoder[i][1])

        return reconstructed_layer_value

    def threshold_input(self,input, method, threshold):
        if method == 'floor':
            input[input > threshold] = 1
            input[input <= threshold] = 0

    def train_stack_autoencoder(self, input, epoch= 10):
        """
        Perform a training layer by layer of the network
        """
        for i, machine in enumerate(self.stack_machine):
            print ("Training layer %.0f...") %(i+1)
            if i == 0:
                # Then the input for learing is the one provided
                # Training
                machine.optimize_network(input,
                    epoch= epoch,
                    error_function = self.architecture[i]['error_function'],
                    lamda= self.architecture[i]['lamda'],
                    beta= self.architecture[i]['beta'],
                    rho= self.architecture[i]['rho'],
                    corruption_level= self.architecture[i]['corruption_level'],
                    max_iter=self.architecture[i]['max_iter'])

            elif i == 1:
                # Then we create 'input_interm' as the propagation of 'input'
                # through the network
                input_interm = self.forward_encoding(input,0,i)
                # Binarization of the 'input':
                #self.threshold_input(input_interm, method,threshold)

                # Training
                machine.optimize_network(input_interm,
                    epoch= epoch,
                    error_function = self.architecture[i]['error_function'],
                    lamda= self.architecture[i]['lamda'],
                    beta= self.architecture[i]['beta'],
                    rho= self.architecture[i]['rho'],
                    corruption_level= self.architecture[i]['corruption_level'],
                    max_iter=self.architecture[i]['max_iter'])


            else:
                # Then we propagate 'input_interm' through the network
                input_interm = self.forward_encoding(input_interm,i-1,i)
                # Binarization of the 'input':
                #self.threshold_input(input_interm, method, threshold)

                # Training
                machine.optimize_network(input_interm,
                    epoch= epoch,
                    error_function = self.architecture[i]['error_function'],
                    lamda= self.architecture[i]['lamda'],
                    beta= self.architecture[i]['beta'],
                    rho= self.architecture[i]['rho'],
                    corruption_level= self.architecture[i]['corruption_level'],
                    max_iter=self.architecture[i]['max_iter'])

            # We update te weights of the encoder and decoder
            self.update_encoder_decoder(layer=i)

        # Full update to be sure:
        self.update_encoder_decoder(layer='all')


    def reconstruct(self, input, threshold=None):
        """
       Reconstruction of input method
        Given an input and a network computes the reconstruction

        ----------
        Inputs:
        ----------
        input: np.array((n_input,n_example), type: float)
            Matrix of the input to be learnt

        method: (optional) string
            Method used to binarized the output

        threshold: (optional) float ( 0 < threshol < 1)
        Limit between 0 and 1

        ----------
        Outputs:
        ----------
        reconstructed_layer: np.array((n_input,n_example), type: float)
            Matrix with the value of the reconstruction for all the input

        recconstructed_error: float
        """

        # Perform the feed-forward pass for testing:
        deconstructed_layer_value = self.forward_encoding (input, 0,
                                                    self.architecture.shape[0])
        reconstructed_layer_value = self.forward_decoding(
                                                    deconstructed_layer_value, 0,
                                                    self.architecture.shape[0])

        if threshold != None:
            reconstructed_layer_value[reconstructed_layer_value > threshold] = 1
            reconstructed_layer_value[reconstructed_layer_value <= threshold] = 0

        if input.shape[0] == 0:
            error = sum(sum(abs(reconstructed_layer_value - input))) \
                        / (input.shape[1]) * 100
        elif input.shape[1] == 0:
            error = sum(sum(abs(reconstructed_layer_value - input))) \
                        / (input.shape[0]) * 100
        else:
            error = sum(sum(abs(reconstructed_layer_value - input))) \
                        / (input.shape[0] * input.shape[1]) * 100

        return reconstructed_layer_value, error


    def visualize_learning(self, layer,  visible_row, visible_column,
                           hidden_row, hidden_column):
        """
        Plot the matrix of weights W1 to see what has been learnt.

        ----------
        Inputs:
        ----------
        layer: int
        Print the required layer

        _row: int
        Number of rows for the plotting

        n_column: int
        Number of column for the plotting
        Note: n_row * n_column = n_hidden

        ----------
        Outputs:
        ----------
        None
        """

        print("Plotting layer %.0f...") %(layer+1)

        if layer == 0:
            # The first layer can be plotted directly using the
            # auto-encoder vizualization function
            self.stack_machine[layer].visualize_learning(visible_row,
                                                         visible_column,
                                                         hidden_row,
                                                         hidden_column)
        else:
            # In this case W1 fist have to be projected in the input space
            weight = self.encoder[layer][0]

            for couple in reversed(self.encoder[0:layer]):
                weight = np.dot(weight,couple[0])

            weight = (weight - weight.min()) / (weight.max() - weight.min())

        # Add the weights as a matrix of images
            fig, axis = plt.subplots(nrows = hidden_row, ncols = hidden_column)
            index = 0

            for axis in axis.flat:
                # Add row of weights as an image to the plot
                img = axis.imshow(weight[index, :].reshape(visible_row,
                    visible_column),cmap = 'gray', interpolation = 'nearest')
                axis.set_frame_on(False)
                axis.set_axis_off()
                index += 1

            plt.show()

    def save_parameters(self, save_dir):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        save_archi_name = "architecture.csv"

        with open(os.path.join(save_dir, save_archi_name), 'w') as outfile:
            fp = csv.DictWriter(outfile, self.architecture[0].keys())
            fp.writeheader()
            fp.writerows(self.architecture)

        for i, machine in enumerate(self.stack_machine):
            save_filename = "machine " + str(i) + ".csv"
            machine.save_parameters(save_dir, save_filename)

def load_parameters(load_dir):

    input_file = csv.DictReader(open(os.path.join(load_dir, "architecture.csv")))

    architecture = []
    for row in input_file:
        architecture.append(dict(row))

    architecture = np.array(architecture)

    # Convert back elmt to string or matrix
    for elt in architecture:
        for key in elt:
            if key in ['W1', 'W2', 'b1', 'b2']:
                elt[key] = np.matrix(elt[key])
            elif key == 'tied_weight':
                elt[key] = bool(elt[key])
            elif key not in ['acti_fun', 'error_function']:
                elt[key] = float(elt[key])


    stack_AE = stack_AutoEncoder(architecture)

    #for i, machine in enumerate(stack_AE):
    #    load_filename = "machine " + str(i) + ".csv"
    #    machine.theta = np.loadtxt(os.path.join(load_dir, load_filename))

    return stack_AE

def test_stack_machine():
    rho              = 0.1       # desired average activation of hidden units
    lamda            = 0.0001    # weight decay parameter
    beta             = 0         # weight of sparsity penalty term
    n_example        = 500       # number of training examples
    n_input          = 10        # dimension of the input
    corruption_level = 0.01
    max_iter         = 400
    n_output         = 5
    acti_fun         = 'Sigmoid'
    error_function   = 'CE'
    tied_weight      = False
    epoch            = 10

    machine_1 = {'n_visible': n_input, 'n_hidden': 10, 'rho': rho,
            'lamda' : lamda, 'beta': beta, 'corruption_level': corruption_level,
            'max_iter': max_iter, 'acti_fun': acti_fun, 'W1': None, 'W2': None,
            'b1': None, 'b2': None, 'error_function': error_function,
            'tied_weight': tied_weight}
    machine_2 = {'n_visible': machine_1['n_hidden'], 'n_hidden': 20,
            'rho': rho, 'lamda' : lamda, 'beta': beta,
            'corruption_level': corruption_level, 'max_iter': max_iter,
            'acti_fun': acti_fun, 'W1': None, 'W2': None, 'b1': None,
            'b2': None, 'error_function': error_function,
            'tied_weight': tied_weight}
    machine_3 = {'n_visible': machine_2['n_hidden'], 'n_hidden': 30,
            'rho': rho, 'lamda' : lamda, 'beta': beta,
            'corruption_level': corruption_level, 'max_iter': max_iter,
            'acti_fun': acti_fun, 'W1': None, 'W2': None, 'b1': None,
            'b2': None, 'error_function': error_function,
            'tied_weight': tied_weight}
    machine_4 = {'n_visible': machine_3['n_hidden'], 'n_hidden': 40,
            'rho': rho, 'lamda' : lamda, 'beta': beta,
            'corruption_level': corruption_level, 'max_iter': max_iter,
            'acti_fun': acti_fun, 'W1': None, 'W2': None, 'b1': None,
            'b2': None, 'error_function': error_function,
            'tied_weight': tied_weight}

    architecture = np.array([machine_1, machine_2, machine_3, machine_4])

    stack_AE = stack_AutoEncoder(architecture)

    # Generating random inputs:
    rand = np.random.RandomState(int(time.time()))
    inputs = np.asarray(rand.randint(low= 0, high= 2, size=(n_input,n_example)))

    # Training:
    stack_AE.train_stack_autoencoder(inputs, epoch= epoch)

    # Saving the model learnt:
    save_dir = "Saving/Test_Stack_AE"
    stack_AE.save_parameters(save_dir)

    # Creating another stack_AE and loading the model:
    stack_AE_2 = load_parameters(save_dir)

    print(" ")

    if stack_AE_2.stack_machine.all() == stack_AE.stack_machine.all():
        print ("Load function is working properly")
    else:
        print ("Bug in the load function")

    # Inference:
    print(" ")
    print("Inference:")
    threshold = 0.3

    entry = rand.randint(low= 0, high= 2, size=(n_input,2))

    recons_infer, recons_error = stack_AE.reconstruct(entry, threshold=None)

    print("The error of reconstruction is:  {0}".format(recons_error), "%")

    for col in range(0,entry.shape[1]):
        print ("    entry      -    results    ")
        for line in range(0,entry.shape[0]):
            if (recons_infer[line,col] > threshold):
                output = 1
            else:
                output = 0

            if (entry[line,col] == output):
                print("%.10f    -    %.10f") %(entry[line,col], output)
            else:
                print("%.10f    -    %.10f  --> error") %(entry[line,col],
                                                          output)
        print(" ")

    return stack_AE



# TEST:
if __name__ == "__main__":
    stack_AE = test_stack_machine()


