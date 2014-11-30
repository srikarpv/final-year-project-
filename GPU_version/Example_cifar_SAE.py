# -*-coding:Utf-8 -*

import os
import sys
import time
import numpy as np

import StackAutoEncoder_theano as SAE
import Tokenizer as tokenizer

def train_cifar():

    print(" ")
    load_only = False
    print "Load_only :                      {0}".format(load_only)

    print "----------------------------- DATASET ----------------------------"
    data = 'cifar-10' # 'mnist.pkl.gz'
    print "Dataset studied :                {0}".format(data)
    print(" ")

    if load_only == False:
        print "--------------------------- PARAMETERS ---------------------------"
        print "-------- Geometry of the network: "
        n_input_width = 32
        n_input_length = 32
        n_channel = 1
        n_input             = n_input_width * n_input_length * n_channel
        print "    Number of visible units :        {0}".format(n_input)
        print "---- LEARNING: "
        epochs              = 50
        print "    Number of epochs :               {0}".format(epochs)
        batch_size          = 20
        print "    Size of batchs :                 {0}".format(batch_size)
        scipy_opt = True
        print "    Scipy_opt :                      {0}".format(scipy_opt)

        if scipy_opt == True:
            optim_method = 'L-BFGS-B'         # L-BFGS-B, CG... (see scipy doc)
            print "    optim_method :                   {0}".format(optim_method)
            max_iter = 400              # Maximum number of evaluation of the
                                        # cost function
            print "    max_iter :                       {0}".format(max_iter)
            epochs = 1                  # No real need for epochs in this case
            print "    epochs :                         {0}".format(epochs)

            lr_dic = {'scipy_opt': scipy_opt, 'optim_method': optim_method,
                      'batch_size': batch_size, 'epochs': epochs,
                      'max_iter': max_iter}

        else:
            epochs = 20
            print "    epochs :                         {0}".format(epochs)
            learning_rate = 0.1         # Initial learning rate
            print "    learning_rate :                  {0}".format(learning_rate)
            tau_learning = None         # Set it to an int if you want the
                                        # learning rate to decrease over time
            print "    tau_learning :                   {0}".format(tau_learning)
            alpha_learning =    1.5     #
            print "    alpha_learning :                 {0}"\
                                                          .format(alpha_learning)

            lr_dic = {'scipy_opt': scipy_opt, 'batch_size': batch_size,
                      'epochs': epochs, 'learning_rate': learning_rate,
                      'tau_learning': tau_learning,
                      'alpha_learning': alpha_learning}

        print(" ")
        print "-------------------------- ARCHITECTURE --------------------------"

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

        machine_1 = {'n_visible': n_input, 'n_hidden': 2000,
            'input': None, 'tied_weight': True,
            'encoder_activation_function': 'sigmoid',
            'decoder_activation_function': 'sigmoid',
            'W1': None, 'W2': None,
            'b1': None, 'b2': None,
            'np_seed': None, 'theano_seed': None,
            'error_function': error_function_1,
            'lr_dic': lr_dic, 'reg_dic': reg_dic_1}


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

        machine_4 = {'n_visible': machine_3['n_hidden'], 'n_hidden': 500,
            'input': None, 'tied_weight': True,
            'encoder_activation_function': 'sigmoid',
            'decoder_activation_function': 'sigmoid',
            'W1': None, 'W2': None,
            'b1': None, 'b2': None,
            'np_seed': None, 'theano_seed': None,
            'error_function': error_function_4,
            'lr_dic': lr_dic, 'reg_dic': reg_dic_4}


        architecture = np.array([machine_1, machine_2, machine_3])#, machine_4])

        for i, machine in enumerate(architecture):
            print "   -Machine %.0f:" %(i+1)
            print machine
            print " "


    print "----------------------------- SAVING -----------------------------"

    save_path = "Saving/Stack_AE_theano"
    print "Path to the saving directory :          {0}".format(save_path)
    save_dir = "10-07_cifar-10_scipyLBFGSB"
    print "Saving directory name :                 {0}".format(save_dir)

    print(" ")
    print "---------------------------- CLASSIFIER --------------------------"

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


    print(" ")
    # Load the dataset:
    print("Loading dataset...")
    datasets = tokenizer.load_cifar10()
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    if load_only == False:
        print(" ")
        # Generating the stack autoencoder:
        print("Generating the stack auto-encoder...")
        stack_AE = SAE.StackAutoEncoder(architecture)

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

    if load_only == True:
        # Load learning:
        print(" ")
        sub_folder = 'unsupervised'
        load_dir = os.path.join(save_path,save_dir,sub_folder)
        stack_AE = SAE.load(load_dir)

        reconstructed_layer_value, error = stack_AE.reconstruct(test_set_x)
        print("The error of the loaded network reconstruction is:  {0}".format(error.eval()), "%")

    # Classifier:
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

# TEST:
if __name__ == "__main__":
    stack_AE = train_cifar()


