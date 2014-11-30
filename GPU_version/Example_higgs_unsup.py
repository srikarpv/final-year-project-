# -*-coding:Utf-8 -*

import os
import sys
import time
import numpy as np
import pickle

import StackAutoEncoder_theano as SAE
import Tokenizer as tokenizer

##################
### PARAMETERS ###
##################
print(" ")
print "----------------------------- DATASET ----------------------------"
data = 'higgs' # 'mnist.pkl.gz'
print "Dataset studied :                {0}".format(data)
normalize= True
remove_999= True
translate = True
noise_variance = 0.
n_classes='binary',

# Import the data:
print(" ")
# Load the dataset:
print("Loading dataset...")
train_s, valid_s, test_s = tokenizer.load_higgs(split= True,
                                           normalize= normalize,
                                           remove_999= remove_999,
                                           noise_variance = noise_variance,
                                           n_classes= n_classes,
                                           translate= True)
train_set_x = train_s[1]
train_set_y = train_s[2]

valid_set_x = valid_s[1]
valid_set_y = valid_s[2]
test_set_x = test_s[1]


# Only one DNN is requiered:
print "--------------------------- PARAMETERS ---------------------------"
print "---- LEARNING: "
epochs              = 2
print "    Number of epochs :               {0}".format(epochs)
batch_size          = 10
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
architecture = []
print "-------- Geometry of the network: "
for i in range(len(train_set_x)):
    n_input             = len(train_s[4][i])
    print " Subset %i: Number of visible units :        {0}".format(n_input) %i

    error_function_1    = 'squared' # 'squared' #CE
    corruption_level_1  = 0
    noise_1             = 'masking'
    lamda_1             = 0
    beta_1              = 0
    rho_1               = 0
    delta_1             = 0.1
    learning_rate_1     = 0.1

    reg_dic_1 = {'corruption_level': corruption_level_1, 'noise': noise_1,
                 'lamda': lamda_1, 'beta': beta_1, 'rho': rho_1,
                 'delta': delta_1}

    machine_1 = {'n_visible': n_input, 'n_hidden': 200,
                 'input': None, 'tied_weight': True,
                 'encoder_activation_function': 'sigmoid',
                 'decoder_activation_function': 'sigmoid',
                 'W1': None, 'W2': None,
                 'b1': None, 'b2': None,
                 'np_seed': None, 'theano_seed': None,
                 'error_function': error_function_1,
                 'lr_dic': lr_dic, 'reg_dic': reg_dic_1}


    error_function_2    = 'squared'
    corruption_level_2  = 0
    noise_2             = 'masking'
    lamda_2             = 0.
    beta_2              = 0
    rho_2               = 0
    delta_2             = 0.1
    learning_rate_2     = 0.01 #0.5

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
    corruption_level_3  = 0
    noise_3             = 'masking'
    lamda_3             = 0 #0.0001
    beta_3              = 0 #0.1
    rho_3               = 0
    delta_3             = 0.1
    learning_rate_3     = 0.01

    reg_dic_3 = {'corruption_level': corruption_level_3, 'noise': noise_3,
                 'lamda': lamda_3, 'beta': beta_3, 'rho': rho_3,
                 'delta': delta_3}

    machine_3 = {'n_visible': machine_2['n_hidden'], 'n_hidden': 1000,
                 'input': None, 'tied_weight': True,
                 'encoder_activation_function': 'sigmoid',
                 'decoder_activation_function': 'sigmoid',
                 'W1': None, 'W2': None,
                 'b1': None, 'b2': None,
                 'np_seed': None, 'theano_seed': None,
                 'error_function': error_function_3,
                 'lr_dic': lr_dic, 'reg_dic': reg_dic_3}

    """
    error_function_4    = 'squared'
    corruption_level_4  = 0.3
    noise_4             = 'masking'
    lamda_4             = 0 #0.0001
    beta_4              = 0 #0.1
    rho_4               = 0 #0.1
    delta_4             = 0.1
    learning_rate_4     = 0.01

    reg_dic_4 = {'corruption_level': corruption_level_4, 'noise': noise_4,
                 'lamda': lamda_4, 'beta': beta_4, 'rho': rho_4,
                 'delta': delta_4}

    machine_4 = {'n_visible': machine_3['n_hidden'], 'n_hidden': 1000,
                 'input': None, 'tied_weight': True,
                 'encoder_activation_function': 'sigmoid',
                 'decoder_activation_function': 'sigmoid',
                 'W1': None, 'W2': None,
                 'b1': None, 'b2': None,
                 'np_seed': None, 'theano_seed': None,
                 'error_function': error_function_4,
                 'lr_dic': lr_dic, 'reg_dic': reg_dic_4}
    """

    architecture.append(np.array([machine_1, machine_2, machine_3]))#, machine_4])


print "----------------------------- SAVING -----------------------------"

save_path = "Saving/"
if not os.path.isdir(save_path):
    os.makedirs(save_path)

save_path = os.path.join(save_path, "Higgs")
if not os.path.isdir(save_path):
    os.makedirs(save_path)
save_path =  os.path.join(save_path, "spy-2L-200-500-1000_D01")
if not os.path.isdir(save_path):
    os.makedirs(save_path)
print "Saving directory name :                 {0}".format(save_path)

print(" ")
print "---------------------------- CLASSIFIER --------------------------"


unsup_class = []
for i in range(len(train_set_x)):
    print(" ")
    print("...Training on the subset %i..." %i)
    # Generating the stack autoencoder:
    print("Generating the stack auto-encoder...")
    stack_AE = SAE.StackAutoEncoder(architecture[i])

    print(" ")
    # Training:
    print("Training the stack auto-encoder...")
    stack_AE.unsupervised_pre_training(train_set_x[i])

    # Reconstruction error:
    reconstructed_layer_value, error = stack_AE.reconstruct(test_set_x[i])

    print(" ")
    print("The error of reconstruction is:  {0}".format(error.eval()), "%")

    # Save learning:
    #sub_folder = 'unsupervised'
    #stack_AE.save_experiment(os.path.join(save_path,save_path,sub_folder),
    #                             n_input_length, n_input_width)

    unsup_class.append(stack_AE)

    save_pkl = "subset_" + str(i) + "_SAE.pkl"

    pickle.dump(stack_AE,  open(os.path.join(save_path,save_pkl), "wb"))
