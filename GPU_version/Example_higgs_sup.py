# -*-coding:Utf-8 -*

import os
import sys
import time
import numpy as np
import pickle

import theano
import theano.tensor as T

import StackAutoEncoder_theano as SAE
import Tokenizer as tokenizer


higgs_path_momo = '/home/momo/Higgs/'
sys.path.append(higgs_path_momo)
import tokenizer

higgs_analyse_path_momo = '/home/momo/Higgs/Analyses'
sys.path.append(higgs_analyse_path_momo)
import analyse

higgs_postT_path_momo = '/home/momo/Higgs/postTreatment'
sys.path.append(higgs_postT_path_momo)

##################
### PARAMETERS ###
##################
print "----------------------------- DATASET ----------------------------"
data = 'higgs' # 'mnist.pkl.gz'
print "Dataset studied :                {0}".format(data)
normalize= True
remove_999= True
translate = True
noise_variance = 0.
n_classes='multiclass'
datapath = 'Datasets/kaggle_higgs/'

# Import the data:
print(" ")
# Load the dataset:
print("Loading dataset...")
train_s, train_s_2, valid_s, test_s = tokenizer.extract_data(
                                                        split = True,
                                                        normalize = normalize,
                                                        remove_999 = remove_999,
                                                        translate = translate,
                                                        noise_variance = 0.,
                                                        n_classes = n_classes,
                                                        datapath = datapath,
                                                        train_size = 180000,
                                                        train_size2 = 35000,
                                                        valid_size = 35000)
train_set_x = []
train_set_x_2 = []
valid_set_x = []
test_set_x = []

for i in range(len(train_s[1])):
    train_set_x.append(theano.shared(np.asarray(train_s[1][i],
                                                dtype=theano.config.floatX),
                                     borrow= True))
    train_set_x_2.append(theano.shared(np.asarray(train_s_2[1][i],
                                                  dtype=theano.config.floatX),
                                       borrow= True))
    valid_set_x.append(theano.shared(np.asarray(valid_s[1][i],
                                                dtype=theano.config.floatX),
                                     borrow= True))
    test_set_x.append(theano.shared(np.asarray(test_s[1][i],
                                               dtype=theano.config.floatX),
                                    borrow= True))

print(" ")
print "----------------------------- LOADING ----------------------------"

load_path = "Saving/"
if not os.path.isdir(load_path):
    os.makedirs(load_path)

load_path = os.path.join(load_path, "Higgs")
if not os.path.isdir(load_path):
    os.makedirs(load_path)
load_path =  os.path.join(load_path, "spy-2L-200-1000_D01")
if not os.path.isdir(load_path):
    os.makedirs(load_path)
print "Loading directory name :                 {0}".format(load_path)


### Loading pickle:
sub_machines = []

for i in range(len(train_set_x)):
    load_pkl = "subset_" + str(i) + "_SAE.pkl"
    stack_machine = pickle.load(open(os.path.join(load_path,load_pkl), "rb"))

    # Reconstruction error:
    reconstructed_layer_value, error = stack_machine.reconstruct(test_set_x[i])

    print("The error of reconstruction is:  {0}".format(error.eval()), "%")

    sub_machines.append(stack_machine)


### Projecting the data into the representation space:
for i in range(len(train_set_x)):
    s_SAE = sub_machines[i]

    train_set_x[i] = s_SAE.forward_encoding(train_set_x[i],
                                            0, s_SAE.architecture.shape[0])\
                                                    .eval()

    train_set_x_2[i] = s_SAE.forward_encoding(train_set_x_2[i],
                                              0, s_SAE.architecture.shape[0])\
                                                    .eval()

    valid_set_x[i] = s_SAE.forward_encoding(valid_set_x[i],
                                            0, s_SAE.architecture.shape[0])\
                                                    .eval()

    test_set_x[i] = s_SAE.forward_encoding(test_set_x[i],
                                           0, s_SAE.architecture.shape[0])\
                                                   .eval()

# Convert the object into lis to allow asignement:
train_s = list(train_s)
train_s_2 = list(train_s_2)
valid_s = list(valid_s)
test_s  = list(test_s)

# Retransformed inputs data
train_s[1] = train_set_x
train_s_2[1] = train_set_x_2
valid_s[1] = valid_set_x
test_s[1] = test_set_x

# Convert the object back into tuple:
train_s = tuple(train_s)
train_s_2 = tuple(train_s_2)
valid_s = tuple(valid_s)
test_s  = tuple(test_s)


print(" ")
### Classifier
# Linear SVM:
dMethods = {}
kwargs_linearSVM= {'penalty': 'l2', 'loss': 'l2', 'dual': True, 'tol': 0.0001,
                   'C': 1.0, 'multi_class': 'ovr', 'fit_intercept': True,
                   'intercept_scaling': 1, 'class_weight': None, 'verbose': 0,
                   'random_state': None}

dMethods['linearSVM'] = analyse.analyse(train_s= train_s,
                                           train2_s= train_s_2,
                                           valid_s= valid_s,
                                           method_name = 'linearSVM',
                                           kwargs = kwargs_linearSVM)

print dMethods['linearSVM']['AMS_treshold_valid']

"""
if load_only == True:
    # Load learning:
    print(" ")
    sub_folder = 'unsupervised'
    load_dir = os.path.join(load_path,load_path,sub_folder)
    stack_AE = SAE.load(load_dir)

    reconstructed_layer_value, error = stack_AE.reconstruct(test_set_x)
    print("The error of the loaded network reconstruction is:  {0}".format(error.eval()), "%")
"""


"""
# Classifier:
print(" ")
# Finetuning:
if finetuning == True:
    for i,stack_AE in enumerate(unsup_class):
        stack_AE.supervised_finetuning(datasets, n_labels = n_label,
                                    batch_size= ft_batch_size,
                                    learning_rate= ft_learning_rate,
                                    n_patience= n_patience,
                                    patience_increase= patience_increase,
                                    improvement_threshold = imprvmt_threshold,
                                    training_epochs = ft_epochs)
        # Save fine tuned model:
        sub_folder = 'supervised'
        stack_AE.save_experiment(os.path.join(load_path,load_path,sub_folder),
                                 n_input_length, n_input_width)
        # Classification:
        else:
    #Perform the feed-forward pass for train & testing sets:
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

"""




