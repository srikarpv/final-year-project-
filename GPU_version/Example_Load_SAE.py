# -*-coding:Utf-8 -*

r"""
Example of how to load a previously saved stack auto encoder
"""


import cPickle
import gzip
import os
import sys
import time
import copy
import numpy as np
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import PIL.Image

import StackAutoEncoder_theano as SAE

sys.path.append('../../Exterieur/DeepLearningTutorials/code/')
from logistic_sgd import load_data
from utils import tile_raster_images



def test_stack_machine():

    # Load learning:
    save_path = "Saving/Stack_AE_theano"
    save_dir = "10-06_3L1000"
    load_dir = os.path.join(save_path,save_dir)
    stack_AE = SAE.load(load_dir)

    # Load the dataset:
    print("Loading dataset...")
    datasets = load_data('mnist.pkl.gz')
    train_set_x, train_set_y = datasets[0]
    test_set_x, test_set_y = datasets[2]

    reconstructed_layer_value, error = stack_AE.reconstruct(test_set_x)

    print("The error of reconstruction is:  {0}".format(error.eval()), "%")

    # Classification:
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
    classifier = 'AdaBoostClassifier'
    print("Classifier used: ", classifier)
    print ("Learning the logistic regression without stack...")
    logReg_withoutStack = stack_AE.supervized_classification(
                train_set_x.eval(), train_set_y.eval(),
                classification_method= classifier)
    print ("Learning the logistic regression with stack...")
    logReg_afterStack = stack_AE.supervized_classification(
            train_reconstructed_layer_value.eval(), train_set_y.eval(),
            classification_method= classifier)

    # Performances:
    print("Without Stack_AE:")
    print ("Accuracy training set:", logReg_withoutStack.score(train_set_x.eval(),
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
    stack_AE = test_stack_machine()



