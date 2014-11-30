# -*-coding:Utf-8 -*

r"""
Tokenizer.py:
Functions to extract a dataset and prepare it for being used for machine learning

The datasets available are: MNIST, CIFAR-10, HIGGS (kaggle)

-------------
DEPENDENCIES:
-------------
    Libraries:
        - theano: Download at http://deeplearning.net/software/theano/#download
        - numpy

    Scripts:
        None
"""

import cPickle
import gzip
import os
import sys
import time
import numpy as np

import theano
import theano.tensor as T


def shared_dataset(data_xy, borrow=True):
    r"""
    Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy

    shared_x = theano.shared(np.asarray(data_x,
                                        dtype=theano.config.floatX),
                             borrow=borrow)

    shared_y = theano.shared(np.asarray(data_y,
                                        dtype=theano.config.floatX),
                             borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')


#############
### MNIST ###
#############
def load_mnist(dataset_dir="../Datasets"):
    r"""
    Load the MNIST image dataset

    ----------
    Inputs:
    ----------
    dataset_dir: (Optional) string
        Path to the dataset

    ----------
    Outputs:
    ----------
    rval: list(tuple(TensorType))
        List with the train set, the validation set and the test set.
        Each se is a tuple of theano tensors composed of 'data points' and
        'labels'
    """
    # Check if MNIST dataset if it is present
    if os.path.isfile(os.path.join(dataset_dir,'mnist.pkl.gz')):

        f = gzip.open(os.path.join(dataset_dir,'mnist.pkl.gz'), 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()

        #train_set, valid_set, test_set format: tuple(input, target)
        #input is an numpy.ndarray of 2 dimensions (a matrix)
        #witch row's correspond to an example. target is a
        #numpy.ndarray of 1 dimensions (vector)) that have the same length as
        #the number of rows in the input. It should give the target
        #target to the example with the same index in the input.

        # Theano format for any dataset:
        test_set_x, test_set_y = shared_dataset(test_set)
        valid_set_x, valid_set_y = shared_dataset(valid_set)
        train_set_x, train_set_y = shared_dataset(train_set)

        rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
                (test_set_x, test_set_y)]

        return rval

    else:
        print("ERROR: The datapath '", os.path.join(dataset_dir,'mnist.pkl.gz'), "' does not exist.")


##################
### CIFAR - 10 ###
##################
def load_cifar10(dataset_dir="../Datasets/cifar-10-batches-py", split=0.8, toGrey= True, R_prop= 0.299, G_prop=0.587, B_prop= 0.114, normalize=True):
    r"""
    Load the Cifar-10 imqge dataset

    ----------
    Inputs:
    ----------
    dataset_dir: (Optional) string
        Path to the dataset

    split: (Optional) float
        Fraction of the dataset to be used for training and testing

    toGrey: (Optinal) Bool
        Cifar-10 is a RGB images datasets. Do you want to conver them to grey
        scale (to reduce the dimension of the inputs)

    R_prop, G_prop, B_prop: (Optional) float
        Proportion of red, green and blue during the conversion to grey scale

    normalize: (Optional) Bool
        Do you want to normalize the dataset. This option is only available in
        grey scale so far

    ----------
    Outputs:
    ----------
    rval: list(tuple(TensorType))
        List with the train set, the validation set and the test set.
        Each se is a tuple of theano tensors composed of 'data points' and
        'labels'

    """
    # The normalization is only available in grey scale:
    if toGrey == False and normalize == True:
        print("WARNING: Tried to load the cifar-10 dataset in RGB with normalization.\nThis option is not available. The dataset will be loaded without nomralization")

    dataset= 'data_batch_'

    if os.path.isfile(os.path.join(dataset_dir,dataset + '1')):
        # We use 80% of the dataset for training, 20 % for validation
        split = split

        # Extracting training and validation set:
        for i in range(1,6):
            dataset_pkl = open(os.path.join(dataset_dir, dataset + str(i)), 'rb')
            dataset_dic = cPickle.load(dataset_pkl)
            dataset_pkl.close()

            # Split the info into data and label arrays
            if i == 1:
                dataset_expl = dataset_dic['data']
                dataset_label = np.array(dataset_dic['labels'])
            else:
                dataset_expl =np.concatenate(
                        (dataset_expl,dataset_dic['data']))
                dataset_label = np.concatenate(
                        (dataset_label,np.array(dataset_dic['labels'])))

        # To grey scale:
        if toGrey == True:
            dataset_expl_grey = []
            for expl in dataset_expl:

                red = np.array(expl[0:1024])
                green = np.array(expl[1024:2048])
                blue = np.array(expl[2048:3072])

                dataset_expl_grey.append((red * R_prop + green * G_prop \
                                            + blue * B_prop))

            dataset_expl_grey = np.asarray(dataset_expl_grey)

            if normalize == True:
                #Normalize:
                mean_expl_grey = np.mean(dataset_expl_grey)
                std_expl_grey = np.std(dataset_expl_grey)

                dataset_expl_grey = (dataset_expl_grey - mean_expl_grey) \
                                    / std_expl_grey

                # Re-scale:
                min_expl_grey = np.min(dataset_expl_grey)
                max_expl_grey = np.max(dataset_expl_grey)

                dataset_expl_grey = (dataset_expl_grey - min_expl_grey) \
                                        / (max_expl_grey - min_expl_grey)

            # Split into train and validation set
            train_set = (dataset_expl_grey[:int(split * \
                                            dataset_expl_grey.shape[0])],
                         dataset_label[:int(split * dataset_label.shape[0])])

            valid_set = (dataset_expl_grey[int(split * \
                                            dataset_expl_grey.shape[0]):],
                         dataset_label[int(split * dataset_label.shape[0]):])

            del dataset_expl, dataset_label, dataset_dic, dataset_expl_grey

        else:
            train_set = (dataset_expl[:int(split * \
                                            dataset_expl.shape[0])],
                         dataset_label[:int(split * dataset_label.shape[0])])

            valid_set = (dataset_expl[int(split * \
                                            dataset_expl.shape[0]):],
                         dataset_label[int(split * dataset_label.shape[0]):])

            del dataset_expl, dataset_label, dataset_dic,


        # Test_set:
        testset = 'test_batch'
        testset_pkl = open(os.path.join(dataset_dir, testset), 'rb')

        testset_dic = cPickle.load(testset_pkl)
        testset_pkl.close()

        # To grey scale:
        if toGrey == True:
            data_test_grey = []

            for i in range(testset_dic['data'].shape[0]):

                red = np.array(testset_dic['data'][i][0:1024])
                green = np.array(testset_dic['data'][i][1024:2048])
                blue = np.array(testset_dic['data'][i][2048:3072])

                data_test_grey.append((red * R_prop + green * G_prop \
                                        + blue * B_prop))

            if normalize == True:
                #Normalize with the training set values:
                data_test_grey = (data_test_grey - mean_expl_grey) \
                                        / std_expl_grey

                # Re-scale:
                min_test_grey = np.min(data_test_grey)
                max_test_grey = np.max(data_test_grey)

                data_test_grey = (data_test_grey - min_test_grey) \
                                        / (max_test_grey - min_test_grey)


            test_set = (np.asarray(data_test_grey),
                            np.array(testset_dic['labels']))


            del testset_dic, data_test_grey

        else:
            test_set = (np.array(testset_dic['data']),
                            np.array(testset_dic['labels']))

        # Theano format for any dataset:
        test_set_x, test_set_y = shared_dataset(test_set)
        valid_set_x, valid_set_y = shared_dataset(valid_set)
        train_set_x, train_set_y = shared_dataset(train_set)

        rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
                    (test_set_x, test_set_y)]

        return rval

    else:
        print("ERROR: The datapath '", os.path.join(dataset_dir,dataset + '1'), "' does not exist.")



#############
### Higgs ###
#############
def load_higgs(split= False, normalize= True, remove_999= True,
                    noise_variance = 0., n_classes='binary',
                    datapath = "../Datasets/kaggle_higgs/",
                    translate= True):

    sys.path.append('/home/momo/Higgs/')
    import tokenizer
    sys.path.append('/home/momo/Higgs/PostTreatment')
    import tresholding

    # Importing the Higgs dataset using the toke,ozer function:
    train_s, valid_s, test_s = tokenizer.extract_data(
                                        split = split,
                                        normalize = normalize,
                                        remove_999 = remove_999,
                                        noise_variance = noise_variance,
                                        n_classes = n_classes,
                                        train_size = 210000,
                                        train_size2 = 0,
                                        valid_size = 40000,
                                        datapath = datapath,
                                        translate = translate)

    # Convert the object into lis to allow asignement:
    train_s = list(train_s)
    valid_s = list(valid_s)
    test_s  = list(test_s)

    # Non splitted version:
    if split == False:
        # Create a shared version of the train_set:
        train_data_x = train_s[1]
        train_data_y = train_s[2]

        train_shared_x = theano.shared(np.asarray(train_data_x,
                                                  dtype=theano.config.floatX),
                                       borrow= True)
        train_shared_y = theano.shared(np.asarray(train_data_y,
                                                  dtype=theano.config.floatX),
                                       borrow= True)

        # Replace the returned value in train_s by the shared ones
        train_s[1] = train_shared_x
        train_s[2] = train_shared_y

        # Create a shared version of the valid_set:
        valid_data_x = valid_s[1]
        valid_data_y = valid_s[2]

        valid_shared_x = theano.shared(np.asarray(valid_data_x,
                                                  dtype=theano.config.floatX),
                                       borrow= True)
        valid_shared_y = theano.shared(np.asarray(valid_data_y,
                                                  dtype=theano.config.floatX),
                                       borrow= True)

        # Replace the returned value in valid_s by the shared ones
        valid_s[1] = valid_shared_x
        valid_s[2] = valid_shared_y

        # Create a share version of the test set:
        test_data_x = test_s[1]

        test_shared_x = theano.shared(np.asarray(test_data_x,
                                                 dtype=theano.config.floatX),
                                      borrow= True)

        # Replace the returned value in test_s by the shared ones
        test_s[1] = test_shared_x

    else:
        for i in range(len(train_s[1])):
            # Create a shared version of the train_set:
            train_data_x = train_s[1][i]
            train_data_y = train_s[2][i]

            train_shared_x = theano.shared(np.asarray(train_data_x,
                                                      dtype=theano.config.floatX),
                                           borrow= True)
            train_shared_y = theano.shared(np.asarray(train_data_y,
                                                      dtype=theano.config.floatX),
                                           borrow= True)

            # Replace the returned value in train_s by the shared ones
            train_s[1][i] = train_shared_x
            train_s[2][i] = train_shared_y

            # Create a shared version of the valid_set:
            valid_data_x = valid_s[1][i]
            valid_data_y = valid_s[2][i]

            valid_shared_x = theano.shared(np.asarray(valid_data_x,
                                                      dtype=theano.config.floatX),
                                           borrow= True)
            valid_shared_y = theano.shared(np.asarray(valid_data_y,
                                                      dtype=theano.config.floatX),
                                           borrow= True)

            # Replace the returned value in valid_s by the shared ones
            valid_s[1][i] = valid_shared_x
            valid_s[2][i] = valid_shared_y

            # Create a share version of the test set:
            test_data_x = test_s[1][i]

            test_shared_x = theano.shared(np.asarray(test_data_x,
                                                     dtype=theano.config.floatX),
                                          borrow= True)

            # Replace the returned value in test_s by the shared ones
            test_s[1][i] = test_shared_x


    # Convert the object back into tuple:
    train_s = tuple(train_s)
    valid_s = tuple(valid_s)
    test_s  = tuple(test_s)


    return train_s, valid_s, test_s
