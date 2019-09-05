'''
Created on Dec, 2016

@author: hugo

'''
from __future__ import absolute_import
import argparse
import os

import numpy as np
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import ShuffleSplit

from autoencoder.preprocessing.preprocessing import generate_20news_doc_labels
from autoencoder.testing.classifier import multiclass_classifier,  multilabel_classifier
from autoencoder.utils.io_utils import load_json

# def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('train_doc_codes', type=str, help='path to the train doc codes file')
    # parser.add_argument('train_doc_labels', type=str, help='path to the train doc codes file')
    # parser.add_argument('val_doc_codes', type=str, help='path to the train doc codes file')
    # parser.add_argument('val_doc_labels', type=str, help='path to the train doc labels file')
    # parser.add_argument('test_doc_codes', type=str, help='path to the test doc codes file')
    # parser.add_argument('test_doc_labels', type=str, help='path to the test doc labels file')
    # parser.add_argument('-ne', '--n_epoch', type=int, default=100, help='num of epoches (default 100)')
    # parser.add_argument('-bs', '--batch_size', type=int, default=100, help='batch size (default 100)')
    # parser.add_argument('-mlc', '--multilabel_clf', action='store_true', help='multilabel classification flag')
    #
    # args = parser.parse_args()

    # autoencoder
train_doc_codes = load_json('/home/sgnbx/Downloads/projects/KATE-master/output/output.train')
# train_doc_labels = load_json('/home/sgnbx/Downloads/projects/KATE-master/output/output.train')
val_doc_codes = load_json('/home/sgnbx/Downloads/projects/KATE-master/output/output.val')
# val_doc_labels = load_json('/home/sgnbx/Downloads/projects/KATE-master/output/output.val')


# test_doc_codes = load_json(args.test_doc_codes)
# test_doc_labels = load_json(args.test_doc_labels)
X_train = np.r_[train_doc_codes.values()]
print X_train.shape
train_labels = generate_20news_doc_labels(train_doc_codes.keys(),  '/home/sgnbx/Downloads/projects/KATE-master/output/train.labels')
Y_train = [train_labels[i] for i in train_doc_codes]
print Y_train
X_val = np.r_[val_doc_codes.values()]
val_labels = generate_20news_doc_labels(val_doc_codes.keys(),  '/home/sgnbx/Downloads/projects/KATE-master/output/val.labels2')
Y_val = [val_labels[i] for i in val_doc_codes]
# X_test = np.r_[test_doc_codes.values()]
# Y_test = [test_doc_labels[i] for i in test_doc_codes]
# print test_doc_labels.shape

# X_test = np.r_[test_doc_codes.values()]
# print X_test.shape
# Y_test = [test_doc_labels[i] for i in test_doc_codes]
# print len(Y_test)
# # DBN
# X_train = np.array(load_pickle(args.train_doc_codes))
# Y_train = load_pickle(args.train_doc_labels)
# X_val = np.array(load_pickle(args.val_doc_codes))
# Y_val = load_pickle(args.val_doc_labels)
# X_test = np.array(load_pickle(args.test_doc_codes))
# Y_test = load_pickle(args.test_doc_labels)

# encoder = MultiLabelBinarizer()
# encoder.fit(Y_train + Y_val)
# Y_train = encoder.transform(Y_train)
# # print Y_train
# Y_val = encoder.transform(Y_val)
# # Y_test = encoder.transform(Y_test)
# print Y_train.shape
# print len(Y_train)

Y = Y_train + Y_val
# print Y
n_train = len(Y_train)
n_val = len(Y_val)
encoder = LabelEncoder()
Y = np_utils.to_categorical(encoder.fit_transform(Y))
Y_train = Y[:n_train]

Y_val = Y[n_train:n_train + n_val]

seed = 7
# print 'train: %s, val: %s, test: %s' % (X_train.shape[0], X_val.shape[0], X_test.shape[0])
# print 'train: %s, val: %s, test: %s' % (X_train.shape[1], X_val.shape[1], X_test.shape[1])
results = multiclass_classifier(X_train, Y_train, X_val, Y_val, \
                                nb_epoch=50, batch_size=10, seed=seed)
# print 'f1 score on test set: macro_f1: %s, micro_f1: %s' % tuple(results)

