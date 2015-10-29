#########################################################################################################
#  Description: Coupon visit train csv file has information on user visits. Need to pre process
#  to obtain data that's useful for training. Key information to be extracted are
#  1. Compare coupon information between different train files
#  2. Purchases statistics
#
#########################################################################################################
from __future__ import division
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RandomForest
from sklearn.neighbors import KNeighborsClassifier as KNearestNeighbors
from sklearn.metrics import precision_recall_fscore_support

import pandas as pd
import numpy as np
import os

import logging
import logging.config
# import support_functions as sf

#########################################################################################################
# Global variables
__author__ = "DataCentric1"
__pass__ = 1
__fail__ = 0

#########################################################################################################
# Setup logging
logging.config.fileConfig('logging.conf')

logger = logging.getLogger("debug")


class PreProcess:
    def __init__(self):
        #  Initialize variables
        self.cwd = os.getcwd()
        self.data_dir = "/home/harsha/seyvu/customer_churn"
        pass


def churn_model():
    print "Importing data"
    churn_df = pd.read_csv('data/train_data.csv', sep=', ')

    col_names = churn_df.columns.tolist()

    print "Column names:"
    print col_names

    to_show = col_names[:6] + col_names[-6:]

    print "\nSample data:"
    print churn_df[to_show].head(6)

    # Isolate target data
    churn_result = churn_df['Churn?']
    y = np.where(churn_result == 'True.', 1, 0)

    logger.debug(y)

    # We don't need these columns
    to_drop = ['State', 'Area Code', 'Phone', 'Churn?']
    churn_feat_space = churn_df.drop(to_drop, axis=1)

    # 'yes'/'no' has to be converted to boolean values
    # NumPy converts these from boolean to 1. and 0. later
    yes_no_cols = ["Int'l Plan", "VMail Plan"]

    churn_feat_space[yes_no_cols] = (churn_feat_space[yes_no_cols] == 'yes')

    # Pull out features for future use
    # features = churn_feat_space.columns

    x = churn_feat_space.as_matrix().astype(np.float)

    # This is important
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    print "Feature space holds %d observations and %d features" % x.shape
    print "Unique target labels:", np.unique(y)

    # Predict accuracy
    print "Support vector machines:"
    print "%.3f" % accuracy(y, run_cv(x, y, SVC))
    print "Random forest:"
    print "%.3f" % accuracy(y, run_cv(x, y, RandomForest))
    print "K-nearest-neighbors:"
    print "%.3f" % accuracy(y, run_cv(x, y, KNearestNeighbors))

    # Precision / Recall
    y = np.array(y)

    prec_recall = precision_recall_fscore_support(y, run_cv(x, y, RandomForest), beta=1.0, average='binary')

    logger.debug("RandomForest Precision %0.3f Recall %0.3f Fbeta-score %0.3f", prec_recall[0], prec_recall[1],
                 prec_recall[2])

    prec_recall = precision_recall_fscore_support(y, run_cv(x, y, SVC), beta=1.0, average='binary')

    logger.debug("SupportVectorMachine Precision %0.3f Recall %0.3f Fbeta-score %0.3f", prec_recall[0], prec_recall[1],
                 prec_recall[2])

    prec_recall = precision_recall_fscore_support(y, run_cv(x, y, KNearestNeighbors), beta=1.0, average='binary')

    logger.debug("KNearestNeighbors Precision %0.3f Recall %0.3f Fbeta-score %0.3f", prec_recall[0], prec_recall[1],
                 prec_recall[2])


def run_cv(x, y, clf_class, **kwargs):
    # Construct a kfolds object
    kf = KFold(len(y), n_folds=5, shuffle=True)
    y_pred = y.copy()

    print kf

    # Iterate through folds
    for train_index, test_index in kf:
        x_train, x_test = x[train_index], x[test_index]
        y_train = y[train_index]
        # Initialize a classifier with key word arguments
        clf = clf_class(**kwargs)
        clf.fit(x_train, y_train)
        y_pred[test_index] = clf.predict(x_test)

    # cnt_not_equal = total_cnt = 0

    # for idx, value in enumerate(y):
    #     if y[idx] != y_pred[idx]:
    #         cnt_not_equal += 1
    #     total_cnt += 1

    # print cnt_not_equal, total_cnt
    # print y_pred

    return y_pred


def accuracy(y_true, y_pred):
    # NumPy interprets True and False as 1. and 0.
    positive_prediction = np.array(y_true)  # Create np array of size y_true, values will be overwritten below

    for idx, value in np.ndenumerate(y_true):
        if y_true[idx] == y_pred[idx]:
            positive_prediction[idx] = 1.0
        else:
            positive_prediction[idx] = 0.0

    return np.mean(positive_prediction)

if __name__ == "__main__":

    churn_model()
