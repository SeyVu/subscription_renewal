#########################################################################################################
#  Description: Coupon visit train csv file has information on user visits. Need to pre process
#  to obtain data that's useful for training. Key information to be extracted are
#  1. Compare coupon information between different train files
#  2. Purchases statistics
#
#########################################################################################################
from __future__ import division
# from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold
# from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RandomForest
# from sklearn.neighbors import KNeighborsClassifier as KNearestNeighbors
# from sklearn import linear_model
from sklearn.metrics import precision_recall_fscore_support

import pandas as pd
import numpy as np
import os

import logging
import logging.config

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


def baseline_models(use_synthetic_data=False):
    print "Importing data"

    if use_synthetic_data:
        if os.path.isfile('data/data_synthetic.csv'):
            churn_df = pd.read_csv('data/data_synthetic.csv', sep=',')
        else:
            raise ValueError("Synthetic data not available")
        # split rows for working on partial data
        start_row = 0
        end_row = 9999
    else:
        churn_df = pd.read_csv('data/train_data.csv', sep=', ')
        # split rows for working on partial data
        start_row = 0
        end_row = 4999

    churn_df = churn_df.iloc[start_row:end_row].copy()

    churn_df = churn_df.reset_index()

    print churn_df

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

    # We don't need these columns. Index is created only when do a partial split
    if 'index' in col_names:
        to_drop = ['index', 'Area Code', 'Phone', 'Churn?']
    else:
        to_drop = ['Area Code', 'Phone', 'Churn?']

    churn_feat_space = churn_df.drop(to_drop, axis=1)

    # 'yes'/'no' has to be converted to boolean values
    # NumPy converts these from boolean to 1. and 0. later
    yes_no_cols = ["Int'l Plan", "VMail Plan"]

    churn_feat_space[yes_no_cols] = (churn_feat_space[yes_no_cols] == 'yes')

    # Below segment replaces column 'State' with a number for each state (alphabetically sorted)
    # separate state into it's own df as it's easier to operate on later
    churn_feat_state = churn_feat_space[['State']]

    state = np.unique(churn_feat_state)

    for index, row in churn_feat_state.iterrows():
        churn_feat_state.iat[index, 0] = int(np.where(state == row['State'])[0])

    churn_feat_space['State'] = churn_feat_state

    logger.debug(churn_feat_space['State'])

    feature_names = churn_feat_space.columns.tolist()

    logger.debug(feature_names)

    x = churn_feat_space.as_matrix().astype(np.float)

    # Feature scaling and normalization
    # scaler = StandardScaler()
    # x = scaler.fit_transform(x)

    logger.debug(x)

    print "Feature space holds %d observations and %d features" % x.shape
    print "Unique target labels:", np.unique(y)

    # Predict accuracy
    # print "Support vector machines:"
    # print "%.3f" % accuracy(y, run_cv(x, y, SVC))
    print "Random forest:"
    print "%.3f" % accuracy(y, run_cv(x, y, RandomForest))
    # print "K-nearest-neighbors:"
    # print "%.3f" % accuracy(y, run_cv(x, y, KNearestNeighbors))
    # print "Logistic regression:"
    # print "%.3f" % accuracy(y, run_cv(x, y, linear_model.LogisticRegression, clf_name="LogReg"))

    # Precision / Recall
    y = np.array(y)

    beta = 2.0  # higher beta prioritizes recall more than precision, default is 1

    # prec_recall = precision_recall_fscore_support(y, run_cv(x, y, SVC), beta=beta, average='binary')
    #
    # logger.debug("SupportVectorMachine Precision %0.3f Recall %0.3f Fbeta-score %0.3f", prec_recall[0],
    #              prec_recall[1], prec_recall[2])

    print "\nBasline model\n"
    prec_recall = precision_recall_fscore_support(y, run_cv(x, y, RandomForest), beta=beta, average='binary')

    logger.debug("\nBaseline model Precision %0.3f Recall %0.3f Fbeta-score %0.3f", prec_recall[0],
                 prec_recall[1], prec_recall[2])

    print "\nOptimized model\n"

    prec_recall = precision_recall_fscore_support(y, run_cv(x, y, RandomForest, n_estimators=200, verbose=0,
                                                            criterion='gini', warm_start='False', n_jobs=-1,
                                                            max_features=5), beta=beta, average='binary')

    logger.debug("\nOptimized model Precision %0.3f Recall %0.3f Fbeta-score %0.3f", prec_recall[0], prec_recall[1],
                 prec_recall[2])

    # Compare probability predictions of algo

    # print "\nPrediction probabilities\n"
    #
    # compare_prob_predictions(x, y, RandomForest, n_estimators=10, verbose=0, criterion='gini', warm_start='False',
    #                          n_jobs=-1, max_features=5)

    # prec_recall = precision_recall_fscore_support(y, run_cv(x, y, KNearestNeighbors), beta=beta, average='binary')
    #
    # logger.debug("KNearestNeighbors Precision %0.3f Recall %0.3f Fbeta-score %0.3f", prec_recall[0], prec_recall[1],
    #              prec_recall[2])
    #
    # prec_recall = precision_recall_fscore_support(y, run_cv(x, y, linear_model.LogisticRegression, C=1e2),
    #                                               beta=beta, average='binary')
    #
    # logger.debug("LogRegression Precision %0.3f Recall %0.3f Fbeta-score %0.3f", prec_recall[0], prec_recall[1],
    #              prec_recall[2])


# Run k-fold cross-validation. Classify users into if they'll churn or no
def run_cv(x, y, clf_class, **kwargs):
    # Construct a kfolds object
    kf = KFold(len(y), n_folds=5, shuffle=True)
    y_pred = y.copy()

    # logger.debug(kf)

    # Initialize a classifier with key word arguments
    clf = clf_class(**kwargs)

    # Iterate through folds
    for train_index, test_index in kf:
        x_train, x_test = x[train_index], x[test_index]
        y_train = y[train_index]
        # Initialize a classifier with key word arguments
        clf = clf_class(**kwargs)

        if train_index[0]:
            print clf

        clf.fit(x_train, y_train)

        y_pred[test_index] = clf.predict(x_test)

    print "\nFeature importance\n"
    # Print importance of the input features and probability for each prediction

    logger.info(clf.feature_importances_)

    return y_pred


# Predict probabilities of churn
def run_prob_cv(x, y, clf_class, **kwargs):
    # Construct a kfolds object
    kf = KFold(len(y), n_folds=5, shuffle=True)

    y_prob = np.zeros((len(y), 2), dtype=float)

    # logger.debug(kf)

    # Initialize a classifier with key word arguments
    clf = clf_class(**kwargs)

    # Iterate through folds
    for train_index, test_index in kf:
        x_train, x_test = x[train_index], x[test_index]
        y_train = y[train_index]

        if train_index[0]:
            print clf

        clf.fit(x_train, y_train)

        # y_prob[idx, class]. Since classes are 2 here, will contain info on prob of both classes
        y_prob[test_index] = clf.predict_proba(x_test)

    # Print importance of the input features and probability for each prediction
    # logger.info(clf.feature_importances_)
    #
    # logger.info(y_prob)

    return y_prob


def compare_prob_predictions(x, y, clf_class, **kwargs):
    import warnings
    warnings.filterwarnings('ignore')

    # Use 10 estimators so predictions are all multiples of 0.1
    pred_prob = run_prob_cv(x, y, clf_class, **kwargs)

    pred_churn = pred_prob[:, 1]

    is_churn = (y == 1)

    # Number of times a predicted probability is assigned to an observation
    counts = pd.value_counts(pred_churn).sort_index()

    # calculate true probabilities
    true_prob = {}
    for prob in counts.index:
        true_prob[prob] = np.mean(is_churn[pred_churn == prob])
        true_prob = pd.Series(true_prob)

    counts = pd.concat([counts, true_prob], axis=1).reset_index()

    counts.columns = ['pred_prob', 'count', 'true_prob']
    print counts
    # print ("Num_wrong_predictions")
    # print (1.0 - counts.icol(0)) * counts.icol(1) * counts.icol(2)


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
    baseline_models(use_synthetic_data=True)
