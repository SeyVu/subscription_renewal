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
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
# from sklearn.learning_curve import learning_curve

import os
import time
import logging
import logging.config

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import modeling_tools

#########################################################################################################
# Global variables
__author__ = "DataCentric1"
__pass__ = 1
__fail__ = 0

#########################################################################################################
# Setup logging
logging.config.fileConfig('logging.conf')

logger = logging.getLogger("debug")


def telecom_churn(use_synthetic_data=False, plot_learning_curve=False):
    print "Importing data"
    if use_synthetic_data:
        if os.path.isfile('data/data_synthetic.csv'):
            churn_df = pd.read_csv('data/data_synthetic.csv', sep=',')
        else:
            raise ValueError("Synthetic data not available")
        # split rows for working on partial data
        start_row = 5000
        end_row = 9999
    else:
        churn_df = pd.read_csv('data/train_data.csv', sep=', ')
        # split rows for working on partial data
        start_row = 0
        end_row = 4999

    churn_df = churn_df.iloc[start_row:end_row].copy()

    churn_df = churn_df.reset_index()

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

    # logger.debug(churn_feat_space['State'])

    feature_names = churn_feat_space.columns.tolist()

    logger.debug(feature_names)

    x = churn_feat_space.as_matrix().astype(np.float)

    # Feature scaling and normalization
    # scaler = StandardScaler()
    # x = scaler.fit_transform(x)

    logger.debug(x)

    y = np.array(y)

    print "Feature space holds %d observations and %d features" % x.shape
    print "Unique target labels:", np.unique(y)

    # Run random forest

    # Plot learning curve
    if plot_learning_curve:
        title = "Learning Curves - telecom Churn (Random Forest)"
        # Cross validation with 100 iterations to get smoother mean test and train
        # score curves, each time with 20% data randomly selected as a validation set.
        cv = cross_validation.ShuffleSplit(x.shape[0], n_iter=100, test_size=0.2, random_state=0)

        estimator = RandomForest(n_estimators=100, verbose=0, criterion='gini', warm_start="False", n_jobs=-1,
                                 max_features=5)

        modeling_tools.plot_learning_curve(estimator, title, x, y, cv=cv, n_jobs=-1)

        plt.show()

    # Predict accuracy
    # print "\nRandom forest k-fold CV:"

    # Baseline model
    # print "\nBaseline model\n"
    #
    # mean_correct_positive_prediction = 0
    # num_iterations = 1
    # for _ in range(num_iterations):
    #     print "%.3f" % accuracy(y, run_cv(x, y, RandomForest))
    #
    # mean_correct_positive_prediction += correct_positive_prediction
    #
    # mean_correct_positive_prediction /= num_iterations
    #
    # print "mean_correct_positive_prediction ", mean_correct_positive_prediction

    # Optimized model
    # print "\nOptimized model k-fold CV\n"
    #
    # mean_correct_positive_prediction = 0
    # num_iterations = 1
    # for _ in range(num_iterations):
    #
    #     print "%.3f" % accuracy(y, run_cv(x, y, RandomForest, n_estimators=400, verbose=0,
    #                                       criterion='gini', warm_start="False", n_jobs=-1,
    #                                       max_features=5))
    #
    # mean_correct_positive_prediction += correct_positive_prediction
    #
    # mean_correct_positive_prediction /= num_iterations
    #
    # print "mean_correct_positive_prediction ", mean_correct_positive_prediction, "\n"

    # Precision / Recall
    beta = 2.0  # higher beta prioritizes recall more than precision, default is 1

    # print "\nBasline model k-fold CV\n"
    # prec_recall = precision_recall_fscore_support(y, run_cv(x, y, RandomForest), beta=beta, average='binary')
    #
    # logger.debug("\nBaseline model k-fold CV Precision %0.3f Recall %0.3f Fbeta-score %0.3f", prec_recall[0],
    #              prec_recall[1], prec_recall[2])

    # print "\nOptimized model k-fold CV\n"

    # prec_recall = precision_recall_fscore_support(y, run_cv(x, y, RandomForest, n_estimators=400, verbose=0,
    #                                                         criterion='gini', warm_start="False", n_jobs=-1,
    #                                                         max_features=5), beta=beta,
    #                                               average='binary')
    #
    # logger.debug("\nOptimized model k-fold CV Precision %0.3f Recall %0.3f Fbeta-score %0.3f", prec_recall[0],
    #              prec_recall[1], prec_recall[2])

    # Compare probability predictions of algo

    # print "\nPrediction probabilities\n"
    #
    # compare_prob_predictions(x, y, RandomForest, n_estimators=10, verbose=0, criterion='gini', warm_start='False',
    #                          n_jobs=-1, max_features=5)

    # Run other models

    # print "Support vector machines:"
    # print "%.3f" % accuracy(y, run_cv(x, y, SVC))
    # print "K-nearest-neighbors:"
    # print "%.3f" % accuracy(y, run_cv(x, y, KNearestNeighbors))
    # print "Logistic regression:"
    # print "%.3f" % accuracy(y, run_cv(x, y, linear_model.LogisticRegression))

    # prec_recall = precision_recall_fscore_support(y, run_cv(x, y, SVC), beta=beta, average='binary')
    #
    # logger.debug("\nSupportVectorMachine Precision %0.3f Recall %0.3f Fbeta-score %0.3f\n", prec_recall[0],
    #              prec_recall[1], prec_recall[2])
    #
    # prec_recall = precision_recall_fscore_support(y, run_cv(x, y, KNearestNeighbors), beta=beta, average='binary')
    #
    # logger.debug("\nKNearestNeighbors Precision %0.3f Recall %0.3f Fbeta-score %0.3f\n", prec_recall[0],
    #              prec_recall[1], prec_recall[2])
    #
    # prec_recall = precision_recall_fscore_support(y, run_cv(x, y, linear_model.LogisticRegression, C=1e2),
    #                                               beta=beta, average='binary')
    #
    # logger.debug("\nLogRegression Precision %0.3f Recall %0.3f Fbeta-score %0.3f", prec_recall[0], prec_recall[1],
    #              prec_recall[2])

    # Test data

    # Create train / test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Use Random forest
    # print "\nRandom forest Test:"
    #
    # print "\nOptimized model\n"
    # mean_correct_positive_prediction = 0
    # num_iterations = 1
    # for _ in range(num_iterations):
    #     print "%.3f" % accuracy(y_test, run_test(x_train, y_train, x_test, RandomForest, n_estimators=400, verbose=0,
    #                                              criterion='gini', warm_start='False', n_jobs=-1,
    #                                              max_features=5))
    #     mean_correct_positive_prediction += correct_positive_prediction
    #
    # mean_correct_positive_prediction /= num_iterations
    #
    # print "mean_correct_positive_prediction ", mean_correct_positive_prediction, "\n"

    # Precision / Recall

    # print "\nBasline model\n"
    # prec_recall = precision_recall_fscore_support(y_test, run_test(x_train, y_train, x_test, RandomForest), beta=beta,
    #                                               average='binary')
    #
    # logger.debug("\nBaseline model Test - Precision %0.3f Recall %0.3f Fbeta-score %0.3f", prec_recall[0],
    #             prec_recall[1], prec_recall[2])

    # print "\nOptimized model\n"

    prec_recall = precision_recall_fscore_support(y_test,
                                                  run_test(x_train, y_train, x_test, RandomForest, n_estimators=400,
                                                           verbose=0, criterion='gini',
                                                           warm_start='False', n_jobs=-1,
                                                           max_features=5), beta=beta, average='binary')

    logger.debug("\nOptimized model Test - Precision %0.3f Recall %0.3f Fbeta-score %0.3f", prec_recall[0],
                 prec_recall[1], prec_recall[2])


def kids_churn(use_synthetic_data=True, plot_learning_curve=False):
    print "Importing data"
    if use_synthetic_data:
        if os.path.isfile('data/synthetic_kids_ver1.csv'):
            churn_df = pd.read_csv('data/synthetic_kids_ver1.csv', sep=',')
        else:
            raise ValueError("Synthetic data not available")
    else:
        raise ValueError("Actual data not available")

    col_names = churn_df.columns.tolist()

    print "Column names:"
    print col_names

    to_show = col_names[:]

    print "\nSample data:"
    print churn_df[to_show].head(6)

    # Isolate target data
    y = np.array(churn_df['Churn'])

    logger.debug(y)

    to_drop = ['Churn']

    churn_feat_space = churn_df.drop(to_drop, axis=1)

    feature_names = churn_feat_space.columns.tolist()

    logger.debug(feature_names)

    x = churn_feat_space.as_matrix().astype(np.float)

    # Feature scaling and normalization
    # scaler = StandardScaler()
    # x = scaler.fit_transform(x)

    logger.debug(x)

    y = np.array(y)

    print "Feature space holds %d observations and %d features" % x.shape
    print "Unique target labels:", np.unique(y)

    # Random forest

    if plot_learning_curve:
        # Plot learning curve
        title = "Learning Curves - kids Churn (Random Forest)"
        # Cross validation with 100 iterations to get smoother mean test and train
        # score curves, each time with 20% data randomly selected as a validation set.
        cv = cross_validation.ShuffleSplit(x.shape[0], n_iter=40, test_size=0.2, random_state=0)

        estimator = RandomForest(n_estimators=400, verbose=0, criterion='gini', warm_start="False", n_jobs=-1,
                                 max_features=6)
        modeling_tools.plot_learning_curve(estimator, title, x, y, cv=cv, n_jobs=-1)

        plt.show()

    # k-fold cross-validation

    # print "Random forest k-fold CV:"
    # mean_correct_positive_prediction = 0
    # num_iterations = 1
    # for _ in range(num_iterations):
    #     print "%.3f" % accuracy(y, run_cv(x, y, RandomForest, n_estimators=100, verbose=0,
    #                                       criterion='gini', warm_start='True', n_jobs=-1,
    #                                       max_features=6))
    #     mean_correct_positive_prediction += correct_positive_prediction
    #
    # mean_correct_positive_prediction /= num_iterations
    #
    # print "mean_correct_positive_prediction ", mean_correct_positive_prediction

    # Precision / Recall
    beta = 2.0  # higher beta prioritizes recall more than precision, default is 1

    # print "\nBasline model\n"
    # prec_recall = precision_recall_fscore_support(y, run_cv(x, y, RandomForest), beta=beta, average='binary')
    #
    # logger.debug("\nBaseline model k-fold CV Precision %0.3f Recall %0.3f Fbeta-score %0.3f", prec_recall[0],
    #              prec_recall[1], prec_recall[2])
    #
    print "\nOptimized model\n"

    prec_recall = precision_recall_fscore_support(y, run_cv(x, y, RandomForest, n_estimators=200, verbose=0,
                                                            criterion='gini', n_jobs=-1,
                                                            max_features=6), beta=beta, average='binary')

    logger.debug("\nOptimized model k-fold CV Precision %0.3f Recall %0.3f Fbeta-score %0.3f", prec_recall[0],
                 prec_recall[1], prec_recall[2])

    # Other models

    # print "Logistic regression:"
    # print "%.3f" % accuracy(y, run_cv(x, y, linear_model.LogisticRegression))
    # print "Support vector machines:"
    # print "%.3f" % accuracy(y, run_cv(x, y, SVC))
    # print "K-nearest-neighbors:"
    # print "%.3f" % accuracy(y, run_cv(x, y, KNearestNeighbors, n_neighbors=20))

    # prec_recall = precision_recall_fscore_support(y, run_cv(x, y, SVC), beta=beta, average='binary')
    #
    # logger.debug("SupportVectorMachine Precision %0.3f Recall %0.3f Fbeta-score %0.3f", prec_recall[0],
    #               prec_recall[1], prec_recall[2])
    #
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

    # Run on test data

    # Create train / test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Use Random forest
    # print "Random forest Test:"
    # mean_correct_positive_prediction = 0
    # num_iterations = 1
    # for _ in range(num_iterations):
    #     print "%.3f" % accuracy(y_test, run_test(x_train, y_train, x_test, RandomForest, n_estimators=100, verbose=0,
    #                                              criterion='gini', warm_start='True', n_jobs=-1,
    #                                              max_features=6))
    #     mean_correct_positive_prediction += correct_positive_prediction
    #
    # mean_correct_positive_prediction /= num_iterations
    #
    # print "mean_correct_positive_prediction ", mean_correct_positive_prediction

    # Precision / Recall
    beta = 2.0  # higher beta prioritizes recall more than precision, default is 1

    # print "\nBasline model Test\n"
    # prec_recall = precision_recall_fscore_support(y_test, run_test(x_train, y_train, x_test, RandomForest), beta=beta,
    #                                               average='binary')
    #
    # logger.debug("\nBaseline model Test - Precision %0.3f Recall %0.3f Fbeta-score %0.3f", prec_recall[0],
    #              prec_recall[1], prec_recall[2], "\n")

    print "\nOptimized model Test\n"

    prec_recall = precision_recall_fscore_support(y_test,
                                                  run_test(x_train, y_train, x_test, RandomForest, n_estimators=1000,
                                                           verbose=0, criterion='gini', warm_start="True", n_jobs=-1,
                                                           max_features=6), beta=beta, average='binary')

    logger.debug("\nOptimized model Test - Precision %0.3f Recall %0.3f Fbeta-score %0.3f", prec_recall[0],
                 prec_recall[1], prec_recall[2])


def accuracy(y_true, y_pred):
    # NumPy interprets True and False as 1. and 0.
    positive_prediction = np.array(y_true)  # Create np array of size y_true, values will be overwritten below

    global correct_positive_prediction
    global correct_negative_prediction
    global incorrect_prediction

    correct_positive_prediction = 0
    correct_negative_prediction = 0
    incorrect_prediction = 0

    for idx, value in np.ndenumerate(y_true):
        if y_true[idx] == y_pred[idx]:
            positive_prediction[idx] = 1.0
        else:
            positive_prediction[idx] = 0.0

        if y_pred[idx] == 1 and y_true[idx] == y_pred[idx]:
            correct_positive_prediction += 1
        elif y_pred[idx] == 0 and y_true[idx] == y_pred[idx]:
            correct_negative_prediction += 1
        else:
            incorrect_prediction += 1

    print "correct_positive_prediction ", correct_positive_prediction
    print "correct_negative_prediction ", correct_negative_prediction
    print "Incorrect_prediction ", incorrect_prediction

    return np.mean(positive_prediction)


# Run k-fold cross-validation. Classify users into if they'll churn or no
def run_cv(x, y, clf_class, **kwargs):
    # Construct a kfolds object
    kf = KFold(len(y), n_folds=5, shuffle=True)
    y_pred = y.copy()

    start_time = time.time()

    # logger.debug(kf)
    # Initialize a classifier with key word arguments
    clf = clf_class(**kwargs)

    # Iterate through folds
    for train_index, test_index in kf:
        # print"Index"
        # print train_index, test_index
        # print np.shape(train_index), np.shape(test_index)

        x_train, x_test = x[train_index], x[test_index]
        y_train = y[train_index]

        print "kf data ", kf
        print np.shape(x_train), np.shape(x_test), np.shape(y_train)
        print np.shape(train_index), np.shape(test_index)

        # if train_index[0]:
        #     print clf

        clf.fit(x_train, y_train)

        y_pred[test_index] = clf.predict(x_test)

        accuracy(y[test_index], y_pred[test_index])

    if hasattr(clf, "feature_importances_"):
        print "\nFeature importance\n"
        # Print importance of the input features and probability for each prediction
        logger.info(clf.feature_importances_)

    # logger.info(clf.estimators_)

    print "Total time taken is ", time.time() - start_time

    return y_pred


# Run k-fold cross-validation. Classify users into if they'll churn or no
def run_cv_splits(x, y, num_of_splits, clf_class, **kwargs):

    if num_of_splits < 2:
        raise ValueError("Invalid number of splits, needs to be atleast 2")

    # Construct a list with different x, y for each split
    # Initialize empty lists
    x_split = []
    y_split = []

    num_rows = x.shape[0] + 1  # TODO - check if equ to np.shape(x)[0] which gives a warning

    for split in range(num_of_splits):
        start_row = split * num_rows / num_of_splits
        end_row = ((split + 1) * num_rows / num_of_splits)

        x_split.append(x[start_row:end_row, :])
        y_split.append(y[start_row:end_row])

    # Make the last split the test data
    x_test = x_split[num_of_splits - 1]
    y_test = y_split[num_of_splits - 1]

    # Initialize to avoid pep8 warning, thought clf and y_pred will always be initialized below
    y_pred = clf = 0

    # Iterate through first n-1 splits
    for split in range(num_of_splits - 1):
        x_train = x_split[split]
        y_train = y_split[split]

        # Initialize a classifier with key word arguments
        clf = clf_class(**kwargs)

        if not split:  # Print just once
            print clf

        clf.fit(x_train, y_train)

        y_pred = clf.predict(x_test)

        accuracy(y_test, y_pred)

    if hasattr(clf, "feature_importances_"):
        print "\nFeature importance\n"
        # Print importance of the input features and probability for each prediction
        logger.info(clf.feature_importances_)

    return [y_test, y_pred]


# Test on different dataset. Classify users into if they'll churn or no
def run_test(x_train, y_train, x_test, clf_class, **kwargs):
    # Initialize a classifier with key word arguments
    clf = clf_class(**kwargs)

    print clf

    time.sleep(5)  # sleep time in seconds

    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    if hasattr(clf, "feature_importances_"):
        print "\nFeature importance\n"
        # Print importance of the input features and probability for each prediction
        logger.info(clf.feature_importances_)

    # logger.info(clf.estimators_)

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

    if hasattr(clf, "feature_importances_"):
        print "\nFeature importance\n"
        # Print importance of the input features and probability for each prediction
        logger.info(clf.feature_importances_)

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


if __name__ == "__main__":
    telecom_churn(use_synthetic_data=True)
    # kids_churn()
