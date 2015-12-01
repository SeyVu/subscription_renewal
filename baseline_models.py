#########################################################################################################
#  Description: Coupon visit train csv file has information on user visits. Need to pre process
#  to obtain data that's useful for training. Key information to be extracted are
#  1. Compare coupon information between different train files
#  2. Purchases statistics
#
#########################################################################################################
from __future__ import division

# Imports for various models (Turn on as needed)
from sklearn.ensemble import RandomForestClassifier as RandomForest
# from sklearn.svm import SVC  # Support vector machines
# from sklearn.neighbors import KNeighborsClassifier as KNearestNeighbors
# from sknn.mlp import Classifier as NeuralNetClassifier, Layer as NeuralNetLayer

# sklearn Toolkit
from sklearn.metrics import precision_recall_fscore_support
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import modeling_tools
import support_functions as sf

import os
import time
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

logger = logging.getLogger("info")
#########################################################################################################


def telecom_churn(use_synthetic_data=False, num_model_iterations=1, plot_learning_curve=False, feature_scaling=True,
                  clf_class=RandomForest, **kwargs):
    logger.debug("Importing data")
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

    logger.info(sf.Color.BOLD + sf.Color.GREEN + "Column names:" + sf.Color.END)
    logger.info(col_names)

    to_show = col_names[:6] + col_names[-6:]

    logger.info(sf.Color.BOLD + sf.Color.GREEN + "Sample data:" + sf.Color.END)
    logger.info(churn_df[to_show].head(6))

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

    if feature_scaling:
        # Feature scaling and normalization
        scaler = StandardScaler()
        x = scaler.fit_transform(x)

    logger.debug(x)

    y = np.array(y)

    logger.info("Feature space holds %d observations and %d features" % x.shape)
    logger.info("Unique target labels: ")
    logger.info(np.unique(y))

    # Run model on cross-validation data
    logger.info(sf.Color.BOLD + sf.Color.GREEN + "\nRunning Cross-Validation" + sf.Color.END)
    run_model(cv_0_test_1=0, x=x, y=y, num_model_iterations=num_model_iterations,
              plot_learning_curve=plot_learning_curve, clf_class=clf_class, **kwargs)
    # Run model on test data
    logger.info(sf.Color.BOLD + sf.Color.GREEN + "\nRunnning Test" + sf.Color.END)
    run_model(cv_0_test_1=1, x=x, y=y, num_model_iterations=num_model_iterations, clf_class=clf_class, **kwargs)

    return None


def kids_churn(use_synthetic_data=False, num_model_iterations=1, plot_learning_curve=False, feature_scaling=True,
               clf_class=RandomForest, **kwargs):
    logger.info("Importing data")
    if use_synthetic_data:
        if os.path.isfile('data/synthetic_kids_ver1.csv'):
            churn_df = pd.read_csv('data/synthetic_kids_ver1.csv', sep=',')
        else:
            raise ValueError("Synthetic data not available")
    else:
        raise ValueError("Actual data not available")

    col_names = churn_df.columns.tolist()

    logger.info(sf.Color.BOLD + sf.Color.GREEN + "Column names:" + sf.Color.END)
    logger.info(col_names)

    to_show = col_names[:]

    logger.info(sf.Color.BOLD + sf.Color.GREEN + "\nSample data:" + sf.Color.END)
    logger.info(churn_df[to_show].head(6))

    # Isolate target data
    y = np.array(churn_df['Churn'])

    logger.debug(y)

    to_drop = ['Churn']

    churn_feat_space = churn_df.drop(to_drop, axis=1)

    feature_names = churn_feat_space.columns.tolist()

    logger.debug(feature_names)

    x = churn_feat_space.as_matrix().astype(np.float)

    if feature_scaling:
        # Feature scaling and normalization
        scaler = StandardScaler()
        x = scaler.fit_transform(x)

    logger.debug(x)

    y = np.array(y)

    logger.info("Feature space holds %d observations and %d features" % x.shape)
    logger.info("Unique target labels:")
    logger.info(np.unique(y))

    # Run model on cross-validation data
    logger.info(sf.Color.BOLD + sf.Color.GREEN + "\nRunning Cross-Validation" + sf.Color.END)
    run_model(cv_0_test_1=0, x=x, y=y, num_model_iterations=num_model_iterations,
              plot_learning_curve=plot_learning_curve, clf_class=clf_class, **kwargs)
    # Run model on test data
    logger.info(sf.Color.BOLD + sf.Color.GREEN + "\nRunnning Test" + sf.Color.END)
    run_model(cv_0_test_1=1, x=x, y=y, num_model_iterations=num_model_iterations, clf_class=clf_class, **kwargs)

    return None


def run_model(cv_0_test_1, x, y, num_model_iterations=1, test_size=0.2, plot_learning_curve=False,
              run_prob_predictions=False, clf_class=RandomForest, **kwargs):
    # # @brief: For cross-validation, Runs the model and gives accuracy and precision / recall
    # #         For test, runs the model and gives accuracy and precision / recall by treating
    # #         a random sample of input data as test data
    # # @param: x - Input features (numpy array)
    # #         y - expected output (numpy array)
    # #         plot_learning_curve (only for cv) - bool
    # #         num_model_iterations - Times to run the model (to average the results)
    # #         test_size (only for test) - % of data that should be treated as test (in decimal)
    # #         clf_class - Model to run (if specified model doesn't run,
    # #                     then it'll have to be imported from sklearn)
    # #         **kwargs  - Model inputs, refer sklearn documentation for your model to see available parameters
    # #         plot_learning_curve - bool
    # # @return: None

    # Create train / test split only for test
    if cv_0_test_1:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)

        x_train = np.array(x_train)
        x_test = np.array(x_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
    else:
        x_train, x_test, y_train, y_test = 0, 0, 0, 0

    # Plot learning curve only for cv
    if not cv_0_test_1 and plot_learning_curve:
        title = "Learning Curves"
        # Cross validation with 100 iterations to get smoother mean test and train
        # score curves, each time with 20% data randomly selected as a validation set.
        cv = cross_validation.ShuffleSplit(x.shape[0], n_iter=25, test_size=0.2, random_state=0)

        modeling_tools.plot_learning_curve(clf_class(**kwargs), title, x, y, cv=cv, n_jobs=-1)

        if not os.path.isdir("temp_pyplot_images_dont_commit"):
            # Create dir if it doesn't exist. Do not commit this directory or contents.
            # It's a temp store for pyplot images
            os.mkdir("temp_pyplot_images_dont_commit")

        # plt.show()
        plt.savefig("temp_pyplot_images_dont_commit/learning_curve.png")

    # Predict accuracy (mean of num_iterations)
    logger.info("k-fold CV:")

    mean_correct_positive_prediction = 0
    mean_accuracy = 0

    for _ in range(num_model_iterations):
        if cv_0_test_1:  # test
            mean_accuracy += accuracy(y_test, run_test(x_train, y_train, x_test, clf_class, **kwargs))
        else:  # cv
            mean_accuracy += accuracy(y, run_cv(x, y, clf_class, **kwargs))

        mean_correct_positive_prediction += correct_positive_prediction

    mean_accuracy /= num_model_iterations
    mean_correct_positive_prediction /= num_model_iterations

    logger.info(sf.Color.BOLD + sf.Color.DARKCYAN + "\nAccuracy {:.2f}".format(mean_accuracy * 100) + sf.Color.END)

    logger.info(sf.Color.BOLD + sf.Color.DARKCYAN + "\nCorrect positive prediction {:.2f}".format(
        mean_correct_positive_prediction) + sf.Color.END)

    # Precision / Recall
    beta = 2.0  # higher beta prioritizes recall more than precision, default is 1

    mean_precision = 0
    mean_recall = 0
    mean_fbeta_score = 0

    for _ in range(num_model_iterations):
        if cv_0_test_1:  # test
            prec_recall = precision_recall_fscore_support(y_test, run_test(x_train, y_train, x_test, clf_class,
                                                                           **kwargs), beta=beta, average='binary')
        else:  # cv
            prec_recall = precision_recall_fscore_support(y, run_cv(x, y, clf_class, **kwargs), beta=beta,
                                                          average='binary')

        mean_precision += prec_recall[0]
        mean_recall += prec_recall[1]
        mean_fbeta_score += prec_recall[2]

    mean_precision /= num_model_iterations
    mean_recall /= num_model_iterations
    mean_fbeta_score /= num_model_iterations

    logger.info(sf.Color.BOLD + sf.Color.DARKCYAN + "\nPrecision {:.2f} Recall {:.2f} Fbeta-score {:.2f}".format(
        mean_precision * 100, mean_recall * 100, mean_fbeta_score * 100) + sf.Color.END)

    # Only for CV, compare probability predictions of the model
    if not cv_0_test_1 and run_prob_predictions:
        logger.info("\nPrediction probabilities\n")

        compare_prob_predictions(x, y, clf_class, **kwargs)

    return None


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

    logger.debug("\nAccuracy method output\n")
    logger.debug("correct_positive_prediction %d", correct_positive_prediction)
    logger.debug("correct_negative_prediction %d", correct_negative_prediction)
    logger.debug("Incorrect_prediction %d", incorrect_prediction)

    return np.mean(positive_prediction)


# Run k-fold cross-validation. Classify users into if they'll churn or no
def run_cv(x, y, clf_class, **kwargs):
    # Construct a kfolds object
    kf = KFold(len(y), n_folds=5, shuffle=True)
    y_pred = y.copy()

    # logger.debug(kf)
    # Initialize to avoid pep8 warning, thought clf will always be initialized below
    clf = 0

    # Iterate through folds
    for train_index, test_index in kf:
        # Initialize a classifier with key word arguments
        clf = clf_class(**kwargs)

        x_train, x_test = x[train_index], x[test_index]
        y_train = y[train_index]

        if train_index[0]:
            logger.debug(clf)

        clf.fit(x_train, y_train)

        y_pred[test_index] = clf.predict(x_test)

        accuracy(y[test_index], y_pred[test_index])

    if hasattr(clf, "feature_importances_"):
        logger.debug(sf.Color.BOLD + sf.Color.BLUE + "Feature importance" + sf.Color.END)
        # Print importance of the input features and probability for each prediction
        logger.debug(clf.feature_importances_)

    # logger.info(clf.estimators_)

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
            logger.debug(clf)

        clf.fit(x_train, y_train)

        y_pred = clf.predict(x_test)

        accuracy(y_test, y_pred)

    if hasattr(clf, "feature_importances_"):
        logger.debug(sf.Color.BOLD + sf.Color.BLUE + "\nFeature importance" + sf.Color.END)
        # Print importance of the input features and probability for each prediction
        logger.debug(clf.feature_importances_)

    return [y_test, y_pred]


# Test on different dataset. Classify users into if they'll churn or no
def run_test(x_train, y_train, x_test, clf_class, **kwargs):
    # Initialize a classifier with key word arguments
    clf = clf_class(**kwargs)

    logger.debug(clf)

    time.sleep(5)  # sleep time in seconds

    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    if hasattr(clf, "feature_importances_"):
        logger.debug(sf.Color.BOLD + sf.Color.BLUE + "Feature importance" + sf.Color.END)
        # Print importance of the input features and probability for each prediction
        logger.debug(clf.feature_importances_)

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
        logger.debug(sf.Color.BOLD + sf.Color.BLUE + "\nFeature importance\n" + sf.Color.END)
        # Print importance of the input features and probability for each prediction
        logger.debug(clf.feature_importances_)

    # logger.info(y_prob)

    return y_prob


def compare_prob_predictions(x, y, clf_class, **kwargs):
    import warnings
    warnings.filterwarnings('ignore')  # TODO - check if we can remove this

    # Use 10 estimators so predictions are all multiples of 0.1
    pred_prob = run_prob_cv(x, y, clf_class, **kwargs)

    pred_churn = pred_prob[:, 1]

    is_churn = (y == 1)

    # Number of times a predicted probability is assigned to an observation
    counts = pd.value_counts(pred_churn).sort_index()

    # calculate true probabilities
    true_prob = {}
    for prob in counts.index:
        # Pep8 shows a warning that's not valid
        true_prob[prob] = np.mean(is_churn[pred_churn == prob])
        true_prob = pd.Series(true_prob)

    counts = pd.concat([counts, true_prob], axis=1).reset_index()

    counts.columns = ['pred_prob', 'count', 'true_prob']
    print counts
    # print ("Num_wrong_predictions")
    # print (1.0 - counts.icol(0)) * counts.icol(1) * counts.icol(2)


##################################################################################################################

if __name__ == "__main__":
    # Choose model

    estimator = RandomForest
    # estimator_keywords = dict()
    # Below keywords valid only for RF, all other models need own arguments or use empty dict for defaults
    estimator_keywords = dict(n_estimators=1000, verbose=0, criterion='entropy', warm_start='False', n_jobs=-1,
                              max_features=5)

    start_time = time.time()

    # Neural network
    # estimator = NeuralNetClassifier
    # estimator_keywords = dict(layers=[NeuralNetLayer("Rectifier", units=50), NeuralNetLayer("Softmax")],
    #                           learning_rate=0.001, n_iter=10)

    # Pep8 shows a warning for all other estimators other than RF (probably because RF is the default class in
    # telecom / kids churn. This is not a valid warning and has been validated

    # Choose problem to solve

    telecom_churn(use_synthetic_data=False, num_model_iterations=1, plot_learning_curve=False, feature_scaling=False,
                  clf_class=estimator, **estimator_keywords)

    # kids_churn(use_synthetic_data=True, num_model_iterations=1, plot_learning_curve=True, feature_scaling=False,
    #            clf_class=estimator, **estimator_keywords)

    print("Total time: %0.3f" % float(time.time() - start_time))
