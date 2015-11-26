#########################################################################################################
#  Description: Coupon visit train csv file has information on user visits. Need to pre process
#  to obtain data that's useful for training. Key information to be extracted are
#  1. Compare coupon information between different train files
#  2. Purchases statistics
#
#########################################################################################################

# Imports for various models (Turn on as needed)
from sklearn.ensemble import RandomForestRegressor as RandomForestReg
# from sklearn.linear_model import LinearRegression as LinearReg

# sklearn Toolkit
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler

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

logger = logging.getLogger("debug")


#########################################################################################################


def nfl_pred(num_model_iterations=1, test_size=0.2, plot_learning_curve=False, feature_scaling=False,
             clf_class=RandomForestReg, **kwargs):
    if os.path.isfile('data/nfl_pred/nfl_pred_inputX.npy'):
        x = np.load('data/nfl_pred/nfl_pred_inputX.npy')
    else:
        raise ValueError("Input npy file not available")

    if os.path.isfile('data/nfl_pred/nfl_pred_outputY.npy'):
        y = np.load('data/nfl_pred/nfl_pred_outputY.npy')
    else:
        raise ValueError("Output npy file not available")

    if feature_scaling:
        # Feature scaling and normalization
        scaler = StandardScaler()
        x = scaler.fit_transform(x)

    logger.debug(x[0:10, :])
    logger.debug(y[0:10])

    y *= 10

    # Eliminate players with less than x points
    for i in np.ndindex(y.shape):
        if y[i] < 0.5:  # TODO-Chose 0.5 iteratively as that gave best MSE, will vary with dataset
            x[i, 0] = "nan"
            y[i] = "nan"

    # Eliminate rows with "nan"
    x = x[~np.isnan(y)]
    y = y[~np.isnan(y)]

    logger.debug(x.shape)
    logger.debug(y.shape)

    logger.debug(x[0:10, :])
    logger.debug(y[0:10])

    logger.info("Feature space holds %d observations and %d features" % x.shape)
    logger.info("Unique target labels:")
    logger.info(np.unique(y))

    # Create train / test split only for test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Run model on cross-validation data and predict test data on trained model
    logger.info(sf.Color.BOLD + sf.Color.GREEN + "\nRunning Cross-Validation / Test" + sf.Color.END)
    run_model_regression(run_test_only=0, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                         num_model_iterations=num_model_iterations, plot_learning_curve=plot_learning_curve,
                         clf_class=clf_class, **kwargs)
    # Train model and predict on test data
    logger.info(sf.Color.BOLD + sf.Color.GREEN + "\nRunnning only Test" + sf.Color.END)
    run_model_regression(run_test_only=1, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                         num_model_iterations=num_model_iterations, plot_learning_curve=plot_learning_curve,
                         clf_class=clf_class, **kwargs)

    return None


def run_model_regression(run_test_only, x_train, y_train, x_test, y_test, num_model_iterations=1,
                         plot_learning_curve=False, clf_class=RandomForestReg, **kwargs):
    # # @brief: For cross-validation, Runs the model and gives rmse / mse. Also, will run the trained model
    # #         on test data if run_test_only is set
    # #         For test, trains the model on train data and predicts rmse / mse for test data
    # # @param: x_train, x_test - Input features (numpy array)
    # #         y_train, y_test - expected output (numpy array)
    # #         plot_learning_curve (only for cv) - bool
    # #         num_model_iterations - Times to run the model (to average the results)
    # #         test_size (only for test) - % of data that should be treated as test (in decimal)
    # #         clf_class - Model to run (if specified model doesn't run,
    # #                     then it'll have to be imported from sklearn)
    # #         **kwargs  - Model inputs, refer sklearn documentation for your model to see available parameters
    # # @return: None

    # Plot learning curve only for cv
    if not run_test_only and plot_learning_curve:
        title = "Learning Curves for regression"
        # Train data further split into train and CV
        # Cross validation with 100 iterations to get smoother mean test and train
        # score curves, each time with 20% data randomly selected as a validation set.
        cv = cross_validation.ShuffleSplit(x_train.shape[0], n_iter=100, test_size=0.2, random_state=0)

        modeling_tools.plot_learning_curve(clf_class(**kwargs), title, x_train, y_train, cv=cv, n_jobs=-1)

        if not os.path.isdir("temp_pyplot_regr_dont_commit"):
            # Create dir if it doesn't exist. Do not commit this directory or contents.
            # It's a temp store for pyplot images
            os.mkdir("temp_pyplot_regr_dont_commit")

        # plt.show()
        plt.savefig("temp_pyplot_regr_dont_commit/learning_curve.png")

    # Predict accuracy (mean of num_iterations)
    logger.info("k-fold CV:")

    # Error metrics - mean-squared error and root mse
    rmse_cv = rmse_test = 0.0
    mse_cv = mse_test = 0.0

    for _ in range(num_model_iterations):
        if run_test_only:  # test
            y_pred_test = run_test(x_train, y_train, x_test, clf_class, **kwargs)
            # calculate root mean squared error
            # Pep8 warning not valid
            rmse_test += ((np.mean((y_pred_test - y_test) ** 2)) ** 0.5)
            mse_test += np.mean((y_pred_test - y_test) ** 2)

            # Print first 10 actual and predicted values for test
            logger.debug(y_test[0:10])
            logger.debug(sf.format_float_0_2f(y_pred_test[0:10]))

            logger.debug(np.mean(y_test))
            logger.debug(np.mean(y_pred_test))

        else:  # cv
            y_pred_cv, y_pred_test = run_cv(x_train, y_train, x_test, clf_class, **kwargs)
            # Pep8 warning not valid
            rmse_cv += ((np.mean((y_pred_cv - y_train) ** 2)) ** 0.5)
            mse_cv += np.mean((y_pred_cv - y_train) ** 2)

            # Pep8 warning not valid
            rmse_test += ((np.mean((y_pred_test - y_test) ** 2)) ** 0.5)
            mse_test += np.mean((y_pred_test - y_test) ** 2)

            # Print first 10 actual and predicted values for cv
            logger.debug(y_train[0:10])
            logger.debug(sf.format_float_0_2f(y_pred_cv[0:10]))

            logger.debug(np.mean(y_train))
            logger.debug(np.mean(y_pred_cv))

            # Print first 10 actual and predicted values for test
            logger.debug(y_test[0:10])
            logger.debug(sf.format_float_0_2f(y_pred_test[0:10]))

            logger.debug(np.mean(y_test))
            logger.debug(np.mean(y_pred_test))

    if not run_test_only:
        rmse_cv /= num_model_iterations
        mse_cv /= num_model_iterations

        logger.info(sf.Color.BOLD + sf.Color.DARKCYAN +
                    "\nCV Root Mean squared error {:.2f} Mean squared error {:.2f}".format(rmse_cv,
                                                                                           mse_cv) + sf.Color.END)

    rmse_test /= num_model_iterations
    mse_test /= num_model_iterations

    logger.info(sf.Color.BOLD + sf.Color.DARKCYAN +
                "\nTest Root Mean squared error {:.2f} Mean squared error {:.2f}".format(rmse_test,
                                                                                         mse_test) + sf.Color.END)

    return None


# Run k-fold cross-validation. Classify users into if they'll churn or no
def run_cv(x_train, y_train, x_test, clf_class, **kwargs):

    # Construct a kfolds object from train data
    kf = KFold(len(y_train), n_folds=5, shuffle=True)
    y_pred_cv = y_train.copy()

    # logger.debug(kf)
    # Initialize to avoid pep8 warning, thought clf will always be initialized below
    clf = 0

    # Iterate through folds
    for train_index, cv_index in kf:
        # Initialize a classifier with key word arguments
        clf = clf_class(**kwargs)

        x_cv_train, x_cv_test = x_train[train_index], x_train[cv_index]
        y_cv_train = y_train[train_index]

        if train_index[0]:
            logger.debug(clf)

        clf.fit(x_cv_train, y_cv_train)

        # Predict on cv data
        y_pred_cv[cv_index] = clf.predict(x_cv_test)

    # Now predict test data on trained model
    y_pred_test = clf.predict(x_test)

    if hasattr(clf, "feature_importances_"):
        logger.debug(sf.Color.BOLD + sf.Color.BLUE + "Feature importance" + sf.Color.END)
        # Print importance of the input features and probability for each prediction
        logger.debug(clf.feature_importances_)

    # logger.info(clf.estimators_)

    return y_pred_cv, y_pred_test


# Test dataset. Classify users into if they'll churn or no
def run_test(x_train, y_train, x_test, clf_class, **kwargs):
    # Initialize a classifier with key word arguments
    clf = clf_class(**kwargs)

    logger.debug(clf)

    time.sleep(5)  # sleep time in seconds

    clf.fit(x_train, y_train)

    y_pred_test = clf.predict(x_test)

    if hasattr(clf, "feature_importances_"):
        logger.debug(sf.Color.BOLD + sf.Color.BLUE + "Feature importance" + sf.Color.END)
        # Print importance of the input features and probability for each prediction
        logger.debug(clf.feature_importances_)

    # logger.info(clf.estimators_)

    return y_pred_test


##################################################################################################################

if __name__ == "__main__":
    # Choose model

    estimator = RandomForestReg
    # estimator_keywords = dict()
    # Below keywords valid only for RF, all other models need own arguments or use empty dict for defaults
    estimator_keywords = dict(n_estimators=100, verbose=0, warm_start='False', n_jobs=-1,
                              max_features=5)

    start_time = time.time()

    # Choose problem to solve

    # Pep8 shows a warning for all other estimators other than RF This is not a valid warning and has been validated

    nfl_pred(num_model_iterations=1, plot_learning_curve=True, feature_scaling=False,
             clf_class=estimator, **estimator_keywords)

    print("Total time: %0.3f" % float(time.time() - start_time))
