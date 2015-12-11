#########################################################################################################
# Description: Function trying to predict fantasy points, yards and TDs for various players based on
# their performance over past many years
#
#########################################################################################################

# Imports for various models (Turn on as needed)
from sklearn.ensemble import RandomForestRegressor as RandomForestReg
# from sklearn.linear_model import LinearRegression as LinearReg

# sklearn Toolkit
from sklearn.preprocessing import StandardScaler

import numpy as np

import baseline_models_regression

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

def nfl_pred(feature_scaling=False):
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
        if y[i] < 0.5:  # Chose 0.5 iteratively as that gave best MSE, will vary with dataset
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

    return [x, y]

##################################################################################################################

if __name__ == "__main__":
    # Choose model

    estimator = RandomForestReg
    # estimator_keywords = dict()
    # Below keywords valid only for RF, all other models need own arguments or use empty dict for defaults
    estimator_keywords = dict(n_estimators=1000, verbose=0, warm_start='True', n_jobs=-1,
                              max_features=5)

    start_time = time.time()

    # Choose problem to solve

    # Pep8 shows a warning for all other estimators other than RF This is not a valid warning and has been validated

    [input_features, output] = nfl_pred(feature_scaling=False)

    baseline_models_regression.run_models_wrapper(input_features, output, run_cv_flag=False, num_model_iterations=1,
                                                  plot_learning_curve=False, test_size=0.2,
                                                  clf_class=estimator, **estimator_keywords)

    print("Total time: %0.3f" % float(time.time() - start_time))

##################################################################################################################
