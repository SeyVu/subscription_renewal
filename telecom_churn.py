#########################################################################################################
#  Description: Main file for telecom_churn dataset. Key function is to transform dataset into needed
#  input_features and output
#########################################################################################################
import os
import time
import logging
import logging.config

# Imports for various models (Turn on as needed)
from sklearn.ensemble import RandomForestClassifier as RandomForest
# from sklearn.ensemble import BaggingClassifier as Bagging
# from sklearn.svm import SVC as SVC  # Support vector machines
# from sklearn.neighbors import KNeighborsClassifier as KNN
# from sklearn.linear_model import LogisticRegression as LogReg
# from sklearn.linear_model import RidgeClassifier as Ridge
# from sknn.mlp import Classifier as NeuralNetClassifier, Layer as NeuralNetLayer
from sklearn.ensemble import GradientBoostingClassifier as GradBoost

from sklearn.preprocessing import StandardScaler

# Import Python libs
import pandas as pd
import numpy as np

# Import from within project
import support_functions as sf
import ensemble_models
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

def telecom_churn(use_synthetic_data=False, feature_scaling=True):
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

    return [x, y]


####################################################################################################

if __name__ == "__main__":
    start_time = time.time()

    # Choose models for the ensemble. Uncomment to choose model needed
    estimator_model0 = RandomForest
    estimator_keywords_model0 = dict(n_estimators=1000, verbose=0, criterion='entropy', n_jobs=-1,
                                     max_features=5, class_weight='auto')

    estimator_model1 = GradBoost
    estimator_keywords_model1 = dict(n_estimators=1000, loss='deviance', learning_rate=0.01, verbose=0, max_depth=5,
                                     subsample=1.0)

    model_names_list = dict(model0=estimator_model0, model1=estimator_model1)
    model_parameters_list = dict(model0=estimator_keywords_model0, model1=estimator_keywords_model1)

    [input_features, output] = telecom_churn(use_synthetic_data=False, feature_scaling=True)

    ensemble_models.majority_voting(input_features, output, model_names_list, model_parameters_list,
                                    run_cv_flag=True, num_model_iterations=1, plot_learning_curve=False,
                                    run_prob_predictions=True, classification_threshold=0.45)

    # prec_recall = ensemble_models.average_prob(input_features, output, model_names_list, model_parameters_list,
    #                                            run_cv_flag=False, num_model_iterations=1, plot_learning_curve=False,
    #                                            run_prob_predictions=True, return_yprob=True,
    #                                            classification_threshold=classification_threshold)

    ##################################
    # Other model
    # estimator = SVC
    # estimator_keywords = dict(C=1, kernel='rbf', class_weight='auto')
    # estimator_model2 = LogReg
    # estimator_keywords_model2 = dict(solver='liblinear')

    # Neural network
    # estimator = NeuralNetClassifier
    # estimator_keywords = dict(layers=[NeuralNetLayer("Rectifier", units=64), NeuralNetLayer("Rectifier", units=32),
    #                                   NeuralNetLayer("Softmax")],
    #                           learning_rate=0.001, n_iter=50)

    ##################################

    print("Total time: %0.3f" % float(time.time() - start_time))
