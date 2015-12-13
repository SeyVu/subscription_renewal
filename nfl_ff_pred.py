#########################################################################################################
# Description: Function trying to predict fantasy points, yards and TDs for various players based on
# their performance over past many years
#
#########################################################################################################

# Imports for various models (Turn on as needed)
from sklearn.ensemble import RandomForestRegressor as RandomForestReg
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
# sklearn Toolkit
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import os
import time
import logging
import support_functions as sf
import baseline_models_regression as ml_reg
import baseline_models_unsupervised as ml_us

#########################################################################################################
# Global variables
__author__ = "DataCentric1"
__pass__ = 1
__fail__ = 0

#########################################################################################################
# Setup logging

# TODO: Figure out a centralized way to install/handle logging. Does it really need to be instantiated per file?
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(filename)s:%(lineno)4s - %(funcName)15s()] %(levelname)8s: %(message)s')


#########################################################################################################


class MultiColumnLabelEncoder:
    def __init__(self, columns=None):
        # Array of column names to encode
        self.columns = columns

    def fit(self, x, y=None):
        return self  # Not Relevant Here

    def transform(self, x):

        """
        def transform(self,x):
        Transforms columns of x specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in x.
        """

        out = x.copy()
        if self.columns is not None:
            for col in self.columns:
                out[col] = LabelEncoder().fit_transform(out[col])
        else:
            for colname, col in out.iteritems():
                out[colname] = LabelEncoder().fit_transform(col)
        return out

    def fit_transform(self, x, y=None):
        return self.fit(x, y).transform(x)


#########################################################################################################


def prep_reg_nfl_pred(feature_scaling=False, data_csv='data/nfl_pred_data.csv'):
    nfl_df = sf.load_model_data(data_csv)

    col_names = nfl_df.columns.tolist()

    logger.info(sf.Color.BOLD + sf.Color.GREEN + "Column Names:" + sf.Color.END)
    logger.info(col_names)

    to_show = col_names[:]

    logger.info(sf.Color.BOLD + sf.Color.GREEN + "Sample Loaded Data:" + sf.Color.END)
    logger.info(nfl_df[to_show].head(3))

    # Isolate Target Data
    y = np.array(nfl_df['Total points'])

    # Columns to Drop (For Features Data Frame)
    to_drop = ['Total points']
    nfl_feat_space = nfl_df.drop(to_drop, axis=1)

    # Capturing Feature Names
    feature_names = nfl_feat_space.columns.tolist()
    logger.info(sf.Color.BOLD + sf.Color.GREEN + "Feature Names:" + sf.Color.END)
    logger.debug(feature_names)

    # Using Label Encoding to Rebase the Values in these Columns
    nfl_feat_space = MultiColumnLabelEncoder(columns=['name', 'year', 'team']).fit_transform(nfl_feat_space)
    logger.info(sf.Color.BOLD + sf.Color.GREEN + "Sample Data Post Label Encoding:" + sf.Color.END)
    logger.info(nfl_feat_space.head(3))

    # Make NumPy Array
    x = nfl_feat_space.as_matrix().astype(np.float)

    # Handle Feature Scaling and Normalization
    if feature_scaling:
        scaler = StandardScaler()
        x = scaler.fit_transform(x)

    logger.info(sf.Color.BOLD + sf.Color.GREEN + "Sample Transformed Data:" + sf.Color.END)
    logger.info(x[0:3])

    logger.info("Feature Space holds %d Observations and %d Features" % x.shape)

    return [x, y, nfl_df]


##################################################################################################################

def prep_us_nfl_pred(input_df, use_csv=False, data_csv='data/output_dont_commit/reg_output.csv'):

    if use_csv:
        input_df = sf.load_model_data(data_csv)

    # Model Run - K-Means Clustering - Data Preparation
    # Adding Mean Squared Error Column
    input_df['Squared Error'] = input_df.apply(lambda row: ((row['Total points'] -
                                                             row['Total points predicted']) ** 2), axis=1)

    # Dropping Unwanted Columns
    us_columns_to_drop = ['Total points', 'Total points predicted']
    us_input_df = input_df.drop(us_columns_to_drop, axis=1)

    logger.info(sf.Color.BOLD + sf.Color.GREEN + "Sample Clustering Input Data:" + sf.Color.END)
    logger.info(us_input_df.head(3))

    # Using Label Encoding to Rebase the Values in these Columns
    us_input_df = MultiColumnLabelEncoder(columns=['name', 'year', 'team']).fit_transform(us_input_df)
    logger.info(sf.Color.BOLD + sf.Color.GREEN + "Sample Data Post Label Encoding:" + sf.Color.END)
    logger.info(us_input_df.head(3))

    # Converting to NumPy Array
    us_input_npa = us_input_df.as_matrix().astype(np.float)

    return input_df, us_input_npa


##################################################################################################################

if __name__ == "__main__":

    # Machine Learning Chosen Models
    reg_estimator = RandomForestReg
    us_estimator = KMeans

    # To Run or Not To?
    run_reg = False
    run_us = True

    start_time = time.time()

    if run_reg:
        # Model Run - Regression - Random Forest - Estimator Keywords = dict()
        reg_estimator_keywords = dict(n_estimators=10000, verbose=0, warm_start='True', n_jobs=-1,
                                      max_features=5)

        # Model Run - Regression - Data Preparation
        [reg_input_npa, reg_output_npa, reg_df] = prep_reg_nfl_pred(feature_scaling=False,
                                                                    data_csv='data/nfl_pred_data.csv')

        # Model Run - Regression - Random Forest
        reg_pred_df = ml_reg.run_models_wrapper(reg_input_npa, reg_output_npa, run_cv_flag=True, num_model_iterations=1,
                                                plot_learning_curve=False, test_size=0.2, clf_class=reg_estimator,
                                                **reg_estimator_keywords)

        logger.debug('Input Dimensions: %s', reg_df.shape)
        logger.debug('Output Dimensions: %s', reg_pred_df.shape)

        # Model Run - Regression - Output Processing
        # Combine Input & Output Data Frames
        reg_result_df = pd.concat([reg_df, reg_pred_df], axis=1)

        # Model Run - Regression - Data Recording
        # Write Regression Results to CSV
        reg_result_df.to_csv('data/output_dont_commit/reg_output.csv')

    if run_us:
        # Model Run - Clustering - K-Means - Estimator Keywords = dict()
        us_estimator_keywords = dict(init='k-means++', n_init=10, verbose=0)

        if run_reg:
            [reg_result_df, us_input_npa] = prep_us_nfl_pred(reg_result_df, use_csv=False, data_csv='dummy.csv')
        else:
            dummy_df = pd.DataFrame(np.nan, index=[0], columns=['A'])
            [reg_result_df, us_input_npa] = prep_us_nfl_pred(dummy_df, use_csv=True, data_csv='data/output_dont_commit/reg_output.csv')

        # Model Run - K-Means Clustering
        us_kcluster_df = ml_us.run_clustering(us_input_npa, make_plots=False, clf_class=us_estimator, min_cluster=3,
                                              max_cluster=5, **us_estimator_keywords)

        # Model Run - K-Means Clustering - Output Processing
        # Combine Input & Output Data Frames
        us_result_df = pd.concat([reg_result_df, us_kcluster_df], axis=1)

        # Model Run - K-Means Clustering - Data Recording
        # Write Regression Results to CSV
        us_result_df.to_csv('data/output_dont_commit/us_output.csv')

    print("Total time: %0.3f" % float(time.time() - start_time))

##################################################################################################################
