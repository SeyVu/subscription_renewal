#########################################################################################################
#  Contains a collection of unsupervised learning algorithms
#
#########################################################################################################

# sklearn Toolkit
from sklearn.cluster import KMeans

import numpy as np
import pandas as pd

import support_functions as sf

import time
import logging

#########################################################################################################
# Global variables
__author__ = "Ananth Gopalakrishnan"
__pass__ = 1
__fail__ = 0

#########################################################################################################
# Setup logging

# TODO: Figure out a centralized way to install/handle logging. Does it really need to be instantiated per file?
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(filename)s:%(lineno)4s - %(funcName)15s()] %(levelname)8s: %(message)s')

#########################################################################################################


def run_clustering(x, make_plots=False, clf_class=KMeans, min_cluster=3, max_cluster=3, **kwargs):
    """
    Runs any supported clustering algorithms (Support List: K-Means)

    :param x:
    :param make_plots:
    :param clf_class:
    :param min_cluster:
    :param max_cluster:
    :param kwargs:
    :return:
    """

    logger.info("Feature Space Holds %d Observations and %d Features" % x.shape)

    # Run model on test data
    logger.info(sf.Color.BOLD + sf.Color.BLUE + "Runnning K-Means" + sf.Color.END)

    # Creating empty output cluster array
    x_clusters_cumm = np.empty([x.shape[0], 0])
    x_clusters_cols = []
    logger.debug('Dimensions of Output Cluster Array @ Start: %s', x_clusters_cumm.shape)

    for i in range(min_cluster, max_cluster+1):

        logger.info(sf.Color.BOLD + sf.Color.YELLOW + "Number of Clusters: %d" % i + sf.Color.END)
        x_clusters_cols.append(str(i) + ' Cluster Run')

        # Run K-Means Clustering
        clf = clf_class(n_clusters=i, **kwargs)
        clf.fit(x)
        x_clusters = clf.predict(x)

        # Re-shape Clustering Output
        x_clusters = x_clusters.reshape(-1, 1)
        logger.debug('Dimensions of K-Means Run Array: %s', x_clusters.shape)

        # Merge individual cluster runs
        x_clusters_cumm = np.hstack((x_clusters_cumm, x_clusters))
        logger.debug('Dimensions of Output Cluster Array: %s', x_clusters_cumm.shape)

    # TODO: Need to write clustering plot routines
    if make_plots:
        pass

    # Create a Pandas Data Frame
    x_clusters_cumm_df = pd.DataFrame(x_clusters_cumm, columns=x_clusters_cols)

    return x_clusters_cumm_df

##################################################################################################################

if __name__ == "__main__":

    start_time = time.time()

    # Machine Learning Chosen Models
    estimator = KMeans

    # Model Run - Clustering - K-Means - Estimator Keywords = dict()
    estimator_keywords = dict(init='k-means++', n_init=10, verbose=0)

    # Model Run - K-Means Clustering - Data Preparation
    # Load Input Data Frame
    input_df = sf.load_model_data('data/output_dont_commit/reg_output.csv')

    # Dropping Unwanted Columns
    columns_to_drop = ['Total points', 'Total points predicted']
    us_input_df = input_df.drop(columns_to_drop, axis=1)

    # Converting to NumPy Array
    us_input_npa = us_input_df.as_matrix().astype(np.float)

    # Model Run - K-Means Clustering
    us_kcluster_df = run_clustering(us_input_npa, make_plots=False, clf_class=KMeans, min_cluster=3,
                                  max_cluster=3, **estimator_keywords)

    # Model Run - K-Means Clustering - Output Processing
    # Combine Input & Output Data Frames
    us_result_df = pd.concat([input_df, us_kcluster_df], axis=1)

    # Model Run - K-Means Clustering - Data Recording
    # Write Regression Results to CSV
    us_result_df.to_csv('data/output_dont_commit/us_output.csv')

    print("Total time: %0.3f" % float(time.time() - start_time))
