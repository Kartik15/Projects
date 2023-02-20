import argparse
import configparser
import pickle
import sys
import os
sys.path.insert(1, r"C:\Users\kartik\AppData\Local\Packages\CanonicalGroupLimited.UbuntuonWindows_79rhkp1fndgsc\LocalState\rootfs\home\kartik\assignment__1_2\mle-training\logs")

import logger_01 as l
logger = l.configure_logger(log_file=os.path.join(r"C:\Users\kartik\AppData\Local\Packages\CanonicalGroupLimited.UbuntuonWindows_79rhkp1fndgsc\LocalState\rootfs\home\kartik\assignment__1_2\mle-training\logs\logging_files","custom_config.log"))
logger.info("Starting train.py")
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedShuffleSplit,
)
from sklearn.tree import DecisionTreeRegressor


def model_training(
    model_instance, housing_prepared, housing_labels, model_path
):
    logger.info("Training Started")
    model_instance.fit(housing_prepared, housing_labels)
    logger.info("Training COmpleted")
    pred = model_instance.predict(housing_prepared)
    mse = mean_squared_error(housing_labels, pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(housing_labels, pred)
    logger.info(f"{model_instance} algorithm, mse : {mse}, rmse {rmse}, mae {mae}")
    logger.info("Saving model")
    pickle.dump(
        model_instance,
        open(model_path + "/" + str(model_instance)[:-2] + ".pkl", "wb"),
    )


if __name__ == "__main__":

    logger.info("Configparser instantiating")
    config = configparser.ConfigParser()

    path = "C:/Users/kartik/AppData/Local/Packages/CanonicalGroupLimited.UbuntuonWindows_79rhkp1fndgsc/LocalState/rootfs/home/kartik/assignment__1_2/mle-training/config.ini"
    config.read(path)
    logger.info("Initializing Default parameters")
    housing_prepared = config["ProcessedData"]["housing_prepared"]
    housing_labels = config["ProcessedData"]["housing_labels"]
    strat_test_set = config["ProcessedData"]["strat_test_set"]
    model_path = config["ProcessedData"]["model_path"]

    logger.info("Instantiating Argument Parser")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--housing_prepared", help="Enter the output folder path to store the dataset", default=housing_prepared
    )
    parser.add_argument(
        "--housing_labels",
        help="Enter the  processed data folder path to store the \
         dataset ready for modeling and inference",default=housing_labels

    )
    parser.add_argument(
        "--strat_test_set",
        help="Enter the  processed data folder path to store the \
         dataset ready for modeling and inference",default=strat_test_set

    )
    parser.add_argument(
        "--model_path",
        help="Enter the  processed data folder path to store the \
         dataset ready for modeling and inference",default=model_path

    )
    args = parser.parse_args()

    housing_prepared = args.housing_prepared
    housing_labels = args.housing_labels
    strat_test_set = args.strat_test_set
    model_path = args.model_path

    logger.info("Reading CSV files from default parameters")
    housing_prepared = pd.read_csv(housing_prepared)
    housing_labels = pd.read_csv(housing_labels)
    strat_test_set = pd.read_csv(strat_test_set)

    logger.info("Calling Model_Training Function")
    model_training(
        LinearRegression(), housing_prepared, housing_labels, model_path
    )

    model_training(
        DecisionTreeRegressor(random_state=42), housing_prepared, housing_labels, model_path
    )

    model_training(
        RandomForestRegressor(n_estimators=100, random_state=42), housing_prepared, housing_labels, model_path
    )

    logger.info("Execution completed\n**********************************************************")