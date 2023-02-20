import argparse
import configparser
import pickle
import sys
sys.path.insert(1, r"C:\Users\kartik\AppData\Local\Packages\CanonicalGroupLimited.UbuntuonWindows_79rhkp1fndgsc\LocalState\rootfs\home\kartik\assignment__1_2\mle-training\logs")
import os
import logger_01 as l
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
logger = l.configure_logger(log_file=os.path.join(r"C:\Users\kartik\AppData\Local\Packages\CanonicalGroupLimited.UbuntuonWindows_79rhkp1fndgsc\LocalState\rootfs\home\kartik\assignment__1_2\mle-training\logs\logging_files","custom_config.log"))

logger.info("Starting Score.py")

def testing_using_strat_test_set(model, imputer, strat_test_set):
    logger.info("Model evalutation : scoring started")
    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()
    logger.info("Dropping Ocean_proximity column")
    X_test_num = X_test.drop("ocean_proximity", axis=1)
    logger.info("Transforming data with the imputer")
    X_test_prepared = imputer.transform(X_test_num)
    logger.info("Creating X_test_prepared dataframe")
    X_test_prepared = pd.DataFrame(
        X_test_prepared, columns=X_test_num.columns, index=X_test.index
    )
    X_test_prepared["rooms_per_household"] = (
        X_test_prepared["total_rooms"] / X_test_prepared["households"]
    )
    X_test_prepared["bedrooms_per_room"] = (
        X_test_prepared["total_bedrooms"] / X_test_prepared["total_rooms"]
    )
    X_test_prepared["population_per_household"] = (
        X_test_prepared["population"] / X_test_prepared["households"]
    )

    X_test_cat = X_test[["ocean_proximity"]]
    X_test_prepared = X_test_prepared.join(
        pd.get_dummies(X_test_cat, drop_first=True)
    )
    logger.info("predicting model")
    model_predictions = model.predict(X_test_prepared)
    logger.info("Calculating Mean Squared Error value")
    model_mse = mean_squared_error(y_test, model_predictions)
    logger.info("Calculating Root Mean Squared Error value")
    model_rmse = np.sqrt(model_mse)
    logger.info("Calculating Mean Absolute error value")
    model_mae = mean_absolute_error(y_test, model_predictions)

    print(f"Model : Mean Squared Error {model_mse}")
    print(f"Model : Root Mean Squared Error {model_rmse}")
    print(f"Model : Mean Squared Error {model_mse}")
    print(f"Model : Mean Squared Error {model_mae}")

if __name__ == "__main__":

    logger.info("Creating Instance of Configparser")
    config = configparser.ConfigParser()

    path = "C:/Users/kartik/AppData/Local/Packages/CanonicalGroupLimited.UbuntuonWindows_79rhkp1fndgsc/LocalState/rootfs/home/kartik/assignment__1_2/mle-training/config.ini"
    config.read(path)
    logger.info("Initializing Variables with config defaults value")
    housing_prepared = config["ProcessedData"]["housing_prepared"]
    housing_labels = config["ProcessedData"]["housing_labels"]
    strat_test_set = config["ProcessedData"]["strat_test_set"]
    model_path = config["ForScoring"]["model_path"]
    imputer_path = config["ForScoring"]["imputer_path"]

    logger.info("ArgumentParser is instanciated")
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
    logger.info("Arguments parsed.")
    housing_prepared = args.housing_prepared
    housing_labels = args.housing_labels
    strat_test_set = args.strat_test_set
    model_path = args.model_path

    print(housing_prepared, housing_labels, strat_test_set, model_path)
    logger.info("Reading csv files with pandas.read_Csv")
    housing_prepared = pd.read_csv(housing_prepared)
    housing_labels = pd.read_csv(housing_labels)
    strat_test_set = pd.read_csv(strat_test_set)

    logger.info("loading pickle models")
    imputer  = pickle.load(open(imputer_path, 'rb'))
    model  = pickle.load(open(model_path, 'rb'))
    logger.info("Calling testing function")
    testing_using_strat_test_set(model, imputer, strat_test_set)