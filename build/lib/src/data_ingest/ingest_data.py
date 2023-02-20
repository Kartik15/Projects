import argparse, configparser
import os
import tarfile
import pickle
import sys
sys.path.insert(1, r"C:\Users\kartik\AppData\Local\Packages\CanonicalGroupLimited.UbuntuonWindows_79rhkp1fndgsc\LocalState\rootfs\home\kartik\assignment__1_2\mle-training\logs")

import logger_01 as l
import numpy as np
import pandas as pd
from six.moves import urllib
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.impute import SimpleImputer
logger = l.configure_logger(log_file=os.path.join(r"C:\Users\kartik\AppData\Local\Packages\CanonicalGroupLimited.UbuntuonWindows_79rhkp1fndgsc\LocalState\rootfs\home\kartik\assignment__1_2\mle-training\logs\logging_files","custom_config.log"))

logger.info("Starting ingest_data.py")

def fetch_housing_data(housing_url, housing_path):
    logger.info("Fetching data started")
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    logger.info(f"tgz file is downloaded at : {housing_path} ")


def load_housing_data(housing_path):
    logger.info("Loading the housing path")
    csv_path = housing_path+"/housing.csv"
    print(csv_path)
    logger.info("Returning the housing_path dataframe")
    return pd.read_csv(csv_path)


def income_cat_proportions(data):
    logger.info("Calculating the income_cat_propoertions")
    return data["income_cat"].value_counts() / len(data)


def data_preprocessing(housing, processed_datapath):
    logger.info("Spliting the DataSet")
    train_set, test_set = train_test_split(
        housing, test_size=0.2, random_state=42
    )


    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    logger.info("Spliting with StratifiedShuffleSplit")
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
    logger.info("Stratified Shuffle split completed")
    train_set, test_set = train_test_split(
        housing, test_size=0.2, random_state=42
    )

    logger.info("Creating DataFrame for data preprocessing with income_cat_proportions")
    compare_props = pd.DataFrame(
        {
            "Overall": income_cat_proportions(housing),
            "Stratified": income_cat_proportions(strat_test_set),
            "Random": income_cat_proportions(test_set),
        }
    ).sort_index()
    logger.info("Rand>%error column creating in compare_props variable")
    compare_props["Rand. %error"] = (
        100 * compare_props["Random"] / compare_props["Overall"] - 100
    )
    compare_props["Strat. %error"] = (
        100 * compare_props["Stratified"] / compare_props["Overall"] - 100
    )

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    housing = strat_train_set.copy()
    housing.plot(kind="scatter", x="longitude", y="latitude")
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

    corr_matrix = housing.corr()
    corr_matrix["median_house_value"].sort_values(ascending=False)
    housing["rooms_per_household"] = (
        housing["total_rooms"] / housing["households"]
    )
    housing["bedrooms_per_room"] = (
        housing["total_bedrooms"] / housing["total_rooms"]
    )
    housing["population_per_household"] = (
        housing["population"] / housing["households"]
    )

    housing = strat_train_set.drop(
        "median_house_value", axis=1
    )  # drop labels for training set
    housing_labels = strat_train_set["median_house_value"].copy()
    logger.info("Creating an instance of SimpleImputer")
    imputer = SimpleImputer(strategy="median")

    housing_num = housing.drop("ocean_proximity", axis=1)
    logger.info("Dropping ocean_proximity Column")
    imputer.fit(housing_num)
    logger.info("imputer fitted on housing_num")
    X = imputer.transform(housing_num)

    housing_tr = pd.DataFrame(
        X, columns=housing_num.columns, index=housing.index
    )
    housing_tr["rooms_per_household"] = (
        housing_tr["total_rooms"] / housing_tr["households"]
    )
    housing_tr["bedrooms_per_room"] = (
        housing_tr["total_bedrooms"] / housing_tr["total_rooms"]
    )
    housing_tr["population_per_household"] = (
        housing_tr["population"] / housing_tr["households"]
    )

    housing_cat = housing[["ocean_proximity"]]
    housing_prepared = housing_tr.join(
        pd.get_dummies(housing_cat, drop_first=True)
    )

    logger.info(f"Storing the processed data at {processed_datapath}")
    housing_prepared.to_csv(
        os.path.join(processed_datapath, "housing_prepared.csv"), index=False
    )
    housing_labels.to_csv(
        os.path.join(processed_datapath, "housing_labels.csv"), index=False
    )
    strat_test_set.to_csv(
        os.path.join(processed_datapath, "strat_test_set.csv"), index=False
    )
    logger.info(f"Dumping Imputer for testing use @ {processed_datapath}")
    pickle.dump(
            imputer,
            open(processed_datapath + "/imputer.pkl", "wb"),
    )
    return True

if __name__ == "__main__":

    config = configparser.ConfigParser()
    logger.info("Cnfig parser initiated")
    path = 'C:/Users/kartik/AppData/Local/Packages/CanonicalGroupLimited.UbuntuonWindows_79rhkp1fndgsc/LocalState/rootfs/home/kartik/assignment__1_2/mle-training/config.ini'
    config.read(path)
    logger.info("Reading Config Parameters")

    output_conf_path = config['Address']['output_path']
    processed_conf_path = config['Address']['processed_datapath']
    logger.info("Config Parameters Reading is completed")

    logger.info("Initiating the Argument Parser")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_path", help="Enter the output folder path to store the dataset", default=output_conf_path
    )
    parser.add_argument(
        "--processed_datapath",
        help="Enter the  processed data folder path to store the \
         dataset ready for modeling and inference",default=processed_conf_path

    )
    args = parser.parse_args()

    output_path = args.output_path
    processed_datapath = args.processed_datapath

    logger.info("Initializing Variables for the Data")

    DOWNLOAD_ROOT = (
        "https://raw.githubusercontent.com/ageron/handson-ml/master/"
    )
    HOUSING_PATH = os.path.join(output_path, "housing")
    HOUSING_URL = DOWNLOAD_ROOT + f"datasets/housing/housing.tgz"

    # calling the data download function
    logger.info("Functional Call to fetch_housing_data")
    fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH)
    logger.info("Function call to fetching_housing_data completed")

    logger.info("Loading Housing Data")
    housing_data = load_housing_data(housing_path=HOUSING_PATH)

    logger.info("Data Preprocessing Started")
    data_preprocessing(
        housing=housing_data, processed_datapath=processed_datapath
    )
    logger.info("Data Preprocessing Completed")
    logger.info("*********************************************************************")