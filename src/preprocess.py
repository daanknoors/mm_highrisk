"""Data preprocess functions for modeling"""
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


"""Load data files"""


def load_clinical_data():
    return pd.read_csv(config.PATH_DATA_RAW / config.FILENAME_CLINICAL_DATA)


def load_expression_data():
    return pd.read_csv(config.PATH_DATA_RAW / config.FILENAME_EXPRESSION_DATA)


def load_dictionary_data():
    return pd.read_csv(config.PATH_DATA_THESAURI / config.FILENAME_DICTIONARY_DATA)


def load_training_data():
    return pd.read_csv(config.PATH_DATA_PREPROCESSED / config.FILENAME_TRAINING_DATA)


def load_testing_data():
    return pd.read_csv(config.PATH_DATA_PREPROCESSED / config.FILENAME_TESTING_DATA)


"""Pre-process data"""


def preprocess_clinical_data(df):
    """Preprocessing steps for clinical data"""
    df = df.copy()

    # remove records where HR_FLAG == 'CENSORED' and convert column to binary
    df = df[df['HR_FLAG'] != 'CENSORED']
    df['HR_FLAG'] = df['HR_FLAG'].replace({'FALSE': 0, 'TRUE': 1})
    return df


def preprocess_expression_data(df):
    """Preprocessing steps for gene expression data"""
    df = df.copy()

    # transpose dataset
    df = df.set_index('Unnamed: 0').T.reset_index().rename_axis(None, axis=1)

    # rename columns
    df.columns = ['SampleID'] + ['Entrez_' + str(c) for c in df.columns[1:]]
    return df


def create_model_input_data(keep_features=False, save=False):
    """Preprocess and merge all data files and create model input"""
    df_clinical_raw = load_clinical_data()
    df_clinical = preprocess_clinical_data(df_clinical_raw)

    df_expression_raw = load_expression_data()
    df_expression = preprocess_expression_data(df_expression_raw)

    df_model = df_clinical.merge(df_expression, left_on='RNASeq_geneLevelExpFileSamplId', right_on='SampleID', how='inner')

    # drop redundant features prior to modeling
    df_model = df_model.drop(columns=config.FEATURES_DROP, errors='ignore')

    # remove columns with only missing values
    df_model = df_model.dropna(axis=1, how='all')

    # optional: restrict features to user-specified list
    if isinstance(keep_features, list):
        df_model = df_model[keep_features]

    df_train, df_test = train_test_split(df_model, test_size=0.2, stratify=df_model[config.TARGET],
                                         random_state=config.RANDOM_STATE)
    if save:
        df_train.to_csv(config.PATH_DATA_PREPROCESSED / 'mm_highrisk_train.csv', index=False, sep=',')
        df_test.to_csv(config.PATH_DATA_PREPROCESSED / 'mm_highrisk_test.csv', index=False, sep=',')
        print(f'Train and test data saved at {config.PATH_DATA_PREPROCESSED}')
    return df_train, df_test


def split_x_y(df, y_column=None):
    """Split X and y in dataframe"""
    if not y_column:
        y_column = config.TARGET

    y = df[y_column]
    X = df.drop(columns=[y_column])
    return X, y
