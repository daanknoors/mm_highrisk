"""Data preprocess functions for modeling"""
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


def load_clinical_data():
    return pd.read_csv(config.PATH_DATA / config.FILENAME_CLINICAL_DATA)

def load_expression_data():
    return pd.read_csv(config.PATH_DATA / config.FILENAME_EXPRESSION_DATA)

def load_dictionary_data():
    return pd.read_csv(config.PATH_DATA / config.FILENAME_DICTIONARY_DATA)


def preprocess_clinical_data(df):
    """Preprocessing steps for clinical data"""
    df = df.copy()

    # remove records where HR_FLAG == 'CENSORED' and convert column to binary
    df = df[df['HR_FLAG'] != 'CENSORED']
    df['HR_FLAG'] = df['HR_FLAG'].replace({'FALSE': 0, 'TRUE': 1})

    # remove columns with only missing values
    df = df.dropna(axis=1, how='all')

    # # remove additional labels that correlate with target: OS and PFS
    # df.drop(columns=config.TARGET_EXTRA, inplace=True)
    # # todo remove D_OS_FLAG and D_PFS_FLAG as well?

    return df


def preprocess_expression_data(df):
    """Preprocessing steps for gene expression data"""
    df = df.copy()

    # transpose dataset
    df = df.set_index('Unnamed: 0').T.reset_index().rename_axis(None, axis=1)

    # rename columns
    df.columns = ['SampleID'] + ['Entrez_' + str(c) for c in df.columns[1:]]
    df.columns = df.columns.astype(str)

    # remove columns with only missing values
    df = df.dropna(axis=1, how='all')
    return df


def create_model_input_data(keep_features=False):
    """Preprocess and merge all data files and create model input"""
    df_clinical_raw = load_clinical_data()
    df_clinical = preprocess_clinical_data(df_clinical_raw)

    df_expression_raw = load_expression_data()
    df_expression = preprocess_expression_data(df_expression_raw)

    df_model = df_clinical.merge(df_expression, left_on='RNASeq_geneLevelExpFileSamplId', right_on='SampleID', how='inner')

    # drop features prior to modeling
    df_model = df_model.drop(columns=config.FEATURES_DROP, errors='ignore')

    # restrict features to user-specified list
    if isinstance(keep_features, list):
        df_model = df_model[keep_features]
    return df_model