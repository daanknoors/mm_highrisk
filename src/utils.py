"""Utility functions"""
import pandas as pd
from src import config


def add_prefix_to_dict_keys(dict, prefix):
    """Adds a prefix to keys in dictionary"""
    return {f"{prefix}__{k}": v for k, v in dict.items()}


def corr_with_target(df, target_name=None, correlation_method='pearson'):
    """Compute correlation of dataframe to target variable"""
    target_name = config.TARGET if not target_name else target_name

    # compute correlation of all columns with target
    corr_target = df.drop(columns=target_name).corrwith(df[target_name], method=correlation_method)
    # drop NaN's, sort, reset index and rename columns
    corr_target = (corr_target
                   .dropna().sort_values()
                   .reset_index().rename(columns={'index': 'feature', 0: 'correlation'}))
    return corr_target
