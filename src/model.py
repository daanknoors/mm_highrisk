"""Functions to train a machine learning classifier"""
import numpy as np
import pandas as pd
import joblib
from datetime import datetime


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src import config
from src import utils

"""Model training pipelines"""


def train_model(X_train, y_train, pipeline, param_grid, model_name, cv=3, n_iter=10, save=False):
    """Helper function to train randomized search model on training data. Specify pipeline and param_grid."""

    # average_precision is more robust than ROC-AUC when target is imbalanced
    clf = RandomizedSearchCV(pipeline, param_distributions=param_grid, cv=3, n_jobs=3, n_iter=10,
                             scoring='average_precision', refit=True, verbose=1)
    clf.fit(X_train, y_train)

    if save:
        save_model(clf, model_name)
    return clf

def train_multiple_models(X_train, y_train, X_test, y_test, model_names, pipelines, param_grids, cv=3, n_iter=10):
    """Train multiple models"""
    for model_name, pipeline, param_grid in zip(model_names, pipelines, param_grids):
        clf = train_model(X_train, y_train, pipeline, param_grid, model_name=model_name, cv=cv, n_iter=10, save=True)
        print(f'{model_name} - {clf.scoring} test data: {clf.score(X_test, y_test)} \n')



def column_transformer_preprocess(X):
    """Generic preprocessing functions for specific data types"""

    # define feature categories and check if in input dataframe
    features_nominal = [x for x in X.columns if x in config.FEATURES_NOMINAL]
    features_ordinal = [x for x in X.columns if x in config.FEATURES_ORDINAL]
    features_continuous = [x for x in X.columns if x not in config.FEATURES_NOMINAL + config.FEATURES_ORDINAL]

    ct_preprocess = ColumnTransformer([
        ('ohe_nominal', OneHotEncoder(sparse=False), features_nominal),
        ('impute_ordinal', SimpleImputer(missing_values=np.nan, strategy='most_frequent'), features_ordinal),
        ('impute_continuous', SimpleImputer(missing_values=np.nan, strategy='median'), features_continuous)
        # ('impute_numeric', SimpleImputer(missing_values=np.nan, strategy='median'), features_numeric)
    ])
    return ct_preprocess

def column_transformer_select_entrez(X, k=1000):
    """Select entrez codes with anova f test"""
    entrez_columns = [c for c in X.columns if 'Entrez' in c]
    select_entrez = ColumnTransformer([
        ('select_entrez', SelectKBest(f_classif, k=k), entrez_columns)
    ], remainder='passthrough')
    return select_entrez


def get_pipeline_rf(X_train, select_k):
    preprocess_transformer = column_transformer_preprocess(X_train)

    # k = _check_select_k(X_train, select_k)

    pipe = Pipeline([
        ('preprocess', preprocess_transformer),
        ('var_thresh', VarianceThreshold(threshold=0)),
        ('select', SelectKBest(f_classif, k=select_k)),
        # ('select', select_entrez_transformer),
        ('clf', RandomForestClassifier(random_state=config.RANDOM_STATE))
    ])
    return pipe


def get_pipeline_xgb(X_train, select_k):
    preprocess_transformer = column_transformer_preprocess(X_train)

    # k = _check_select_k(X_train, select_k)

    pipe = Pipeline([
        ('preprocess', preprocess_transformer),
        ('var_thresh', VarianceThreshold(threshold=0)),
        ('select', SelectKBest(f_classif, k=select_k)),
        # ('select', select_entrez_transformer),
        ('clf', XGBClassifier(random_state=config.RANDOM_STATE))
    ])
    return pipe


def _check_select_k(X, select_k):
    """check if select k is higher than number of columns, else select all columns"""
    if isinstance(select_k, int):
        select_k = select_k if select_k > X.shape[1] else 'all'
    return select_k


def get_param_grid_rf(prefix=None):
    """RandomForest param grid with option to add prefix"""
    grid = {
        "bootstrap": [True, False],
        "class_weight": ["balanced", "balanced_subsample", None],
        "criterion": ["gini", "entropy"],
        "max_depth": range(5, 250, 5),
        "max_features": ["sqrt", "log2", None],
        "min_samples_leaf": range(2, 30, 2),
        "min_samples_split": range(2, 30, 2),
        "n_estimators": range(100, 2500, 100),
    }

    if prefix:
        grid = utils.add_prefix_to_dict_keys(grid, prefix)
    return grid

def get_param_grid_xgb(prefix=None):
    """XGBoost param grid with option to add prefix"""
    grid = {
        "learning_rate": np.arange(0.05, 1, 0.05),
        "max_depth": np.arange(2, 10, 1),
        "n_estimators": np.arange(10, 500, 10)
    }

    if prefix:
        grid = utils.add_prefix_to_dict_keys(grid, prefix)
    return grid





"""Trained model utilities"""


def get_feature_importance_df(clf, input_features):
    """Create dataframe with features importances"""
    pipeline = clf if isinstance(clf, Pipeline) else clf.best_estimator_

    features = input_features

    # OHE creates additional features
    if 'preprocess' in pipeline.named_steps:
        features = pipeline.named_steps['preprocess'].get_feature_names_out()
        # strip column transformer names from feature
        features = np.array([x.split('__', 1)[1] for x in features])

    # if variance threshold in pipeline, filter out removed features
    if 'var_thresh' in pipeline.named_steps:
        selected_features = pipeline.named_steps["var_thresh"].get_support(indices=True)
        features = features[selected_features]

    # if feature selection in pipeline, filter out non-selected features
    if 'select' in pipeline.named_steps:
        selected_features = pipeline.named_steps["select"].get_support(indices=True)
        features = features[selected_features]

    feature_importance = pipeline.named_steps["clf"].feature_importances_
    df_feature_importance = pd.DataFrame(
        {"feature": features,
         "importance": feature_importance}
    ).sort_values(by="importance", ascending=False).reset_index(drop=True)

    return df_feature_importance

def save_model(model, name):
    filename_model = f"model_{name}_{datetime.today().date().__str__()}.joblib"
    joblib.dump(model, config.PATH_MODEL / filename_model)
    print(f"Model saved to: {config.PATH_MODEL / filename_model}")


def load_model(model_date, name):
    filename_model = f"model_{name}_{model_date}.joblib"
    print(f"Load model from: {config.PATH_MODEL / filename_model}")
    return joblib.load(config.PATH_MODEL / filename_model)