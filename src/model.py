"""Functions to train and inspect machine learning classifiers"""
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from tempfile import mkdtemp

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.decomposition import PCA
from xgboost import XGBClassifier

from src import config
from src import utils

"""Model training"""


def train_model(X_train, y_train, pipeline, param_grid, model_name, cv=3, n_iter=10, n_jobs=4, verbose=1, save=False):
    """Helper function to train randomized search model on training data.
    Specify pipeline and param_grid and model name."""
    clf = RandomizedSearchCV(pipeline, param_distributions=param_grid, cv=cv, n_jobs=n_jobs, n_iter=n_iter,
                             scoring=config.SCORING_FUNCTION, refit=True, verbose=verbose)
    clf.fit(X_train, y_train)

    if save:
        save_model(clf, model_name)
    return clf


def train_multiple_models(X_train, y_train, X_test, y_test, model_names, pipelines, param_grids,
                          cv=3, n_iter=10, n_jobs=4, verbose=1):
    """Helper function to train multiple models sequentially"""
    clfs = []
    for model_name, pipeline, param_grid in zip(model_names, pipelines, param_grids):
        clf = train_model(X_train, y_train, pipeline, param_grid, model_name=model_name,
                          cv=cv, n_iter=n_iter, n_jobs=n_jobs, verbose=verbose, save=True)
        print(f'{model_name} - {clf.scoring} test data: {clf.score(X_test, y_test)} \n')
        clfs.append(clf)
    return clfs


"""Transformer pipelines"""


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
    ])
    return ct_preprocess


def column_transformer_preprocess_pca(X):
    """Generic preprocessing functions for specific data types, with PCA for continuous variables."""

    # define feature categories and check if in input dataframe
    features_nominal = [x for x in X.columns if x in config.FEATURES_NOMINAL]
    features_ordinal = [x for x in X.columns if x in config.FEATURES_ORDINAL]
    features_continuous = [x for x in X.columns if x not in config.FEATURES_NOMINAL + config.FEATURES_ORDINAL]

    pca_pipeline = Pipeline([
        ('remove_constant', VarianceThreshold(threshold=0)),
        ('impute_continuous', SimpleImputer(missing_values=np.nan, strategy='median')),
        ('scale_continuous', RobustScaler()),
        ('pca', PCA(random_state=config.RANDOM_STATE))
    ])

    ct_preprocess = ColumnTransformer([
        ('ohe_nominal', OneHotEncoder(sparse=False), features_nominal),
        ('impute_ordinal', SimpleImputer(missing_values=np.nan, strategy='most_frequent'), features_ordinal),
        ('pca_continuous', pca_pipeline, features_continuous)
    ])
    return ct_preprocess


def get_pipeline_transformers_baseline(X_train):
    """Baseline Pipeline of transformer functions without classifier
    Reference MM Dream Challenge article (Mason et al, 2020)
    Note: model and pipeline not specified. Except that the input data only consists of four features"""
    assert set(X_train.columns) == set(config.FEATURES_MINIMAL), \
        f'Baseline model can only have these 4 features: {config.FEATURES_MINIMAL}, input data has {X_train.columns}'

    # get standard transformer
    preprocess_transformer = column_transformer_preprocess(X_train)

    pipe = Pipeline([
        ('preprocess', preprocess_transformer)
    ])
    return pipe


def get_pipeline_transformers_select(X_train, select_k):
    """Feature Selection: pipeline of transformer functions without classifier."""
    preprocess_transformer = column_transformer_preprocess(X_train)

    # add cache directory to Pipeline memory to prevent Transformers
    # without parameter changes from re-computing every iteration
    cachedir = mkdtemp()

    pipe = Pipeline([
        ('preprocess', preprocess_transformer),
        ('select1', VarianceThreshold(threshold=0)),
        ('select2', SelectKBest(f_classif, k=select_k)),
        ('scale', RobustScaler()),
        ('select3', RFECV(RandomForestClassifier(random_state=config.RANDOM_STATE), step=2, cv=3,
                          scoring=config.SCORING_FUNCTION, min_features_to_select=500))
    ], memory=cachedir)
    return pipe


def get_pipeline_transformers_pca(X_train):
    """Dimensionality Reduction Pipeline of transformer functions without classifier"""
    preprocess_transformer_pca = column_transformer_preprocess_pca(X_train)

    pipe = Pipeline([
        ('preprocess', preprocess_transformer_pca)
    ])
    return pipe


"""Classifiers"""


def add_clf_rf(pipe):
    """Add RandomForest Classifier to pipeline"""
    pipe.steps.append(('clf', RandomForestClassifier(class_weight='balanced', random_state=config.RANDOM_STATE)))
    return pipe


def add_clf_xgb(pipe):
    """Add XGBoost Classifier to pipeline"""
    pipe.steps.append(('clf', XGBClassifier(random_state=config.RANDOM_STATE)))
    return pipe


"""Parameter grids"""

def get_param_grid_rf(prefix=None):
    """RandomForest param grid with option to add prefix"""
    grid = {
        "bootstrap": [True, False],
        "criterion": ["gini", "entropy"],
        "max_depth": range(2, 100, 5),
        "max_features": ["sqrt", "log2"],
        "min_samples_leaf": range(2, 15, 2),
        "min_samples_split": range(2, 15, 2),
        "n_estimators": range(10, 1000, 50),
    }

    if prefix:
        grid = utils.add_prefix_to_dict_keys(grid, prefix)
    return grid

def get_param_grid_xgb(prefix=None):
    """XGBoost param grid with option to add prefix"""
    grid = {
        "learning_rate": np.arange(0.05, 1, 0.05),
        "max_depth": np.arange(2, 100, 5),
        "n_estimators": np.arange(10, 1000, 50)
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

    # reduce features for every feature selection step
    select_steps = [c for c in pipeline.named_steps if 'select' in c]
    if select_steps:
        for s in select_steps:
            selected_features = pipeline.named_steps[s].get_support(indices=True)
            features = features[selected_features]

    # if pca is applied, take output features
    if 'pca' in pipeline.named_steps:
        features = pipeline.named_steps["pca"].get_feature_names_out()

    # create feature importance dataframe
    feature_importance = pipeline.named_steps["clf"].feature_importances_
    df_feature_importance = pd.DataFrame(
        {"feature": features,
         "importance": feature_importance}
    ).sort_values(by="importance", ascending=False).reset_index(drop=True)
    return df_feature_importance


def save_model(model, name):
    """Save classifier by name and date"""
    filename_model = f"model_{name}_{datetime.today().date().__str__()}.joblib"
    joblib.dump(model, config.PATH_MODEL / filename_model)
    print(f"Model saved to: {config.PATH_MODEL / filename_model}")


def load_model(model_date, name):
    """Load classifier by name and date"""
    filename_model = f"model_{name}_{model_date}.joblib"
    print(f"Load model from: {config.PATH_MODEL / filename_model}")
    return joblib.load(config.PATH_MODEL / filename_model)