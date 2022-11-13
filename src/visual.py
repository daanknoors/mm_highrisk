"""Visualization methods"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import precision_recall_curve, average_precision_score, RocCurveDisplay, ConfusionMatrixDisplay
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, roc_auc_score


from src import model
from src import gene

def plot_roc_curve(clf, X_test, y_test, name='Classifier'):
    """Plot ROC Curve"""
    fig, ax = plt.subplots()
    sns.despine()
    RocCurveDisplay.from_estimator(clf, X_test, y_test, name=name, ax=ax)
    ax.plot(
        [0, 1], [0, 1], linestyle="--", lw=1, color="black", alpha=0.7, label="No skill"
    )
    plt.legend()
    plt.title("ROC Curve")
    plt.show()


def plot_pr_curve(y_true, y_pred_prob,  marker="."):
    """Plot Precision-Recall Curve"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_prob)
    avg_precision = average_precision_score(y_true, y_pred_prob)

    # plot the roc curve for the model
    no_skill = len(y_true[y_true == 1]) / len(y_true)
    plt.figure()
    sns.despine()
    plt.plot(
        recall,
        precision,
        marker=marker,
        lw=1,
        alpha=0.7,
        label=f"Classifier (Average Precision: {avg_precision:.2f}",
    )
    plt.plot(
        [0, 1],
        [no_skill, no_skill],
        lw=1,
        linestyle="--",
        color="black",
        alpha=0.7,
        label="No skill",
    )

    # axis labels
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()

    # show the plot
    plt.show()


def plot_feature_importances(clf, input_features, top=50, replace_gene_descriptions=False):
    """Plot feature importances of classifier.
    Requires that classifier has feature importance attribute"""

    df_feature_importance = model.get_feature_importance_df(clf, input_features=input_features)

    # for the top 1500 EntrezIDS we scraped the actual name, rename if possible.
    if replace_gene_descriptions:
        gene_symbol_dict = gene.get_gene_dict()
        df_feature_importance['feature'] = df_feature_importance['feature'].replace(gene_symbol_dict)

    if top < 25:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig, ax = plt.subplots(figsize=(8, 8))

    sns.set()
    sns.despine()
    sns.barplot(
        data=df_feature_importance[:top],
        x="importance",
        y="feature",
        ax=ax,
        dodge=False,
    )

    plt.tight_layout()
    plt.show()


def plot_top_corrwith_target(corr_target, n_top=25, n_bottom=25):
    """Plot top correlations of columns with target"""

    # select top and bottom correlations
    corr_high = corr_target[-n_top:]
    corr_low = corr_target[:n_bottom]
    top_corr = pd.concat([corr_low, corr_high])

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.despine()
    sns.barplot(top_corr, y='feature', x='correlation', ax=ax)
    plt.title('Top correlations to target variable')
    plt.xlabel('correlation')
    plt.ylabel('feature')
    plt.show()


def plot_confusion_matrix(clf, X_test, y_test, labels, normalize=None):
    """Plot confusion matrix"""
    fig, ax = plt.subplots()
    ax.grid(False)
    cm_display = ConfusionMatrixDisplay.from_estimator(
        clf,
        X_test,
        y_test,
        display_labels=labels,
        cmap=plt.cm.Blues,
        normalize=normalize,
        ax=ax,
    )
    return cm_display


def model_evaluation_report(clf, X_test, y_test):
    """Create a comprehensive model evaluation report that combines several
    metrics and visualizations."""

    # print score test data with scoring function used for training
    print(f'{clf.scoring} test data: {clf.score(X_test, y_test)}')

    # get the y predictions with default threshold 0.5 and the prediction probabilities
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)

    # compute roc_auc and balanced accuracy
    print(f'roc_auc test data: {roc_auc_score(y_test, y_pred_proba[:, 1])}')
    print(f'balanced_accuracy test data: {balanced_accuracy_score(y_test, y_pred)}\n')

    # print classification report
    print(classification_report(y_test, y_pred))

    # show params of best performing classifier
    print(f'best params: {clf.best_params_}')

    # create plots
    plot_feature_importances(clf, input_features=list(X_test.columns), top=50, replace_gene_descriptions=True)
    plot_roc_curve(clf, X_test, y_test)
    plot_pr_curve(y_test, y_pred_proba[:, 1], marker=None)
    plot_confusion_matrix(clf, X_test, y_test, labels=[0, 1])
