import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.model_selection import cross_val_score


def run_cross_validation(X_train, y_train, hyperparams_list, classifier_constructor, cv=None, scoring=None, plot=None,
                         plot_args=None):
    """
    Use cross-validation to determine set of hyperparameters from options in hyperparams.

    Reference: https://towardsdatascience.com/how-to-find-decision-tree-depth-via-cross-validation-2bf143f0f3d6
    :param X_train: Training features.
    :param y_train: Training response variable.
    :param hyperparams_list: List of hyperparam dicts {param: value}. If plotting afterward, only tune one param at a time.
    :param classifier_constructor: Classifier constructor to run cross-validation on.
    :param cv: Number of folds for cross validation. Defaults to 10.
    :param scoring: Scoring method. Defaults to accuracy.
    :param plot: Plots results using plot_cross_validation if True. Defaults to True if using only one hyperparameter.
        If using more than one hyperparameter, plot_hyperparameter should be specified in plot_args.
    :param plot_args: Dictionary of keyword args to be passed to plot_cross_validation.
    :return: cv_scores_means: mean cross-validation score, cv_scores_stds: standard deviation of cross-validation scores,
        train_scores: list of training scores for each set of hyperparameters, best_hyperparams: hyperparams with
        highest mean_cv_score
    """
    cv = 10 if cv is None else cv
    scoring = 'accuracy' if scoring is None else scoring
    if plot is None:
        plot = True
    elif plot is True and len(hyperparams_list[0].keys() > 1):
        raise Warning("Cannot plot if using more than one hyperparameter.")
    plot_args = {} if plot_args is None else plot_args
    cv_scores_list = []
    cv_scores_stds = []
    cv_scores_means = []
    train_scores = []
    num_runs = len(hyperparams_list)
    i = 1
    for hyperparams in hyperparams_list:
        print('Starting run {} out of {}'.format(i, num_runs))
        model = classifier_constructor(**hyperparams)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
        cv_scores_list.append(cv_scores)
        cv_scores_means.append(cv_scores.mean())
        cv_scores_stds.append(cv_scores.std(ddof=1))
        train_scores.append(model.fit(X_train, y_train).score(X_train, y_train))
        i += 1
    cv_scores_means = np.array(cv_scores_means)
    cv_scores_stds = np.array(cv_scores_stds)
    train_scores = np.array(train_scores)
    if plot:
        _plot_cross_validation(hyperparams_list, cv_scores_means, cv_scores_stds, train_scores, str(scoring),
                               **plot_args)
    return cv_scores_means, cv_scores_stds, train_scores, hyperparams_list[np.argmax(cv_scores_means)], hyperparams_list


def _plot_cross_validation(hyperparams_list, cv_scores_means, cv_scores_stds, train_scores, scoring_name,
                           plot_hyperparam=None, x_ticks=None, confidence_level=None, y_lim_training=None):
    """
    Plot score vs. hyperparam_values for determination of best value for the hyperparameter being examined.
    Only called within run_cross_validation.

    Reference: https://towardsdatascience.com/how-to-find-decision-tree-depth-via-cross-validation-2bf143f0f3d6
    :param hyperparams_list:
    :param cv_scores_means:
    :param cv_scores_stds:
    :param train_scores:
    :param hyperparam_name:
    :param scoring_name:
    :param confidence_level: Confidence level for confidence interval shown on plot. Defaults to .8.
    :param y_lim_training: If True, set y_lim to include training scores. If False, set y_lim based on mean cv scores.
        Defaults to True.
    :return:
    """
    if plot_hyperparam is None and len(list(hyperparams_list[0].keys())) != 1:
        raise Warning('Must specify plot_hyperparam argument when using more than one hyperparameter.')
    y_lim_training = True if y_lim_training is None else y_lim_training
    plot_hyperparam = list(hyperparams_list[0].keys())[0] if plot_hyperparam is None else plot_hyperparam
    x_ticks = [hyperparams[plot_hyperparam] for hyperparams in hyperparams_list] if x_ticks is None else x_ticks
    confidence_level = .8 if confidence_level is None else confidence_level
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    ax.plot(x_ticks, cv_scores_means, '-o', label='mean cross-validation {}'.format(scoring_name), alpha=0.9)
    n = len(cv_scores_means)
    confidence_intervals = np.array([norm.interval(confidence_level,
                                                   loc=cv_scores_means[i], scale=cv_scores_stds[i] / np.sqrt(n))
                                     for i in range(cv_scores_means.shape[0])])
    ax.fill_between(x_ticks, confidence_intervals[:, 0], confidence_intervals[:, 1], alpha=0.2)
    ylim = plt.ylim()
    ax.plot(x_ticks, train_scores, '-*', label='train {}'.format(scoring_name), alpha=0.9)
    title = "{} vs. {}".format(plot_hyperparam, scoring_name)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(plot_hyperparam, fontsize=14)
    ax.set_ylabel(scoring_name, fontsize=14)
    if not y_lim_training:
        ax.set_ylim(ylim)
    ax.set_xticks(x_ticks)
    ax.legend()