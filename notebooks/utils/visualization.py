import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from lime.lime_text import LimeTextExplainer


def plot_grid_search_results(cv_results, param_name, metric_names: list[str], title=''):
    fig = plt.figure(figsize=(15, 9))
    ax = fig.add_subplot()

    for metric_name in metric_names:
        ax.errorbar(range(len(cv_results['params'])), cv_results[f'mean_test_{metric_name}'],
                    yerr=3 * cv_results[f'std_test_{metric_name}'], fmt='o-', label=metric_name,
                    elinewidth=3, capsize=3)

    ax.set_xticks(range(len(cv_results['params'])))
    ax.set_xticklabels(['\n'.join(f'{v}' for k, v in param_dict.items()) for param_dict in cv_results['params']],
                       fontsize=10)
    ax.set_xlabel(param_name)
    ax.set_ylabel('Metric values')
    ax.set_title(title)
    fig.legend()
    plt.show()


def plot_confusion_matrix(cm, title='', x_label='', y_label=''):
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot()
    ax = sns.heatmap(cm,
                     linewidth=0.5,
                     cmap=sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True),
                     annot=True,
                     fmt='8g',
                     ax=ax
                     )
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.show()


def plot_coefficients(classifier, feature_names, class_idx, n_top_features=10):
    coef = classifier.coef_[class_idx]
    positive_coefficients = np.argsort(coef)[-n_top_features:]
    negative_coefficients = np.argsort(coef)[:n_top_features]
    interesting_coefficients = np.hstack([negative_coefficients, positive_coefficients])

    plt.figure(figsize=(18, 8))
    plt.subplots_adjust(left=0, bottom=0.2, right=0.95, top=1)
    colors = ['red' if c < 0 else 'blue' for c in coef[interesting_coefficients]]
    plt.bar(np.arange(2 * n_top_features), coef[interesting_coefficients], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(np.arange(0.5, 2 * n_top_features), feature_names[interesting_coefficients], rotation=60,
               ha='right')
    plt.tick_params(axis='x', labelsize=12)


def show_lime(label_id, text, class_names, model, num_features=20):
    explainer = LimeTextExplainer(class_names=class_names)
    explanation = explainer.explain_instance(text,
                                             model.predict_proba,
                                             num_features=num_features,
                                             labels=(label_id,),
                                             )
    explanation.show_in_notebook(text=True)


def plot_validation_curve(*metrics, param_change=None, title='', x_label=''):
    fig = plt.figure(figsize=(15, 9))
    ax = fig.add_subplot()

    for metric_name, metric_values in metrics:
        ax.plot(metric_values, label=metric_name)

    ax.set_xticks(range(len(param_change)))
    ax.set_xticklabels(param_change)
    ax.set_xlabel(x_label)
    ax.set_ylabel('Metric value')
    ax.set_title(title)
    fig.legend()
    plt.show()
