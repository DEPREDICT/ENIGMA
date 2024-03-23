import pickle
import os
from datetime import datetime
from collections.abc import Iterable, MutableSet
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from neurocombat_sklearn import CombatModel
from sklearn.model_selection import train_test_split
import json
import csv
import torch
from ipywidgets import IntProgress, HTML, VBox, Layout
from IPython.display import display
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from torch.nn import BCEWithLogitsLoss
import random
import time
from glob import glob
from classes import cohorts, colors
import seaborn as sns
from pingouin import ttest, rm_anova, compute_effsize, bayesfactor_ttest
from scipy import stats
from sklearn.linear_model import LinearRegression
from pathlib import Path
np.seterr(divide='ignore')


def count_values(d: dict, i=0):
    """
    Recursively counts items in a nested dictionary.

    Args:
        d (dict): The input dictionary.
        i (int, optional): Counter for items. Defaults to 0.

    Returns:
        int: Total count of items in the nested dictionary.
    """
    if not isinstance(d, dict):
        return i + 1
    else:
        return i + sum(count_values(v) for v in d.values())


def flatten_dict(nested_dict):
    res = {}
    if isinstance(nested_dict, dict):
        for k in nested_dict:
            flattened_dict = flatten_dict(nested_dict[k])
            for key, val in flattened_dict.items():
                key = list(key)
                key.insert(0, k)
                res[tuple(key)] = val
    else:
        # res[()] = nested_dict
        res[()] = nested_dict,  # altered this to make calculations easier
    return res


def nested_dict_to_df(values_dict: dict) -> pd.DataFrame:
    """
    Concerts a nested dictionary to a Pandas DataFrame object using
     MultiIndexing for each nested data frame level.

    :param values_dict: A dictionary
    :return:

    Example Usage:
    ```python
    nested_dict = {
        'Level1A': {
            'Level2A': {
                'Value1': 1,
                'Value2': 2,
            },
            'Level2B': {
                'Value1': 3,
            }
        },
        'Level1B': {
            'Level2A': {
                'Value2': 4,
            }
        }
    }
    nested_dict_to_df(nested_dict)
    ```
    This returns
    ```
                     Value1  Value2
    Level1A Level2A     1.0     2.0
            Level2B     3.0     NaN
    Level1B Level2A     NaN     4.0
    ```
    """
    flat_dict = flatten_dict(values_dict)
    df = pd.DataFrame.from_dict(flat_dict, orient="index")
    df.index = pd.MultiIndex.from_tuples(df.index)
    df = df.unstack(level=-1)
    df.columns = df.columns.map("{0[1]}".format)
    return df


def collect_stats_per_site(data: pd.DataFrame):
    bool_fmt = lambda v: f'{np.sum(v):,.0f} '.ljust(3) + f'({np.mean(v):,.1%})'
    perc_fmt = lambda v: f'{np.mean(v):,.1%} ± {np.std(v):,.1%}'
    con1_fmt = lambda v: f'{np.mean(v):,.1f} ± {np.std(v):,.1f}'
    con2_fmt = lambda v: f'{np.mean(v):,.2f}'.ljust(5) + f' ± {np.std(v):,.2f}'

    col_formatters = {  # `col_formatters` can be used for printing Pandas DataFrames nicely.
        'Cohort': '{}'.format,
        'Numel': '{:,.0f}'.format,
        'ADcur': lambda v: f'{np.mean(v):,.1f} ± {np.std(v):,.1f}',
        'Age': lambda v: f'{np.mean(v):,.1f} ± {np.std(v):,.1f}',
        'Age_of_Onset': con1_fmt,
        'Normalized_Pretreatment_Severity': con2_fmt,
        'Response_percentage': perc_fmt,
        'is_extreme': bool_fmt,
        'is_female': bool_fmt,
        'is_recurrent': bool_fmt,
        'is_remitter': perc_fmt,
        'is_responder': perc_fmt,
        'uses_SSRI': bool_fmt,
        'uses_SNRI': bool_fmt,
        'uses_ATYP': bool_fmt,
    }

    stat_cols = list(col_formatters)[2:]

    site_stats = None  # print site_means, we use site_stats later
    site_means = [['Total', len(data), *[np.mean(data[stat_col]) for stat_col in stat_cols]]]

    # For every cohort subpopulation for each statistic columns, derived from the col_formatters
    for cohort in set(data.cohort):
        subjs = data.loc[[i for i in data.index if cohort in i]]
        cohort_name = cohorts.get(cohort).name
        if site_stats is None:
            site_stats = [['Total', len(data), *[data[stat_col] for stat_col in stat_cols]]]
        site_stat = [cohort_name, len(subjs), *[subjs[stat_col] for stat_col in stat_cols]]
        site_means.append([cohort_name] + [np.mean(stat) for stat in site_stat[1:]])
        site_stats.append(site_stat)
    site_stats = pd.DataFrame(site_stats, columns=col_formatters).set_index('Cohort')

    # Let is create another version of this table with site statistics, where we appropriately format everything
    # for printing.
    site_means = pd.DataFrame(columns=col_formatters,
                              data=[['Total'] + [col_formatters[col](site_stats.loc['Total'][col]) for col in
                                                 site_stats.columns]] +
                                   [[cohort.name] + [col_formatters[col](site_stats.loc[cohort.name][col]) for col in
                                                     site_stats.columns] for
                                    cohort in cohorts if cohort.name in site_stats.index], ).set_index('Cohort')
    site_stats = site_stats.drop('Total')
    return site_stats, site_means


def normalize_scores(data_frame, verbose=True, drop_old=False):
    """
    1) Read the three symptom scoring columns
    2) Normalizes: centers them around zero and scales it to one standard deviation.
    3) It then selects what is available in order: MADRS, HDRS, BDI
    4) Insert it into the dataframe as Normalized_Pretreatment_Severity
    5) show scatter plots for subj_thc that have two outcome scores and display Pearson's Correlation Coefficient.
    6) Drop old MADRS_pre and MADRS_norm columns if drop_old is True
    :param data_frame:    pandas data frame containing syptom scores as columns
    :param verbose:       whether to show a figure of the normalization procedure
    :param drop_old:      whether to retain the old MADRS_pre and MADRS_pre_norm columns in the data frame
    :return:
    """
    # 1) Read the three symptom scoring columns
    c_cols = 'MADRS_pre', 'HDRS_pre', 'BDI_pre'
    cn_cols = [c_col + '_norm' for c_col in c_cols]

    # 2) Normalizes: centers them around zero and scales it to one standard deviation.
    for c_col, cn_col in zip(c_cols, cn_cols):
        norm = data_frame[c_col] - data_frame[c_col].min()
        norm = norm - norm.mean()
        norm = norm / norm.std()
        data_frame.insert(loc=0, column=cn_col, value=norm)

    # 3) It then selects what is available in order: MADRS, HDRS, BDI
    normalized_score = data_frame.MADRS_pre_norm.fillna(data_frame.HDRS_pre_norm).fillna(data_frame.BDI_pre_norm)

    # 4) Insert it into the dataframe as Normalized_Pretreatment_Severity
    data_frame.insert(
        loc=0,
        column='Normalized_Pretreatment_Severity',
        value=normalized_score,
    )

    # 5) show scatter plots for subj_thc that have two outcome scores and display Pearson's Correlation Coefficient.
    if verbose:
        fig, axes = plt.subplots(1, 4, figsize=(9, 3))
        axes = axes.reshape(-1)
        for i in range(3):
            clin_ab = list(cn_cols[i:] + cn_cols[:i])[:-1]
            two_clin_data = data_frame.loc[data_frame[clin_ab].notna().sum(axis=1) > 1][clin_ab]
            clin_a, clin_b = ['n-' + clin.split("_")[0] for clin in clin_ab]
            two_clin_data = pd.DataFrame(data=two_clin_data.values, columns=[clin_a, clin_b])
            p_r = two_clin_data.corr().values[1, 0]
            sns.scatterplot(data=two_clin_data, x=clin_a, y=clin_b, ax=axes[i]).set(
                title=f'{clin_a}-{clin_b} r={p_r:.2f}')

    # 6) Drop old MADRS_pre and MADRS_norm columns if drop_old is True
    if drop_old:
        data_frame = data_frame.drop(columns=cn_cols + cn_cols)

    if verbose:
        sns.histplot(data=data_frame['Normalized_Pretreatment_Severity'], ax=axes[-1]).set(
            xlabel='Normalized Pretreatment\n Symptom Score')
        fig.tight_layout()
    else:
        fig = None
    return data_frame, fig


def load_proj_df(data: pd.DataFrame, property='Thickness', dim=32, channels=3, verbose=False) -> pd.DataFrame:
    """
    :param data:      Pandas Dataframe containing subjects that have been included. Using the index of Data makes sure that not ALL .gii files are loaded.
    :param property:  The projection property that is read as string. Options are 'Thickness' and 'Area'.
    :param dim:       Dimensions of the loaded array as integer. Corresponds with the data that has been created, but it is generally 32 or 128.
    :param channels:  Number of channels in the output array. 1 for a data frame or 3 for deep learning.
    :param verbose:   if verbose, the load function withh show three example files
    :return:          returns a Pandas Data Frame with pixels as named column items, and raw array.
    """
    sbj, arr, _ = load_surface_projections(data, channels=channels, dim=dim, property=property, verbose=verbose)
    arr = arr[:, :, 1:-1, 1:-1]
    vec = arr[:, 0]
    col = flatten([[f'{a},{b}' for b in range(vec.shape[-1])] for a in range(vec.shape[-2])])
    vec = vec.reshape([len(vec), vec.size // len(vec)])  # Flatten
    df = pd.DataFrame(data=vec, index=sbj, columns=col)
    return df, arr


class Timer(object):
    """
    A simple timer context manager for measuring code execution time.

    Attributes:
    name (str): Name to identify the timer (optional).

    Methods:
    __init__(self, name=None): Initialize the Timer instance with an optional name.
    __enter__(self): Start the timer when entering a context.
    __exit__(self, type, value, traceback): Print elapsed time when exiting the context.

    Usage:
    with Timer('Task Name'):
        # Code to be timed
    """

    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name, )
        print('Elapsed: %s' % (time.time() - self.tstart))


def corr_rep_kfold_cv_test(a: list, b: list, n_splits: int, n_samples: int) -> tuple[float, float]:
    """
        Implementation of Bouckaert and Franks (2004) corrected repeated k-fold cv test.

        This function calculates a corrected test statistic and p-value for comparing
        the performance of two models using the corrected repeated k-fold cross-validation
        test described by Bouckaert and Franks (2004).

        Parameters:
        a (list): List of performance scores for model A.
        b (list): List of performance scores for model B.
        n_splits (int): Number of splits/folds in the cross-validation.
        n_samples (int): Total number of instances in the dataset.

        Returns:
        tuple[float, float]: Corrected test statistic and corresponding p-value.
    """
    r = len(a) // n_splits  # number of r-times repeats as integer
    k = n_splits  # number of k-folds as integer
    n1 = n_samples // n_splits * (n_splits - 1)  # number of instances used for training
    n2 = n_samples // n_splits  # number of instances used for testing
    x = np.subtract(a, b)  # observed differences
    m = x.mean()  # mean estimate
    s = np.sum((x - m) ** 2) / (k * r - 1)  # variance estimate
    t_stat = m / np.sqrt((1 / (k * r) + n2 / n1) * s)  # corrected test statistic
    p_val = stats.t.sf(np.abs(t_stat), n_splits - 1) * 2  # p-value
    return t_stat, p_val


def get_rgb_cbar(n_shades=200, bar_width=25) -> np.array:
    """
    Return a numpy array with colors going from Red to Blue, through Green
    :param n_shades:    Number of shades in the color bar (higher is more fine-grained)
    :param bar_width:   Width of the array (usefull for formatting the color bar without adjusting matplotlib axes
    :return:            A np.array of shape (n_shades x bar_wdith x 3)
    """
    shades = np.zeros((n_shades, 3))
    for i, c in enumerate(np.linspace(0, 1, n_shades)):
        r = clamp(2 * c - 1, 0, 1)
        b = clamp(1 - 2 * c, 0, 1)
        g = clamp(0.7 - (r + b), 0, 1)  # g stands for gray
        r += g
        b += g
        shades[i, :] = r, g, b
    colorbar = np.flipud(np.moveaxis(np.dstack([shades] * bar_width), 1, 2))
    return colorbar


def load_surface_projections(data, dim=32, channels=3, property='Thickness', verbose=False, ):
    """
    --- DataLoader for Cortical Surface data ---
    Since we can actually easily fit all training data in memory!
    :param dim: image dimensions of data loaded, 32 pixels by default
    :param channels: number of dimensions of output, 3 by default
    :param verbose: plot examples yes or no
    :return: numpy array shaped (n_samples, n_channels, img_height, img_width)
    """

    files = glob(get_root('data', 'derivatives', f'*_hem-LH_map-{property}_flat_res-{dim}_Projection.npy'))
    # files = glob(get_root('data', 'derivatives', '*npatches-6_res-32_Projection.npy'))
    files = flatten([[f for f in files if idx.replace('SanRaffaele', 'surf') in f] for idx in data.index])
    subs = [f'{os.path.basename(f).split("__")[0][4:]}|{f[f.find("H_") - 1]}H' for f in files]
    subs = [s.replace('SanRaffaele_surf', 'SanRaffaele') for s in subs]

    # Load data
    surface_projections = None
    for i, file in enumerate(tqdm(files)):
        array = np.load(file)
        if surface_projections is None:
            surface_projections = np.zeros((len(files), *array.shape))
        surface_projections[i] = array
    if surface_projections.shape[1] > 3:
        surface_projections = surface_projections[:, :3, :, :]  # This removes all channels above 3
    surface_projections = surface_projections / surface_projections.max()

    # Display three examples
    if verbose:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for i, ax in enumerate(axes.reshape(-1)):
            sub_name, img_name = os.path.basename(files[i]).split('__')
            hemisphere = img_name.split('-')[1].split('_')[0]
            ax.imshow(surface_projections[i][0], cmap='jet')
            ax.set(title=f'{sub_name[4:]}, {hemisphere} Thickness ({i}/{len(surface_projections)})')
            ax.set_aspect('equal', 'box')
            ax.grid(False)
    else:
        fig = None
    surface_projections = surface_projections[:, :channels].squeeze()  # set number of channels
    return subs, surface_projections, fig


def clean_data_df(data_df, verbose=True, drop_old_symptom_scores=False):
    data_df.insert(loc=0, column='cohort', value=[i.split('_')[0] for i in data_df.index])
    data_df.insert(loc=0, column='cohort_idx', value=[cohorts.get(c).idx for c in data_df.cohort])
    data_df.insert(loc=data_df.columns.get_loc('Age') + 1, column='Age2', value=data_df.Age ** 2)
    data_df = data_df.drop(columns=[c for c in data_df.columns if 'Site' in c])  # Empty or covered by 'Cohort'
    data_df.drop(columns=['PHQ9_pre', 'PHQ9_post'])  # Empty

    # Remplace empty cells by NaN
    data_df = data_df.replace(' ', np.nan)

    # Rename and set proper type of bools
    bools = {'Sex': 'female',
             'Recur': 'recurrent',
             'AD': 'ad',  # anti depressant use lifetime
             'Rem': 'rem',
             'Responder': 'responder',
             'Remitter': 'remitter',
             'filter_$': None,
             'Responder_remitter': None}

    if 'ICV' in data_df:
        # The following fields are present in ICV data: 123456 (str), nan (float), 123.456.789 (str)
        # Cleaning this is was a bit tricky since .replace does not work on floats and not al strs should be / 1000
        data_df['ICV'] = flatten(
            [[int(i.replace('.', '')) // 1000 if '.' in i else int(i)] if isinstance(i, str) else [i] for i in
             data_df['ICV']])

    for key, value in bools.items():
        if value is not None:
            new_items = data_df[key].astype(float)
            if new_items.max() > 1:
                new_items = new_items.subtract(1)
            data_df.insert(loc=0, column='is_' + value, value=new_items.astype(bool))
        data_df = data_df.drop(columns=key)

    # Set proper type of floats
    floats = ['MADRS_post', 'HDRS_post', 'BDI_post', 'Treatment_duration', 'ADcur', 'Epi', 'MADRS_pre', 'HDRS_pre',
              'BDI_pre', 'Sev', 'AO', 'Age']
    for f in floats:
        data_df[f] = data_df[f].astype(float)

    # Clean some more complex columns (column separated lists of strings, and comma separated decimals
    clean_list_of_strings = lambda s: s.replace(' ', '').split(',')
    data_df.Treatment_type = data_df.Treatment_type.apply(clean_list_of_strings)
    data_df.Treatment_type_category = data_df.Treatment_type_category.apply(clean_list_of_strings)
    data_df.Response_percentage = data_df.Response_percentage.apply(lambda s: float(s.replace(',', '.')) / 100)
    data_df['Age_of_Onset'] = data_df.AO
    data_df.drop(columns=['AO'])
    data_df, normalize_fig = normalize_scores(data_df, verbose, drop_old=drop_old_symptom_scores)
    return data_df, normalize_fig


def stacked_hist(data_frame, site_stats, name='Age', n_bins=30, ax=None, make_legend=True):
    """
    :param data_frame:
    :param site_stats:
    :param name:
    :return:
    """
    bins = np.linspace(data_frame[name].min(), data_frame[name].max(), n_bins)
    step = bins[1] - bins[0]
    x = bins[:-1] + step
    histograms = {cohort: np.histogram(stat[name], bins=bins)[0] for cohort, stat in site_stats.iterrows()}

    # Plot histograms as stacked bar graphs
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    for z_level, cohort in enumerate(cohorts):
        if cohort.name in histograms:
            if not cohort.idx:
                cumulative = histograms[cohort.name]
            else:
                cumulative = cumulative + histograms[cohort.name]
            n_subjs = len(site_stats.loc[cohort.name][name])
            ax.bar(
                x=x,
                height=cumulative,
                zorder=len(cohorts) - z_level,
                color=cohort.color,
                label=f'{"" if make_legend else "_"} {cohort.name} (n={n_subjs})',
                width=step,
                lw=0.1,
                edgecolor=colors.dark
            )
    # Make pretty
    if make_legend:
        ax.get_figure().legend(bbox_to_anchor=(0.900, 0.879))
    ax.set_title(f'Stacked Histogram of {name} by Cohort (n={len(data_frame)})')
    ax.set_xlim(data_frame[name].min() + step / 2, data_frame[name].max() + step / 2)
    ax.set_xlabel(name)
    ax.set_ylabel('Count')
    return ax


def flatten(lst: list) -> list:
    """
    Flatten a nested list of iterators into a single list.
    Does not handle multiple levels of nesting!

    This function takes a list of lists and flattens it into a single list,
    where each element from the sublists is included in the resulting list.

    Parameters:
    lst (list): The list of lists to be flattened.

    Returns:
    list: A flattened list containing elements from the input list of lists.

    Example:
    input_list = [[1, 2, 3], [4, 5], [6]]
    flattened_list = flatten(input_list)  # Returns [1, 2, 3, 4, 5, 6]
    """
    return [item for sublist in lst for item in sublist]


def clamp(n, minn=0, maxn=1):
    """
    Clamp a value or a list of values within a specified range.

    This function clamps a given value or a list of values to be within a specified
    minimum and maximum range.

    Parameters:
    n (float or iterable): The value or list of values to be clamped.
    minn (float, optional): The minimum value in the range (default is 0).
    maxn (float, optional): The maximum value in the range (default is 1).

    Returns:
    float or list: The clamped value or list of values.

    Example Usage:
    1. Clamping a single value:
    ```
    result = clamp(0.5, 0, 1)  # Clamps the value to be between 0 and 1
    ```

    2. Clamping a list of values:
    ```
    values = [0.2, 1.5, -0.1, 0.8]
    clamped_values = clamp(values, 0, 1)  # Clamps all values to be between 0 and 1
    ```

    Note:
    - If 'n' is a list, the function recursively clamps each element in the list.
    - If 'n' is a scalar, the function clamps the value within the specified range.

    """
    if isinstance(n, Iterable):
        return [clamp(m, minn, maxn) for m in n]
    else:
        return max(min(maxn, n), minn)


def accuracy(input, target):
    return torch.sum(torch.argmin(input.data, 1) == target[:, 0]) / len(target)


class TorchTrainer(BaseEstimator):
    """
    TorchTrainer is a custom scikit-learn compatible wrapper for PyTorch models,
     enabling easy training, evaluation, and prediction.

    Parameters:
    ----------
    subestimator : torch.nn.Module
        The PyTorch model to be trained and used for prediction.

    epochs : int, optional (default=10)
        Number of training epochs.

    batch_size : int, optional (default=16)
        Batch size used for training.

    patience : int, optional (default=0)
        Number of epochs with no improvement after which training will be stopped early.

    val_size : float, optional (default=0.2)
        Fraction of the training data to be used as validation set.

    optimizer : torch.optim.Optimizer, optional (default=None)
        Optimizer to use during training. If None, Adam optimizer is used.

    criterion : torch.nn.Module, optional (default=None)
        Loss function to be used during training. If None, Binary Cross Entropy with Logits (BCEWithLogitsLoss) is used.

    metrics : tuple, optional (default=None)
        Metrics to be tracked during training. If None, accuracy is used.

    device : str or torch.device, optional (default=None)
        Device to use for training and prediction. If None, 'cuda' is used if available, otherwise 'cpu'.

    random_state : int, optional (default=None)
        Seed for reproducibility.

    verbose : bool, optional (default=True)
        If True, display a progress bar during training.

    Attributes:
    ----------
    scores : dict
        Dictionary to store training and validation scores for each epoch.

    best_state : dict or None
        The state of the model with the lowest validation loss.

    lowest_loss : float
        The lowest validation loss encountered during training.

    Methods:
    ----------
    fit(X, y, random_state=None)
        Fit the PyTorch model to the training data.

    reinstate(state_dict=None)
        Restore the model to a specific state.

    track_performance(outputs, label, is_train)
        Track and store performance metrics during training.

    check_is_fitted()
        Check if the model has been fitted.

    score(X, y)
        Compute the score for the given input data and target values.

    predict_outp(X)
        Predict the output for the given input data.

    predict_proba(X)
        Predict probabilities for the given input data.

    predict(X)
        Predict the target values for the given input data.

    """

    def __init__(
            self,
            subestimator,
            epochs: int = 10,
            batch_size: int = 16,
            patience: int = 0,
            val_size: float = 0.2,
            optimizer=None,
            criterion=None,
            metrics: tuple = None,
            device: str = None,
            random_state: int = None,
            verbose: bool = True,
    ):
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
            torch.manual_seed(random_state)

        # Decorate with default values
        self.scores = {'train': {}, 'validation': {}}
        self.best_state = None
        self.lowest_loss = np.inf

        # Store attributes
        self.val_size = val_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.metrics = metrics
        self.device = device
        self.subestimator = deepcopy(subestimator)
        self.optimizer = optimizer
        self.criterion = criterion
        self.random_state = random_state
        self.verbose = verbose

    def _split_batches(self, *arrays):
        l = tuple(set([len(a) for a in arrays]))
        if len(l) > 1:
            raise ValueError(f'Unequal array sizes of {l}')
        else:
            l = l[0]
        for ndx in range(0, l, self.batch_size):
            yield [a[ndx:min(ndx + self.batch_size, l)] for a in arrays]

    def _prep(self, *iterables):
        def _as_classes(*iterables):
            # Stacks any 1D iterable into 2D to be compatible with binary classes
            return (i if len(i.shape) > 1 else torch.stack(
                (replace_in_arr(i, {a: b for a, b in zip(i.unique(), i.unique().logical_not())}), i), 1) for i in
                    iterables)

        def _as_tensor(*iterables):
            # Conversion to PyTorch tensor type
            return (i if isinstance(i, torch.Tensor) else torch.tensor(i) for i in iterables)

        def _on_device(*iterables):
            # Send to torch.Device
            return (i.to(self.device) for i in iterables)

        def _as_float(*iterables):
            # Convert tensor type to float
            return (i.float() for i in iterables)

        welcomed = _as_float(*_on_device(*_as_classes(*_as_tensor(*iterables))))
        if len(iterables) == 1:
            return next(welcomed)
        else:
            return welcomed

    def fit(self, X, y, random_state=None):
        """
        Fit PyTorch Model.

        Parameters
        ----------
        X : {PyTorch Tensor} of shape (n_samples, n_channels, n_dims...)
            Training data.

        y : array-like of shape (n_samples,) or (n_samples,)
            Target values. Will be cast to X's dtype if necessary.

        val_size : Size of validation partition. See sklearn
            train_test_split docs. Default is 25%.

        random_state : integer to control from randomness when
            partitioning into training and validation sets

        Returns
        -------
        self : object
            Fitted Estimator.
        """
        random_state = random_state if random_state is not None else self.random_state

        if self.verbose:
            f = IntProgress(min=0, max=self.epochs + self.patience)  # instantiate the bar
            display(f)  # display the bar
        else:
            f = classmethod(lambda _: _)

        patience_remaining = self.patience
        epochs_remaining = self.epochs
        # Split into training and testing partitions
        if self.val_size is not None:
            X_train, X_val, y_train, y_val = train_test_split(
                *self._prep(X, y),
                test_size=self.val_size,
                random_state=random_state)
        else:
            X_train, y_train = self._prep(X, y)

        while epochs_remaining + patience_remaining > 0:
            # Tracking and providing updates on progress
            patience_remaining = clamp(patience_remaining - 1, minn=0, maxn=np.inf)
            epochs_remaining = clamp(epochs_remaining - 1, minn=0, maxn=np.inf)
            f.value = self.epochs - epochs_remaining + self.patience - patience_remaining
            self.subestimator.train()
            for X_batch, y_batch in self._split_batches(X_train, y_train):
                # PyTorch breaks with single channel input
                if len(X_batch) > 1:
                    # Train
                    self.optimizer.zero_grad()
                    outputs = self.subestimator(X_batch)
                    self.track_performance(outputs, y_batch, is_train=True)
                    loss = self.criterion(outputs, y_batch)
                    loss.backward()
                    self.optimizer.step()

            if self.val_size is not None:
                self.subestimator.eval()
                with torch.no_grad():
                    outputs = self.subestimator(X_val)
                    self.track_performance(outputs, y_val, is_train=False)
                    loss = self.criterion(outputs, y_val)
                    found_loss = loss.item()
                if found_loss < self.lowest_loss:
                    self.lowest_loss = found_loss
                    self.best_state = self.subestimator.state_dict()
                    if self.patience or epochs_remaining > 0:
                        patience_remaining = self.patience
        if self.val_size is not None:
            self.reinstate()
        self._is_fitted_ = True
        return self

    def reinstate(self, state_dict=None):
        if state_dict is None:
            state_dict = self.best_state
        self.subestimator.load_state_dict(state_dict)

    def track_performance(self, outputs, label, is_train):
        for metric in (self.criterion,) + self.metrics:
            part = 'train' if is_train else 'validation'
            if hasattr(metric, '__name__'):
                name = metric.__name__
            else:
                name = metric._get_name()
            if name not in self.scores[part]:
                self.scores[part][name] = []
            score = metric(input=outputs, target=label).item()
            self.scores[part][name].append(score)

    def check_is_fitted(self):
        if hasattr(self, '_is_fitted_'):
            return
        else:
            msg = (
                "This %(name)s instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this estimator."
            )
            raise NotFittedError(msg % {"name": type(self.subestimator).__name__})

    def score(self, X, y):
        outputs = self.predict_outp(X)
        score = self.metrics[0](outputs, self._prep(y)).item()
        return score

    def predict_outp(self, X):
        self.check_is_fitted()
        self.subestimator.eval()
        with torch.no_grad():
            return self.subestimator(self._prep(X))

    def predict_proba(self, X):
        # TODO: SignError check with outp
        outputs = self.predict_outp(X)
        # outputs = self.predict_outp(X)
        n_classes = outputs.shape[1]
        even = outputs - torch.stack([outputs.mean(1)] * n_classes, 1)
        norm = even / torch.stack([even.var(1)] * n_classes, 1)
        proba = torch.sigmoid(norm)
        return proba

    def _decision_function(self, outputs):
        # TODO: SignError check with outp
        # return torch.argmin(outputs.data, 1)
        return torch.argmax(outputs.data, 1)

    def predict(self, X):
        return self._decision_function(self.predict_outp(X))

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if isinstance(device, str):
            device = torch.device(device)
        self._device = device

    @property
    def metrics(self):
        return self._metrics

    @metrics.setter
    def metrics(self, metrics=None):
        if metrics is None:
            self._metrics = accuracy,

    @property
    def criterion(self):
        return self._criterion

    @criterion.setter
    def criterion(self, criterion=None):
        if criterion is None:
            self._criterion = BCEWithLogitsLoss()

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        if optimizer is None:
            optimizer = torch.optim.Adam(params=self.subestimator.parameters())
        self._optimizer = optimizer

    @property
    def subestimator(self):
        return self._subestimator

    @subestimator.setter
    def subestimator(self, model):
        self._subestimator = model.to(self.device)


def safe_dict_get(start_dict: dict, *args: str) -> dict:
    """
    Safely retrieve or add a value within a nested dictionary using provided keys.

    This function helps prevent key errors when accessing nested dictionaries. If a key is missing,
    it will be added to the dictionary hierarchy, allowing safe assignment of a value to the deepest key.

    Parameters:
    ----------
    start_dict : dict
        The initial dictionary to retrieve or add values to.

    *args : str
        A variable number of keys specifying the path to the desired value within the nested dictionary.

    Returns:
    -------
    dict
        A reference to the deepest dictionary where the value is stored or can be assigned.

    Example Usage:
    ```python
    my_dict = {'one': {'value': 1}, 'two': {'value': 2}}
    reference = safe_dict_get(my_dict, 'three', 'value')
    reference['value'] = 3
    print(my_dict)  # Output: {'one': {'value': 1}, 'two': {'value': 2}, 'three': {'value': 3}}
    ```
    """
    read_dict = start_dict
    for arg in args:
        if arg not in read_dict:
            v = None if arg is args[-1] else {}
            read_dict[arg] = v
        # Keep a reference, to the dict; prevent a copy
        if arg is not args[-1]:
            read_dict = read_dict[arg]
    return read_dict


def torch_val_score(pipeline, X: np.array, y: pd.Series, cv, groups, verbose=False, return_pipeline='none'):
    """
    A cross_val_score implementation for the TorchTrainer class.
    Only difference is that this method returns a dict of outcomes, and not just a float score.
    The dict contains all scores (train, test, val, null) and the trained model.
    """
    if return_pipeline not in ['none', 'full', 'aggregate']:
        raise ValueError(f'torch_val_score received unacceptable value "{return_pipeline}" for "return_pipeline".\n'
                         'Acceptable arguments are:\n'
                         '- none        do not return any pipeline information\n'
                         '- full        return all pipeline objects\n'
                         '- aggregate   return train and validation scores as NumPy array\n')
    # Apply verbosity
    progress = tqdm if verbose else lambda x: x

    # Preallocate results storage
    scores = {'train': [], 'validation': [], 'test': [], 'null': []}
    if return_pipeline == 'full':
        scores['pipeline'] = []

    # Run cross validation
    for train_index, test_index in progress(cv.split(X, y=groups, groups=groups)):
        pipeline_copy = deepcopy(pipeline)
        # Get data
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        # Fit
        pipeline_copy.fit(X_train, y_train)
        # Get measures
        best_val = max(pipeline_copy.scores['validation']['accuracy'])
        scores['test'].append(pipeline_copy.score(X_test, y_test))
        scores['validation'].append(best_val)
        scores['train'].append(
            pipeline_copy.scores['train']['accuracy'][pipeline_copy.scores['validation']['accuracy'].index(best_val)])
        if y.mean() == 0.5:
            null_score = np.mean(y_test)
        else:
            null_score = np.mean(y_test == y_train.mode()[0])
        scores['null'].append(null_score)

        # Store pipeline
        if return_pipeline == 'full':
            scores['pipeline'].append(pipeline_copy)
        elif return_pipeline == 'aggregate':
            for part, dct_a in pipeline_copy.scores.items():
                for metr, values in dct_a.items():
                    prev_result = safe_dict_get(scores, 'pipeline', part, metr)
                    cur_result = np.array([values], )
                    if prev_result[metr] is None:
                        prev_result[metr] = cur_result
                    else:
                        # Make sure array sizes match when aggregating
                        ext_sz = prev_result[metr].shape[-1]
                        new_sz = cur_result.shape[-1]
                        if new_sz != ext_sz:
                            filler = np.empty((1, np.abs(prev_result[metr].shape[-1] - cur_result.shape[-1])))
                            filler[:] = np.nan
                            if ext_sz > new_sz:
                                cur_result = np.hstack((cur_result, filler))
                            else:
                                filler = np.repeat(filler, len(prev_result[metr]), axis=0)
                                prev_result[metr] = np.hstack((prev_result[metr], filler))
                        prev_result[metr] = np.concatenate((prev_result[metr], cur_result))
    return scores


def replace_in_arr(array, to_replace={}, inplace=False):
    """
    Replaces value in array similar to pandas.DataFrame.replace.
    Compatibility tested with NumPy Arrays and PyTorch Tensors.
    """
    memo = {b: array == a for a, b in to_replace.items()}
    if not inplace:
        array = deepcopy(array)
    for b, idxs in memo.items():
        array[idxs] = b
    return array


def json_in(source: str):
    with open(source, 'r') as file:
        return json.load(file)


def json_out(obj: dict, target: str):
    with open(target, 'w') as file:
        json.dump(obj, file, indent=4, sort_keys=True)


class PipeWrapper:
    """
        --- Wraps Sklearn Pipeline Components to preserve DataFrame information ---
        Some Sklearn pipeline components strip your beautiful DataFrame of Index and Column information.
        This is information you might need in a later step.
        This wrapper preserves this information.
    """

    def __init__(self, pipe):
        self.pipe = pipe()

    def fit(self, X, y=None):
        return self.pipe.fit(X, y)

    def transform(self, X):
        return pd.DataFrame(self.pipe.transform(X), index=X.index, columns=X.columns)

    def fit_transform(self, X, y=None):
        self.pipe.fit(X, y)
        return self.transform(X)


class CombatWrapper:
    """
    This Wrapper makes the NeuroCombat module useful for in SKLearn pipelines.
    """

    def __init__(self, data: pd.DataFrame, discrete_covariates=[], continuous_covariates=[]):
        # Clean input
        if isinstance(discrete_covariates, str):
            discrete_covariates = [discrete_covariates]
        if isinstance(continuous_covariates, str):
            continuous_covariates = [continuous_covariates]

        # Set attributes
        self.data = data
        self.transformer = CombatModel()
        self.dc = discrete_covariates
        self.cc = continuous_covariates

    def _get_kwargs(self, X):
        # Remove hemisphere tag information to match the index with the covariates data sheet
        xindex = [idx.split('|')[0] for idx in X.index]

        # Create a list of sites as int
        sites = [x.split("_")[0] for x in xindex]
        site2idx = {k: v for k, v in zip(set(sites), range(len(set(sites))))}
        site_idx = [[site2idx[i]] for i in sites]

        # Return the appropriate discrete and continuous covariates
        disc_cov = np.array([self.data.loc[xindex][dc].values.tolist() for dc in self.dc]).T if any(self.dc) else None
        cont_cov = np.array([self.data.loc[xindex][cc].values.tolist() for cc in self.cc]).T if any(self.cc) else None
        return {'sites': site_idx, 'discrete_covariates': disc_cov, 'continuous_covariates': cont_cov}

    def fit_transform(self, X, y=None):
        """Fit to data, then transform it"""
        return self.fit(X).transform(X)

    def fit(self, X, y=None):
        self.transformer.fit(X, **self._get_kwargs(X))
        return self

    def transform(self, X):
        return pd.DataFrame(self.transformer.transform(X, **self._get_kwargs(X)), index=X.index, columns=X.columns)


class RegressorWrapper:
    def __init__(self, data: pd.DataFrame, continuous_covariates=[]):
        # Clean input
        if isinstance(continuous_covariates, str):
            continuous_covariates = [continuous_covariates]

        # Set attributes
        self.data = data
        self.continuous_covariates = continuous_covariates
        self.transformers = {cc: LinearRegression() for cc in continuous_covariates}

    def _get_args(self, X):
        # Remove hemisphere tag information to match the index with the covariates data sheet
        xindex = [idx.split('|')[0] for idx in X.index]

        # Create a list of sites as int
        sites = [x.split("_")[0] for x in xindex]
        site2idx = {k: v for k, v in zip(set(sites), range(len(set(sites))))}
        site_idx = [[site2idx[i]] for i in sites]

        # Return the appropriate discrete and continuous covariates
        cont_cov = np.array([self.data.loc[xindex][cc].values.tolist() for cc in self.continuous_covariates]).T if any(self.continuous_covariates) else None
        return site_idx, cont_cov

    def fit_transform(self, X, y=None):
        """Fit to data, then transform it"""
        return self.fit(X).transform(X)

    def fit(self, X, y=None):
        for cc, transformer in self.transformers.items():
            site_idx, cont_cov = self._get_args(X)
            transformer.fit(cont_cov, X)
        return self

    def transform(self, X):
        residuals = []
        for cc, transformer in self.transformers.items():
            site_idx, cont_cov = self._get_args(X)
            X_pred = transformer.predict(cont_cov)
            residuals.append(X_pred)
        return X - np.sum(residuals, axis=0)


def to_device(*data, device=None):
    if device is None:
        """ Pick GPU if available, else CPU """
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    """ Move tensor(s) to chosen device """
    data_on_device = []

    for dat in data:
        try:
            data_on_device.append(dat.to(device, non_blocking=True))
        except AttributeError:
            data_on_device.append(torch.tensor(dat).to(device, non_blocking=True))
    """ Unpack singular data """
    if len(data_on_device) == 1:
        return data_on_device[0]
    else:
        return data_on_device


def read_any_csv(dfp, **kwargs):
    # Automatically snifs CSV dialect
    with open(dfp) as csvfile:
        return pd.read_csv(dfp, dialect=csv.Sniffer().sniff(csvfile.readline()))


def get_joined_df(*df_paths, nan_threshold=10, drop_nondemographics=True, drop_site=True, verbose=True):
    # Read all dataframes
    dfs = []
    for dfp in df_paths:
        dfs.append(read_any_csv(dfp, index_col='SubjID'))

    # Join all data frames
    df = pd.concat(dfs, axis=1, join="inner")

    # Adjust Sex to binary
    df.Sex -= 1

    # Dropping columns
    if drop_nondemographics:
        df.drop(columns=['Recur', 'AD', 'Rem', 'AO', 'Sev', 'BDI', 'HDRS',
                         'Epi', 'ADcur', 'Antidepressant_Use_Lifetime'], inplace=True)
    if drop_site:
        df.drop(columns=[c for c in df.columns if 'Site' in c], inplace=True)

    # Drop Rows from X if more column values are missing than nan_threshold
    if nan_threshold is not None:
        excl_subjs = df.isna().sum(axis=1) > nan_threshold
        if verbose:
            print('Excluded {:.1%} (>{} NaNs)'.format(np.sum(excl_subjs) / len(df), nan_threshold))
        df.drop(index=[idx for idx, is_na in excl_subjs.items() if is_na], inplace=True)

    # Impute Columns where data is missing
    if verbose:
        print('Features for which more than 2% was imputed:')
        n_nan_series = df.isna().sum(axis=0).sort_values()

        for name, n_nans in [(name, n_nans) for name, n_nans in n_nan_series.iteritems() if n_nans / len(df) > 0.02]:
            print('   ', name.ljust(40), '{:.1%}'.format(n_nans / len(df)))
    df.fillna(df.select_dtypes(include=np.number).median(), inplace=True)
    return df


def load_and_join_dfs(*df_paths: str) -> pd.DataFrame:
    # Read all dataframes
    dfs = []
    for dfp in df_paths:
        with open(dfp) as csvfile:
            dfs.append(pd.read_csv(dfp, dialect=csv.Sniffer().sniff(csvfile.readline()), index_col='SubjID'))

    # Join all data frames
    df = pd.concat(dfs, axis=1, join="inner")
    return df


def sanitize_storage_path(storage_path=None, desc=None, ext='') -> str:
    """
        Takes input as string and sanitizes it for:
        1) No dir, nor fname given: 	Use default fname
        2) Just dir, no fname given:	Combine dir with default fname
        3) Dir and fname given:			Check for fname extension
        Returns sanitized storage_path as string.
    """

    # Setup a default file name
    tm_stamp = datetime.now().strftime('%Y%m%d-%H%M')
    if desc is None:
        default_fname = desc + tm_stamp + ext
    else:
        default_fname = desc + ext

    # Sanitize input:
    if not storage_path:
        # 1) If no storage location was provided:
        #    Store at location using default filename
        return default_fname
    elif os.path.isdir(storage_path):
        # 2) If only a storage directory was provided:
        #    Store at this location using default filename
        return os.path.join(storage_path, default_fname)
    else:
        # 3) If everything was provided:
        #    Check file name extension and save this location
        if os.extsep not in storage_path:
            storage_path += ext
        return storage_path


class Logger:
    """
		Logger works like the Python 3+ print function, but
		it also writes to a file and has handy logging modules like:
		- timestr: returns the current time as formatted string
		- collect debug information
	"""

    def __init__(self, storage_path=None, desc='Logger'):
        self.storage_path = sanitize_storage_path(storage_path, desc=desc, ext='.txt')
        with open(self.storage_path, 'a') as log_file:
            print(timestr(), 'log stored at', self.storage_path)
            log_file.write(timestr() + ' log stored at ' + self.storage_path + '\n')

        self.debug = {}

    def __call__(self, *args):
        # First, print the log entry
        print(timestr(), *args)
        # Then write the log entry to file
        with open(self.storage_path, 'a') as log_file:
            log_file.write(' '.join([timestr()] + [str(arg) for arg in list(args)]) + '\n')


class Squeeze(torch.nn.Module):
    """ Small simple function that can be used to squeeze subestimator output in nn.Sequentials"""

    def forward(self, x):
        return torch.squeeze(x)


def timestr() -> str: return datetime.now().strftime('%H:%M')


class PathStr(str):
    """
    Sits somewhere in between a str and Pathlib.path object
    Unlike Pathlib.path it does allow for string mutations like .format and .replace
    Unlike str it allows for easy append of child locations.
    While it does not offer all Pathlib.path functionality, it's good enough for me.
    """

    def __new__(cls, *args):
        return str.__new__(cls, os.path.join(*args))

    def append(self, *args):
        return PathStr(os.path.join(self, *args))


def get_root(*args) -> str:
    return PathStr(str(Path.cwd()), *args)


def pickle_in(source: str) -> dict:
    with open(source, 'rb') as file:
        return pickle.load(file)


def pickle_out(obj: dict, target: str):
    with open(target, 'wb') as file:
        pickle.dump(obj, file)


def dumb_md_formatter(*args, spacing=8) -> str:
    return '|'.join([arg.center(spacing) for arg in args])


def subset_score(subset, alpha=0.05):
    sep = ';'
    formats = '{:.1%}', '{:.1%}', '{:.1%}', '{}', '{}', '{:.3f}'

    broken = False
    if not subset.empty:
        if not subset["score"].empty and not subset['score'].isna().all():
            sub_score = flatten(subset["score"])
            sub_null = flatten(subset["null"])
            if np.isnan(sub_score).all(): broken = True
        else:
            broken = True
    else:
        broken = True

    if broken:
        formatted = sep * (len(formats) - 1)
    else:
        sub_balanced_score = np.mean(sub_score) / np.mean(sub_null) / 2
        _, p_val = stats.combine_pvalues(subset['pvalue'].dropna().astype(float))
        t_stat = np.mean(subset['tstat'])
        notice = '*' if p_val < alpha and np.mean(sub_score) > np.mean(sub_null) else ' '

        p_val = stats.t.sf(np.abs(t_stat), 10 - 1) * 2
        fmted_pval = f'{p_val:.3f}' if p_val > 0.001 else f'{p_val:.1e}'
        fmted_tstat = f'{t_stat:.3f}' if t_stat < 100 else f'{t_stat:.1e}'
        bf = np.mean([np.log10(x) for x in subset['bayesfactor'].dropna()])
        values = sub_balanced_score, np.mean(sub_score), np.mean(sub_null), f'{fmted_pval}{notice}', fmted_tstat, bf,
        formatted = sep.join(formats).format(*values)

    return dumb_md_formatter(*formatted.split(sep)), broken


def subpop_hists(data, populations: dict):
    n_populations = len(populations)
    resp_fig, axes = plt.subplots(1, n_populations, figsize=(10, 3))
    # Colors and titles are not managable from
    subpop_colors = [colors.blue, cohorts.get('Hiroshima').color, cohorts.get('AFFDIS').color, colors.orange, ] + [
        None] * n_populations
    for ax, population, p_col in zip(axes, populations, subpop_colors):
        pop_idx = populations[population]
        ax.hist(data.loc[pop_idx]['Response_percentage'], bins=np.linspace(-1, 1, 20), label=population, color=p_col)
        ax.set_title(f'{population} (n={len(pop_idx)})')
        ax.set_xlim(-1, 1)
        ax.set_xticklabels([f'{t:.0%}' for t in ax.get_xticks()])
        ax.set_ylim(0, 40)
    axes[0].set_ylabel('Count')
    plt.suptitle(f'Histogram of "Response_percentage" for {n_populations} Subpopulations')
    plt.tight_layout()
    return resp_fig


class TableDistiller:
    """

    :param alpha:       Significance threshold
    :param results_table:
    :param args:
    :param spacing:
    :param do_bayesian:
    :param is_strict:   Bool if comparissons should stay strict to the global default.
                        For example, when the global default population is set to 'Hiroshima",
                        all comparissons will be performed on this population. But when asked
                        for a comparisson across populations, what should be returned?
                         - strict:      Retain the definition provided by global_default, thus only look at
                                        the Hiroshima cohort. Do not perform inter-population comparisson.
                         - not strict:  Incorporate all other populations for this specific analysis.
    :return:
    """

    def __init__(self, results_table, *args, spacing=20, n_splits=10, is_strict=False, verbose=False):
        self.alpha = 0.05
        self.col_dict = {}
        for j in range(len(results_table.index[0])):
            self.col_dict[j] = set([i[j] for i in results_table.index])
            if verbose:
                print(f'({j + 1}) {str(self.col_dict[j]).replace("{", "").replace("}", "")}')
        self.global_default = [slice(None) if arg is None else arg for arg in args]
        self.n_splits = n_splits
        self.is_strict = is_strict
        self.spacing = spacing
        self.results_table = results_table

    def __call__(self, *col_nums, print_head=True):
        if not col_nums:
            for col_num in range(len(self.global_default)):
                self(col_num + 1, print_head=not bool(col_num))
            return
        elif len(col_nums) == 1:
            col_num = col_nums[0] - 1
        else:
            for i, col_num in enumerate(col_nums):
                self(col_num, print_head=not bool(i))
            return
        if col_num < 0 or col_num > len(self.global_default):
            raise IndexError
        """

        :param col_num:     The column number as integer.
        :param print_head:  Bool,Should a header to the table be printed?
        :return:
        """
        if print_head:
            print(''.ljust(self.spacing + 1), dumb_md_formatter('bacc', 'acc', 'null', 'p_val', 't_stat', 'L10BF'))
        default = deepcopy(self.global_default)

        if default[col_num] != slice(None) and self.is_strict:
            col_vals = [default[col_num]]
        else:
            col_vals = self.col_dict[col_num]
        subsets = []
        row_strs = []

        good_col_vals = deepcopy(col_vals)
        for col_val in col_vals:
            default[col_num] = col_val
            try:
                subset = self.results_table.loc[tuple(default)].dropna()
            except KeyError:
                good_col_vals.remove(col_val)
                subset = np.ndarray(0)  # is this possible?
            if any(subset):
                if len(subset.shape) < 2:
                    # In case only one config was found, .loc has returned a Series instead of DataFrame
                    subset = subset.to_frame().T

                subsets.append(subset)
                row_item, broken = subset_score(subset, self.alpha)
                if broken:
                    good_col_vals.remove(col_val)
                row_desc = col_val if len(col_val) < self.spacing else col_val[:self.spacing - 3] + '...'
                row_strs.append(f'-{row_desc.ljust(self.spacing)}{row_item}\n')

        # Examples of when NA: train & test on Extremes, or subcortical data with deep learning
        subsets = [s for s in subsets if not s['score'].isna().all()]

        # Perform statistics
        if len(subsets) <= 1:
            # print(f'{col_num + 1}. Single option {", ".join(good_col_vals)}. \n', *row_strs)
            print(f'{col_num + 1}.' + row_strs[0][1:], *row_strs[1:])
            return
        elif len(subsets) > 2:
            # We can do this with Scipy.stats, but it does not return as many outcomes
            # stats = f_oneway(subsets)
            subset_scores = [ss['score'] for ss in subsets]
            # So we do it with Pingouin. Shout out for the shittiest data formatting requirement!
            # This is earlly complex because values can be None (and you cannot iterate over them
            # and there are also different cross-validation schemes, for which the fold_n needs
            # to be ajusted!
            vals = []
            result_length_stack = []
            cats = []
            for i, ss in enumerate(subset_scores):
                for cf in ss.keys():
                    if cf in ss:
                        if ss[cf] is not None:
                            val = ss[cf]
                            if not isinstance(val, list):
                                val = val.tolist()
                            n_vals = len(val)
                            vals.extend(val)
                            result_length_stack.append(n_vals)
                            cats.append([i] * n_vals)
            from IPython.core.debugger import Pdb
            fold_ns = flatten(
                [list(range(self.n_splits)) * (rl // self.n_splits) + list(range(rl % self.n_splits)) for rl in
                 result_length_stack])
            cats = flatten(cats)
            pingouin_df = pd.DataFrame(np.stack([cats, vals, fold_ns]).T, columns=['category', 'value', 'fold'])
            # stats = anova(pingouin_df, dv='value', between='category')

            stacked_uniques = []
            stacked_categories = []  ## adjust for repeats by aggregating them
            stacked_folds = []
            for cat in pingouin_df['category'].unique():
                for fol in pingouin_df['fold'].unique():
                    duplicates = pingouin_df[(pingouin_df['category'] == cat) & (pingouin_df['fold'] == fol)]
                    stacked_uniques.append(np.nanmean(duplicates['value']))
                    stacked_categories.append(cat)
                    stacked_folds.append(fol)
                    # if fol:
                    #    break
            pingouin_df = pd.DataFrame(zip(stacked_categories, stacked_uniques, stacked_folds),
                                       columns=['category', 'value', 'fold'])
            anova_stats = rm_anova(pingouin_df, dv='value', within='category', subject='fold')
            # As far as I have tested, pingouin.anova()['p-unc'] = f_oneway()['p-val']
            p_val = anova_stats['p-unc'].values[0]
            eta = anova_stats['F'].values[0]
            word = 'among'
            summary_str = f'F={eta:.4f}'
        else:
            a, b = subsets
            # We can compute effect size using Bayesian Estimation Supersedes the t-test
            p_values = []
            t_stats = []
            b_factors = []
            effsizes = []
            # For every "configuration"
            for r, subset_idx in enumerate(a.index):
                if subset_idx not in b.index:
                    # Skip configs not available for both (should not be possible)
                    continue

                model_a_score = a.loc[subset_idx]['score']
                model_b_score = b.loc[subset_idx]['score']

                model_a_score = np.divide(model_a_score, a.loc[subset_idx]['null']) / 2
                model_b_score = np.divide(model_b_score, b.loc[subset_idx]['null']) / 2

                if isinstance(model_a_score, Iterable) and isinstance(model_b_score, Iterable):
                    population_n = a.iloc[r]['population']
                    n_spl = len(model_a_score) if 'Site' in a.index[r] else self.n_splits
                    try:
                        t_stat, p_value = corr_rep_kfold_cv_test(model_a_score, model_b_score, n_spl, population_n)
                    except ValueError:  # not related t-test e.g. K-fold vs Site
                        ttest_stats = ttest(model_a_score, model_b_score, paired=False)
                        t_stat, p_value = ttest_stats['T'][0], ttest_stats['p-val'][0]
                    bf = bayesfactor_ttest(t_stat, nx=population_n, paired=True)
                    d2 = compute_effsize(model_a_score, model_b_score, paired=True)
                    b_factors.append(bf if bf < 100 else 100)
                    p_values.append(p_value)
                    t_stats.append(t_stat)
                    effsizes.append(d2)
            if any(p_values):
                _, p_val = stats.combine_pvalues(p_values)
                p_val = stats.t.sf(np.abs(np.mean(t_stats)), 10 - 1) * 2
                bf = np.nanmean(np.log10(b_factors))
                d1 = np.mean(effsizes)
            else:
                print(f'{col_num + 1}. Failed ({", ".join(col_vals)})\n')
                return

            # Scaled Jeffrey-Zellner-Siow (JZS) Bayes Factor (BF10)

            word = 'between'
            summary_str = f't_stat{np.mean(t_stats):.3f} BF={bf:.2e}, d={d1:.2e}'
        is_signif = 'No S' if p_val > 0.05 else 'S'
        fmted_pval = f'{p_val:.3f}' if p_val > 0.001 else f'{p_val:.1e}'
        print(f'{col_num + 1}.{is_signif}ignificant difference (p_val: {fmted_pval}, {summary_str}) {word}:\n',
              *row_strs)


def make_empty_nest(*args, bottom=None, **kwargs):
    """
    Creates a None filled nested dictionary
    :param fill_value: the value that should be used to fill the end of the hierarchy branches
    :param args: Every arg should be an iterable
    :param kwargs: Are treated exactly like args, and appended
    keywords are discarded, but they can be used to make code more readable.
    :return: Preallocated nested dict with depth len(args) and None values at the bottom.
    """
    if any(args) and any(kwargs):
        raise ValueError('Using both positional and keyword arguments is not supported to retain a clear hierarcy.')
    elif any(kwargs):
        args = tuple(kwargs.values())

    if any(args):
        return {arg: make_empty_nest(*args[1:], bottom=bottom) for arg in args[0]}
    else:
        return bottom


class OrderedSet(MutableSet):

    def __init__(self, iterable=None):
        self.end = end = []
        end += [None, end, end]  # sentinel node for doubly linked list
        self.map = {}  # key --> [key, prev, next]
        if iterable is not None:
            self |= iterable

    def __len__(self):
        return len(self.map)

    def __contains__(self, key):
        return key in self.map

    def add(self, key):
        if key not in self.map:
            end = self.end
            curr = end[1]
            curr[2] = end[1] = self.map[key] = [key, curr, end]

    def discard(self, key):
        if key in self.map:
            key, prev, next = self.map.pop(key)
            prev[2] = next
            next[1] = prev

    def __iter__(self):
        end = self.end
        curr = end[2]
        while curr is not end:
            yield curr[0]
            curr = curr[2]

    def __reversed__(self):
        end = self.end
        curr = end[1]
        while curr is not end:
            yield curr[0]
            curr = curr[1]

    def pop(self, last=True):
        if not self:
            raise KeyError('set is empty')
        key = self.end[1][0] if last else self.end[2][0]
        self.discard(key)
        return key

    def __repr__(self):
        if not self:
            return '%s()' % (self.__class__.__name__,)
        return '%s(%r)' % (self.__class__.__name__, list(self))

    def __eq__(self, other):
        if isinstance(other, OrderedSet):
            return len(self) == len(other) and list(self) == list(other)
        return set(self) == set(other)


dkt_atlas_lut = {'unknown': 1,
                 'bankssts': 2,
                 'caudalanteriorcingulate': 3,
                 'caudalmiddlefrontal': 4,
                 'corpuscallosum': 5,
                 'cuneus': 6,
                 'entorhinal': 7,
                 'fusiform': 8,
                 'inferiorparietal': 9,
                 'inferiortemporal': 10,
                 'isthmuscingulate': 11,
                 'lateraloccipital': 12,
                 'lateralorbitofrontal': 13,
                 'lingual': 14,
                 'medialorbitofrontal': 15,
                 'middletemporal': 16,
                 'parahippocampal': 17,
                 'paracentral': 18,
                 'parsopercularis': 19,
                 'parsorbitalis': 20,
                 'parstriangularis': 21,
                 'pericalcarine': 22,
                 'postcentral': 23,
                 'posteriorcingulate': 24,
                 'precentral': 25,
                 'precuneus': 26,
                 'rostralanteriorcingulate': 27,
                 'rostralmiddlefrontal': 28,
                 'superiorfrontal': 29,
                 'superiorparietal': 30,
                 'superiortemporal': 31,
                 'supramarginal': 32,
                 'frontalpole': 33,
                 'temporalpole': 34,
                 'transversetemporal': 35,
                 'insula': 36, }


def make_coef_lut(coef_means: pd.Series, hm='L', property='thick', sd=None) -> dict:
    """
    Give me a dict with as keys label names (insula) and value coefficient value
    :param coef_per_label: pandas data series that contains a mapping from anatomical ROI (as Index) to coefficient
    (as float value)
    :param hm: string that expresses the hemosphere to look at (either L (left), R (right) or M (mean)
    :param metric: string that denotes the coefficient to look at (either thick for thickavg or surf for surfavg)
    :return: dictionary that contains a mapping from ROI label as integer to the coefficient
    """

    idx2coef = {k: v for k, v in coef_means.items() if k[0] == hm and property in k}
    idx2coef = {k: v for k, v in idx2coef.items() if '_' in k}
    idx2coef = {k.split('_')[1]: v for k, v in idx2coef.items()}
    idx2coef = {k: v for k, v in idx2coef.items() if k in dkt_atlas_lut}
    if sd is None:
        scaler = 2 * max([abs(v) for v in idx2coef.values()])  # clamp between -1 and 1
    else:
        scaler = 2 * sd
    coef_lut = {dkt_atlas_lut[k]: clamp(v / scaler + 0.5, 0, 1) for k, v in idx2coef.items()}  # center around 0.5
    return coef_lut


class ProgressBar:
    """
    Just an object to make handling of the existing progressbar functionality more minimal. Use:
    pb = ProgressBar(10)  # or ProgressBar(1, 10)
    pb()  # counts one step
    """

    def __init__(self, *args, desc=''):
        if len(args) == 1:
            start, stop = 0, args[0]
        elif len(args) == 2:
            start, stop = args
        self.progress_bar = IntProgress(
            value=0,
            min=start,
            max=stop,
            description='Progress:',
            bar_style='info',
            style={'description_width': 'initial'},
            layout=Layout(width='50%')
        )
        self.text_label = HTML(value='')
        self.container = VBox([self.progress_bar, self.text_label])
        self.current_value = 0
        self.stop = stop
        self.desc = desc
        display(self.container)

    def __call__(self, *args, **kwargs):
        self.count(1)

    def count(self, increment=1):
        self.current_value += increment
        self.progress_bar.value = self.current_value
        self.update_label()

    def update_label(self):
        self.text_label.value = f'{self.current_value}/{self.stop} {self.desc}'
