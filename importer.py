# Base imports
import os
from collections.abc import Iterable

# Logging and progress
from datetime import datetime
from copy import deepcopy
from itertools import tee

# Custom classes and methods
from classes import cohorts, colors
from utils import get_root, load_and_join_dfs, CombatWrapper, PipeWrapper, TorchTrainer, flatten, Squeeze,\
  pickle_out, pickle_in, stacked_hist, clean_data_df, make_empty_nest, get_rgb_cbar,\
  nested_dict_to_df, TableDistiller, make_coef_lut, dkt_atlas_lut, collect_stats_per_site, subpop_hists,\
  torch_val_score, safe_dict_get, load_proj_df, corr_rep_kfold_cv_test, count_values, ProgressBar,\
  Timer, RegressorWrapper

# Data and statistics
import numpy as np
import pandas as pd
from pingouin import bayesfactor_ttest, compute_effsize

# Statistics
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)
np.seterr(divide='ignore', invalid='ignore')  # We store None values which will raise errors in statistics and return nans, which we then omit, so the warnings are not interesting

# Machine learning
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import LeaveOneGroupOut, RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from imblearn.over_sampling import RandomOverSampler

# Deep learning
from tqdm import tqdm
from torchvision.models import resnet18
from torchvision.models import ResNet18_Weights
from torch import nn

# Visualization
from matplotlib import font_manager
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import image as mpimg
import matplotlib.ticker as mtick
import seaborn as sns
font_path = get_root('fonts', 'GuardianSansRegular.otf')  # Your font path goes here
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = prop.get_name()
sns.set()