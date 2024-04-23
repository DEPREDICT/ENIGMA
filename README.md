```python
from importer import *
sN = slice(None)
now = datetime.now()
# results_uid       = now.strftime("%Y%m%d-%H%M%S")
results_uid       = '20240418-181259-postRebuttal'
results_dir       = get_root(f'results', results_uid)
figdir            = results_dir.append('figures')
results_file      = results_dir.append('result_dict.pkl')
restats_file      = results_dir.append('result_and_stats_dict.pkl')
dl_aggregate_file = results_dir.append('aggregate_dict.pkl')  # Stores models
os.makedirs(figdir, exist_ok=True)
print('Results stored at:', results_dir)
```

    Results stored at: D:\repositories\ENIGMA\results\20240418-181259-postRebuttal
​    

*Code supplement to:*

# Predicting Antidepressant Treatment Response from Cortical Structure on MRI: A Mega-Analysis from the ENIGMA-MDD Working Group

![png](results/20240418-181259-postRebuttal/figures/lh_annot.png)

This README file is an export of `main.ipynb`, the primary report on the results presented in the article with this name. It is best to read this documentation in the original file in a notebook server like Jupyter. In this notebook export we will traverse through the data in the same order that things are discussed in the manuscript:
1. Load, join and clean data
2. Data inspection and exploration
3. Preparation of Machine Learning Set Up
4. Train Machine Learning Models
5. Calculate and Print Statistics
6. Interpretation of Results

## 1. Load, join and clean data
Each partner in the ENIGMA consortium provided three types of data:
* `dbc` Demographic, Behavioural and Clinical patient data
* `roi` Cortical measurements on thickness and surface area for each anatomical Region Of Interest
* `map` A 2D projection of the surface data underlying to the `roi` data type.
  Let us start with loading and cleaning this data.

### 1.1 Load, join and clean data
#### 1.1.1 Load Structured data
We first load the data and join sheets using the `load_and_join_dfs` method, which is not fancy only performs an inner join on the two number of data frames.


```python
path_roi = get_root('data', 'ENIGMA-MDD_patients-included_data-subcortical.csv')
path_dbc = get_root('data', 'ENIGMA-MDD_patients-included_data-clinical_wk2.csv')
data = load_and_join_dfs(path_dbc, path_roi, index_col='SubjID', decimal=',')
```


```python
path_dbc_wk1 = get_root('data', 'ENIGMA-MDD_patients-included_data-clinical_wk1.csv')
wk1_subjs = read_any_csv(path_dbc_wk1, index_col='SubjID').index
print(
    f'Number of patients excluded if scan threshold is week 1 instead of week 2: {len([i for i in data.index if i not in wk1_subjs])}')
```

    Number of patients excluded if scan threshold is week 1 instead of week 2: 40
​    

#### 1.1.2 Load 2D-projections of cortical thickness and area
After loading the structured `dbc` and `roi` data, we now load the `map` 2D-projection data.
Briefly on the `map` data: it represents the thickness of an individual on the surface of a hemisphere of the brain. The brain hemisphere is represented as a 3D mesh, you can imagine this as a 3D point cloud, the points are called vertex, plural vertices. Thus, the original data was just a long list (164k) of thickness values (scalars) for each vertex in the mesh. By inflating the hemisphere we can find the position of each scalar on the surface of a sphere. The scalar values were then projected to a plane using stereographic projection using `s12_sphere2place.py`. This code makes sure that left an right hemispheres line up very well (Fig 2.). This scalar data was supplied to us as GIfTY file type. GIfTY (`.gii`) is the geometric equivalent to NIfTY (`.nii`) files, `s12_sphere2place.py` converts this into 2D NumPy arrays, which is the only thick we will be working with in this Notebook. We Clean the data right away by: removing borders, converting it to a 1D vector and dropping zero variance pixels.


```python
# Remove edges and extra channel, and make it into a nice DataFrame with x,y pixel coordinates as column labels
dim = 32
X_thc, X_thc_arr = load_proj_df(data, 'Thickness', dim=dim)
X_are, X_are_arr = load_proj_df(data, 'Area', dim=dim)
```

    100%|██████████| 252/252 [00:00<00:00, 1145.44it/s]
    100%|██████████| 252/252 [00:00<00:00, 2598.01it/s]


### 1.2 Sanitizing and feature engineering
#### 1.2.1 Removing zero variance pixels from 2D projection data


```python
# Find zero variance (corpus callosum (CC))
bool_var = [var < 0.005 for _, var in X_thc.var().iteritems()]
zero_var = [col for col, var in X_thc.var().iteritems() if var < 0.002]

# Drop CC
X_thc = X_thc.drop(columns=zero_var)
X_are = X_are.drop(columns=zero_var)

# Visualization of loaded data
fig, axes = plt.subplots(1, 3)
img_arrays = X_thc_arr.mean(axis=0)[0], X_are_arr.mean(axis=0)[0], np.array(bool_var).reshape(dim - 2, dim - 2)
titles = 'Mean Thickness', 'Mean Area', 'Zero Variance (white)'
cmaps = 'jet', 'jet', 'gray'
for ax, arr, title, cmap in zip(axes, img_arrays, titles, cmaps):
    ax.imshow(arr, cmap=cmap)
    ax.set(title=title, xticks=[], yticks=[])
```


​    
![png](.readme/output_7_0.png)
​    


#### 1.2.2 Normalized Symptom Severity
Secondly we clean the joined data frame using  the `clean_data_df` method. Although this method is mainly concerned with the sanitation of data, (e.g. renaming AO to Age_of_Onset) it also introduces `Normalized_Pretreatment_Severity`, a single numeric pretreatment symptom score. Briefly, we center all scoring methods around zero, and rescale them to their standard deviation. The normalized score we use is preferably MADRS, then HDRS, then BDI, depending on what is available. Since some subjects have more than one scoring we can check if different scoring lists are actually correlated. Correlations are fine (Fig.1)


```python
scoring_methods = 'MADRS', 'HDRS', 'BDI'
data, norm_fig = clean_data_df(data, drop_old_symptom_scores=True)
norm_fig.savefig(figdir.append('NormalizedSeverityScores.png'))
```


​    
![png](.readme/output_9_0.png)
​    


Three instruments to score treatment were used across all cohorts. The following scoring methods were used per site. The method used in order of preference is MADRS, HDRS then BDI.


```python
score_df = pd.DataFrame(columns=scoring_methods)
for cohort in cohorts:
    cohort_data = pd.DataFrame(data=data[data.cohort == cohort.key], columns=data.columns)
    if len(cohort_data):
        contains = lambda scoring_method: cohort_data[f'{scoring_method}_pre'].notna().sum()
        score_df.loc[f'{cohort.name} (N={len(cohort_data)})'] = [
            contains(score) if contains(score) else '' for score in scoring_methods]
print(score_df)
```

                            MADRS HDRS BDI
    AFFDIS (N=16)              16   16  16
    DEP-ARREST-CLIN (N=57)          57    
    Hiroshima cohort (N=92)         92  92
    Melbourne (N=49)           49         
    Minnesota (N=13)                    13
    Milano OSR (N=35)               34  26


#### 1.2.3 Treatment Type
Convert cells with treatment type information into column categories.


```python
# Split up the treatment information into separate boolean columns
drug_types = set(flatten(data['Treatment_type_category'].values))
for drug_type in drug_types:
    data[f'uses_{drug_type}'] = [drug_type in i for i in data['Treatment_type_category']]
```

#### 1.2.4 Extreme non-responders/responders
Let us also insert an `is_extreme label`, we will talk about the "Extremes subpopulation" later


```python
q1, q3 = np.percentile(data['Response_percentage'], [25, 75])
data.insert(loc=0, column='is_extreme',
            value=[(data['Response_percentage'] < q1) | (data['Response_percentage'] > q3)][0])
same_responders = 'AFFDIS', 'Hiroshima', 'Melb'
data.insert(loc=0, column='is_same_responder', value=[c in same_responders for c in data.cohort])
```

### 1.3 Defining Population Subgroups
Now that we know that there are substantial differences among sites in our full data set (`total`) we would like to define more homogeneous subgroups for future analyses. Due to the differences in both measured training data as primary outcomes we would like to initially define two subgroups. The first consists of just a single site, namely the Hiroshima cohort (`Hiroshima`), since it has the largest sample size and its population characteristics are reflective of the larger study. Second, we define a subpopulation as showing a similar treatment response rate under 50% (`Same_resp`). The three cohorts we chose we AFFDIS, Hiroshima cohort and Melbourne. Lastly, we define a third subgroup. These are subjects in the upper and lower qualtile in terms of treatment response. The hypothesis behind this subpopulation is that if there is a treatment effect in the data, one would expect this to be larger in patients showing more response.

Let us define subsets of our full data frame. Let us define the indices for each of our population subgroups that we can use to filter our data frame later:


```python

long_treated = 'MOODS', 'Melb', 'Minnesota'
q1, q3 = np.percentile(data['Response_percentage'], [25, 75])
populations = {
    'All': data.index,
    'Hiroshima': data[data.cohort == 'Hiroshima'].index,
    'SameResponders': pd.Index([i for i in data.index if i.split('_')[0] in same_responders]),
    'Extremes': data[data.is_extreme].index,
    'LongTreated': pd.Index([i for i in data.index if i.split('_')[0] in long_treated]),
}

# Plot histograms of response percentage for each subpopulation
resp_fig = subpop_hists(data, populations)
resp_fig.savefig(figdir.append('ResponseHistogram.png'))
plt.show()
```


​    
![png](.readme/output_17_0.png)
​    


Later on in this study, we present interesting findings in the Extremes subpopulation. An alternative explanation was that treatment duration could positively affect treatment outcome (longer follow up -> more chance to recover), thus that prediction in the Extremes cohort would primarily separate cohorts, and coincidentally, response and non-response. Our six cohorts do show an interesting difference among follow-up durations (see table below or table 1 in the manuscript). We roughly split them up in a group of Short treated ( < 6 weeks ) and `LongTreated` ( > 10 weeks ).

The hypothesis that the results in the Extremes population stems from a selection bias is weakened by the fact that the Extremes subpopulation consists of substantial (>37%) parts of each cohort. Additionally, the histogram of the Response_percentage in the LongTreated population does not have a clear trend to the right that supports the hypothesis that a longer follow-up leads to higher response rates. For details on this population see the table below.

However, interestingly, the `LongTreated` histogram shape looks like a blend between the `All` population and `Extremes` subpopulation, balanced accuracy in the `LongTreated` subpopulation as bad (49.7%) as in the `All` population (47.7%), especially when compared to the `Extremes` subpopulation (68.6%).
Adding the `LongTreated` subgroup will not affect our later analyses because we will tell the `distiller` object to look at the `All` population by default.


```python
duration_based_cohort_printer = lambda in_only: '\n  '.join([''.join([
    c.name.ljust(20),
    f'n={data.cohort.value_counts()[c.key]}'.ljust(7),
    f'{data[data.cohort == c.key].is_responder.mean():.1%}  ',
    f'{str(c.treatment_duration_mu).rjust(4)}+{c.treatment_duration_sd}'
]) for c in cohorts if c.key in set(data.cohort) and (c.key in long_treated if in_only else c.key not in long_treated)])

print('Long treated:\n ', duration_based_cohort_printer(True), '\nShort treated:\n ',
      duration_based_cohort_printer(False))
```

    Long treated:
      DEP-ARREST-CLIN     n=57   84.2%  12.0+0.0
      Melbourne           n=49   44.9%  12.0+0.0
      Minnesota           n=13   61.5%   9.9+2.0 
    Short treated:
      AFFDIS              n=16   37.5%   5.1+0.7
      Hiroshima cohort    n=92   48.9%   6.0+0.0
      Milano OSR          n=35   57.1%   4.2+0.7



```python
expl = {
    'All': '# samples that this cohort contributed to the entire population and the fraction of the entire population that comes from this sample.',
    'All_Responders': '# Responders that this cohort contributed to the entire population.',
    'F_of_All_Responders': "% all responders that comes from this cohort",
    'Extremes': '# samples that this cohort contributed to the Extremes subpopulation and the fraction of the extremes population that comes from this sample.',
    'Extreme_Responders': '# Responders that this cohort contributed to the Extremes subpopulation.',
    'F_of_Extreme_Responders': "% Extreme responders that comes from this cohort",
}
extremes_contribution_df = pd.DataFrame(columns=list(expl.keys()))
extremes_contribution_df.loc['Total'] = [
    f'{len(data)} (100%)',
    f'{data.is_responder.sum()} ({data.is_responder.mean():.1%})',
    '100%',
    f'{len(data[data.is_extreme])} ({len(data[data.is_extreme]) / len(data):.1%})',
    f'{data[data.is_extreme].is_responder.sum()} ({data[data.is_extreme].is_responder.mean():.1%})',
    '100%',
]

for cohort in [c for c in cohorts if c.key in set(data.cohort)]:
    cohort_data = data[data.cohort == cohort.key]
    extreme_cohort_data = cohort_data[cohort_data.is_extreme]
    values = [
        f'{len(cohort_data)}' + f'({len(cohort_data) / len(data):.0%})'.rjust(7),
        bool_fmt(cohort_data.is_responder),
        f'{cohort_data.is_responder.sum() / data.is_responder.sum():.1%}',
        f'{len(extreme_cohort_data)}' + f'({len(extreme_cohort_data) / data.is_extreme.sum():.0%})'.rjust(6),
        bool_fmt(extreme_cohort_data.is_responder),
        f'{cohort_data.is_responder.sum() / data.is_responder.sum():.1%}',
    ]
    extremes_contribution_df.loc[cohort.name] = values
print(extremes_contribution_df)
print('\n'.join(['', 'Explained:'] + [f'{k}:'.ljust(np.max([len(i) for i in expl]) + 4) + v for k, v in expl.items()]))
```

                             All All_Responders F_of_All_Responders     Extremes Extreme_Responders F_of_Extreme_Responders
    Total             262 (100%)    149 (56.9%)                100%  132 (50.4%)         66 (50.0%)                    100%
    AFFDIS             16   (6%)      6 (37.5%)                4.0%      8  (6%)          2 (25.0%)                    4.0%
    DEP-ARREST-CLIN    57  (22%)     48 (84.2%)               32.2%     36 (27%)         33 (91.7%)                   32.2%
    Hiroshima cohort   92  (35%)     45 (48.9%)               30.2%     33 (25%)          8 (24.2%)                   30.2%
    Melbourne          49  (19%)     22 (44.9%)               14.8%     26 (20%)          8 (30.8%)                   14.8%
    Minnesota          13   (5%)      8 (61.5%)                5.4%     11  (8%)          6 (54.5%)                    5.4%
    Milano OSR         35  (13%)     20 (57.1%)               13.4%     18 (14%)          9 (50.0%)                   13.4%
    
    Explained:
    All:                       # samples that this cohort contributed to the entire population and the fraction of the entire population that comes from this sample.
    All_Responders:            # Responders that this cohort contributed to the entire population.
    F_of_All_Responders:       % all responders that comes from this cohort
    Extremes:                  # samples that this cohort contributed to the Extremes subpopulation and the fraction of the extremes population that comes from this sample.
    Extreme_Responders:        # Responders that this cohort contributed to the Extremes subpopulation.
    F_of_Extreme_Responders:   % Extreme responders that comes from this cohort


## 2. Population and data exploration
The ENIGMA consortium provided is with a large amount of subjects from a number of consortia. It is crucial to have a good grasp of the properties of our population as well as differences among sites for the design of our methods and the interpretation of our results

### 2.1 Inspect Patient selection
We start off with a table that shows exclusion of patients. In the code below we go through two exclusion steps: First to select patients with follow-up data available for our classification target, second with complete data available for training. Thus, we go through three stages which we call: `stg1`, `stg2` and `stg3`, of which the last one is equal to the cleaned data set `data` above.


```python
# Load subjects in Stage 1
path_stg1 = get_root('data', 'ENIGMA-MDD_patients-all_data-clinical.csv')
df_stg1 = read_any_csv(path_stg1, index_col='SubjID').iloc[:, 1:]
# Load subjects in Stage 2
df_stg2 = read_any_csv(path_dbc, index_col='SubjID')
# We already have Stage 3
df_stg3 = data
# For Stage 4 we need to edit our projections data frame a bit
df_stg4 = X_thc.loc[[i for i in X_thc.index if 'LH' in i]]
df_stg4.index = df_stg4.index.map(lambda tag: tag.split('|')[0])

# Get the sites, and the data that are in each of the three steps
dfs = df_stg1, df_stg2, data, df_stg4,
sts = [sorted(set([i.split('_')[0] for i in df.index])) for df in dfs]

# Print a verbose overview of the subject selection process
print(f'Prior to subject selection (stage 1) we have {len(dfs[0])} subjects from {len(sts[0])} cohorts.')
print(f'After subject selection (stage 2) we have {len(dfs[1])} subjects from {len(sts[1])} cohorts.')
print(
    f'After checking for completeness of data for ROI analysis (stage 3) we still have {len(dfs[2])} subjects from {len(sts[2])} cohorts.')
print(f'For Vec and 2D analysis, we have {len(dfs[3])} subjects from {len(sts[3])} cohorts')

# Print a table header
n_dfs = [
    {'Total': len(df), '--': 0, **{f'{c.name} ({c.key})': len([i for i in df.index if c.key in i]) for c in cohorts}}
    for df in dfs]
n_all, n_cov, n_dat, _ = n_dfs

# We can split this up further
n_series = [pd.Series(data=n_df, name=f'Stage {i}') for i, n_df in enumerate(n_dfs, 1)]
series_list = flatten([[ser, None] for ser in n_series])[:-1]

# Set the even numbered columns
for i, (df_before, df_after) in enumerate(zip(dfs[:-1], dfs[1:])):
    series_list[1 + i * 2] = pd.Series(data={
        'Total': len(df_before) - len(df_after),
        '--': 0,
        **{f'{cohorts[k].name} ({k})': len(
            [i for i in df_before.index if i not in df_after.index and i.split('_')[0] == k]) for k in cohorts.keys()}
    }, dtype=float, name=f'--excl-->')
print(pd.concat(series_list, axis=1).fillna(0).astype(int).replace(0, '').head(12))

# Uncomment this line if you would like to see which subjects do NOT have Cortical Data
# print("Subjects without Raw Cortigal data are:\n  -", "\n  - ".join([i for i in df_stg3.index if i not in df_stg4.index]))
```

    Prior to subject selection (stage 1) we have 795 subjects from 9 cohorts.
    After subject selection (stage 2) we have 262 subjects from 6 cohorts.
    After checking for completeness of data for ROI analysis (stage 3) we still have 262 subjects from 6 cohorts.
    For Vec and 2D analysis, we have 252 subjects from 6 cohorts
                                 Stage 1 --excl--> Stage 2 --excl--> Stage 3 --excl--> Stage 4
    Total                            795       533     262               262        10     252
    --                                                                                        
    AFFDIS (AFFDIS)                   29        13      16                16                16
    Cardiff (CARDIFF)                 40        40                                            
    DEP-ARREST-CLIN (MOODS)           64         7      57                57                57
    Hiroshima cohort (Hiroshima)     150        58      92                92         1      91
    Melbourne (Melb)                 156       107      49                49                49
    Minnesota (Minnesota)             70        57      13                13                13
    Milano OSR (SanRaffaele)         160       125      35                35         9      26
    Stanford TIGER (TIGER)            48        48                                            
    UCSF Adolescent MDD (SF)          78        78                                            



```python
# print(f'{data.loc[df_stg3.index].is_responder.mean():.1%} {data.loc[df_stg4.index].is_responder.mean():.1%}')
1-data.loc[populations['SameResponders']].is_responder.mean()
```




    0.535031847133758



### 2.2 Inspect provided columns
Before we move to the `map` data, let us take a look at the `roi` data that we received.
There are also a lot of non-cortext columns. Do not worry about them, later on we will mention which are included in the model as BDC features explicitly.


```python
# Load structured data
dkt_atlas_features = [i for i in data.columns if any([j in i for j in dkt_atlas_lut])]
subcortical_rois = [i for i in pd.read_csv(path_roi, sep=';', index_col='SubjID').columns if
                    i[1] != '_' and not 'Site' in i][:-2]
subcortical_rois = [i for i in subcortical_rois if 'Thick' not in i and 'Surf' not in i]

print(
    f'The number of cortical features we found is {len(dkt_atlas_features)} ({32 * 2 * 3} = 32 ROIs × 2 maps (Thickness & Sufrace) × 3 hemispheres (Left, Right, Mean). That means there are {len(data.columns) - len(dkt_atlas_features)} non-surface columns.')
subc_rois = subcortical_rois
subc_no_icv = subc_rois[:-1]
print(f'\nWe also have ICV and {len(subc_rois) - 1} subcortical_rois:', ', '.join(subc_rois[:-1]))
unique_subc_rois = sorted(set([i[1:].capitalize() if i[0] in 'LRM' else i for i in subc_rois]))
print(f'Of these {len(unique_subc_rois)} are unique:', ', '.join(unique_subc_rois),
      'Vent, LatVent and ICV are not included, so 7 remain.')
# data[dkt_atlas_features].head()  # You can take a look at the DKT Atlas features by uncommenting this line
data[subc_no_icv] = data[subc_no_icv].astype(float).div(data['ICV'], axis=0)
```

    The number of cortical features we found is 192 (192 = 32 ROIs × 2 maps (Thickness & Sufrace) × 3 hemispheres (Left, Right, Mean). That means there are 82 non-surface columns.

    We also have ICV and 24 subcortical_rois: LLatVent, RLatVent, Lthal, Rthal, Lcaud, Rcaud, Lput, Rput, Lpal, Rpal, Lhippo, Rhippo, Lamyg, Ramyg, Laccumb, Raccumb, Mvent, Mthal, Mcaud, Mput, Mpal, Mhippo, Mamyg, Maccumb
    Of these 10 are unique: Accumb, Amyg, Caud, Hippo, ICV, Latvent, Pal, Put, Thal, Vent Vent, LatVent and ICV are not included, so 7 remain.


### 2.3 General Population Characteristics
Each data set was collected using different inclusion protocols. Thus, our the population per site can vary a lot. Let us start with group level statistics. We first create a data frame containing subject properties per site called `site_stats` using the `collect_stats_per_site` method. This data frame will be the input to the `stacked_hist` method. This method can read a column from the `site_stats` table and create a stacked histogram. In addition to `site_stats` we also receive a `site_means` table, which contains the same information but nicely formatted for printing, so let us start with that:

As you can see, the percentage of participants that is female (`is_female`) is more than twice as high in the Milano OSR cohort than in the AFFDIS cohor. Also, mean age of the participants included (`Age`) varies (15.0 for the Minnesota, and 50.3 for Milano OSR. Lastly, response rate varies from 37% at AFFDIS to 74% for DEP-ARREST-CLIN.


```python
site_stats, site_means = collect_stats_per_site(data)
print(site_means.T)
```

    Cohort                                    Total         AFFDIS DEP-ARREST-CLIN Hiroshima cohort      Melbourne      Minnesota     Milano OSR
    Numel                                       262             16              57               92             49             13             35
    Age                                 36.5 ± 15.3    44.1 ± 14.1     34.1 ± 12.7      43.0 ± 11.6     19.6 ± 3.0     15.0 ± 2.2     51.3 ± 9.6
    Treatment_duration                    8.3 ± 3.2      5.1 ± 0.7      12.0 ± 0.0        6.0 ± 0.0     12.0 ± 0.0      9.9 ± 1.9      4.3 ± 0.7
    Normalized_Pretreatment_Severity    0.04 ± 0.99   -0.85 ± 1.20     0.79 ± 0.78     -0.31 ± 0.79    0.28 ± 0.72    0.15 ± 0.95   -0.25 ± 1.12
    Age_of_Onset                        30.3 ± 14.8    35.6 ± 15.7     28.0 ± 11.7      39.2 ± 13.3     15.7 ± 2.7     11.2 ± 2.6    36.5 ± 11.1
    is_female                           154 (58.8%)      5 (31.2%)      39 (68.4%)       47 (51.1%)     33 (67.3%)      9 (69.2%)     21 (60.0%)
    is_recurrent                        160 (61.1%)     15 (93.8%)      29 (50.9%)       46 (50.0%)     31 (63.3%)      9 (69.2%)     30 (85.7%)
    is_responder                        149 (56.9%)      6 (37.5%)      48 (84.2%)       45 (48.9%)     22 (44.9%)      8 (61.5%)     20 (57.1%)
    uses_SSRI                           194 (74.0%)      9 (56.2%)        0 (0.0%)       91 (98.9%)    49 (100.0%)    13 (100.0%)     32 (91.4%)
    uses_SNRI                            68 (26.0%)      3 (18.8%)     57 (100.0%)         0 (0.0%)       0 (0.0%)       0 (0.0%)      8 (22.9%)
    uses_ATYP                             16 (6.1%)      8 (50.0%)        0 (0.0%)         1 (1.1%)       0 (0.0%)       0 (0.0%)      7 (20.0%)
    is_same_responder                   157 (59.9%)    16 (100.0%)        0 (0.0%)      92 (100.0%)    49 (100.0%)       0 (0.0%)       0 (0.0%)
    is_extreme                          132 (50.4%)      8 (50.0%)      36 (63.2%)       33 (35.9%)     26 (53.1%)     11 (84.6%)     18 (51.4%)
    ADcur                                 1.5 ± 5.0    10.8 ± 14.4       0.0 ± 0.0        0.9 ± 0.6      nan ± nan      nan ± nan      1.0 ± 0.5
    Response_percentage               50.9% ± 34.9%  36.7% ± 35.5%   74.6% ± 23.1%    43.9% ± 30.1%  43.9% ± 30.0%  41.0% ± 60.6%  50.8% ± 37.7%
    is_remitter                         118 (45.0%)      7 (43.8%)      42 (73.7%)       31 (33.7%)     17 (34.7%)      7 (53.8%)     14 (40.0%)



```python
responder_stats, responder_means = collect_stats_per_site(data.loc[data.is_responder])
non_responder_stats, non_responder_means = collect_stats_per_site(data.loc[~data.is_responder])
non_vs_responders = pd.concat([site_means.T.Total, responder_means.T.Total, non_responder_means.T.Total], axis=1)
non_vs_responders.columns = ['Total', 'Responders', 'Non-responders']
print(non_vs_responders)
```

                                              Total     Responders Non-responders
    Numel                                       262            149            113
    Age                                 36.5 ± 15.3    35.6 ± 15.0    37.6 ± 15.6
    Treatment_duration                    8.3 ± 3.2      8.7 ± 3.3      7.8 ± 3.1
    Normalized_Pretreatment_Severity    0.04 ± 0.99    0.18 ± 0.97   -0.15 ± 0.99
    Age_of_Onset                        30.3 ± 14.8    30.2 ± 14.5    30.3 ± 15.0
    is_female                           154 (58.8%)     84 (56.4%)     70 (61.9%)
    is_recurrent                        160 (61.1%)     83 (55.7%)     77 (68.1%)
    is_responder                        149 (56.9%)   149 (100.0%)       0 (0.0%)
    uses_SSRI                           194 (74.0%)     94 (63.1%)    100 (88.5%)
    uses_SNRI                            68 (26.0%)     53 (35.6%)     15 (13.3%)
    uses_ATYP                             16 (6.1%)       9 (6.0%)       7 (6.2%)
    is_same_responder                   157 (59.9%)     73 (49.0%)     84 (74.3%)
    is_extreme                          132 (50.4%)     66 (44.3%)     66 (58.4%)
    ADcur                                 1.5 ± 5.0      0.7 ± 1.2      2.5 ± 7.5
    Response_percentage               50.9% ± 34.9%  76.2% ± 14.5%  17.6% ± 24.2%
    is_remitter                         118 (45.0%)    115 (77.2%)       3 (2.7%)


##### 2.4.1 Can the large differences in response rates among sites be explained?
We observe a large difference among sites in treatment response rate (37%-84%).
Is this obviously related to any other properties, say age, treatment duration or medication?

We find that this is the case, average age for responders is 35.1 and 37.8.
Average treatment duration is 9 weeks for responders, and 7.9 for non-responders.
Finally, we do find a large difference in the antidepressant used: 82% of SSRI-users does not respond, while 79% of SNRI-users does.
However, these properties are strongly correlated with site, 57/66 SNRI users are from the highest performing site with an average response rate of 84%.


```python
covariates = 'Age', 'Treatment_duration', 'is_responder'
variables = ('is_responder',), ('is_responder',), ('uses_SSRI', 'uses_SNRI', 'uses_ATYP')
response_correlates = pd.DataFrame(columns=['Population', 'Property', '+', '-', 'n+', 'n-'])
for covariate, variable in zip(covariates, variables):
    for var in variable:
        group_a = data[data[var]]
        group_b = data[~data[var]]
        num_a = group_a[covariate].mean()
        num_b = group_b[covariate].mean()
        fmt = '{:.1%}' if num_a < 1 else '{:.1f}'
        response_correlates.loc[len(response_correlates)] = var, covariate, fmt.format(num_a), fmt.format(num_b), len(
            group_a), len(group_b)
print(response_correlates)
```

         Population            Property      +      -   n+   n-
    0  is_responder                 Age   35.6   37.6  149  113
    1  is_responder  Treatment_duration    8.7    7.8  149  113
    2     uses_SSRI        is_responder  48.5%  80.9%  194   68
    3     uses_SNRI        is_responder  77.9%  49.5%   68  194
    4     uses_ATYP        is_responder  56.2%  56.9%   16  246


### 2.4 Comparissons among cohorts
#### 2.4.1 Adolescents per cohort


```python
adolescent_df = pd.DataFrame(columns=('N', '%'))
for cohort in [c for c in cohorts if c.name in site_stats.index]:
    adolescents = np.sum(site_stats.loc[cohort.name].Age < 20)
    total = site_stats.loc[cohort.name].Numel
    table_head = f'{cohort.name} (N={total})'
    adolescent_df.loc[table_head] = (adolescents, f'{adolescents / total * 100:.0f}')
print(adolescent_df)
```

                              N    %
    AFFDIS (N=16)             1    6
    DEP-ARREST-CLIN (N=57)    6   11
    Hiroshima cohort (N=92)   0    0
    Melbourne (N=49)         23   47
    Minnesota (N=13)         13  100
    Milano OSR (N=35)         0    0


#### 2.4.1 Symptom severity
Let us see if treatment response varies by site. This will be important for harmonization. The first way to look at it is simply looking at remission rates per site:


```python
outcomes = 'is_female', 'is_responder', 'is_remitter', 'is_extreme'
outcomes_to_plot = outcomes[1], outcomes[3]
n_otp = len(outcomes_to_plot)
pie_fig, axes = plt.subplots(n_otp, len(set(data.cohort)), figsize=(12, 2 * n_otp))
#axes = axes.reshape(-1)
for u, outcome in enumerate(outcomes_to_plot):  # 'is_remitter']):
    v = 0
    for cohort in cohorts:
        df_cohort = data[data.cohort == cohort.key]
        if not any(df_cohort.index):
            continue
        axes[u, v].pie(
            x=[df_cohort[outcome].sum(), len(df_cohort) - df_cohort[outcome].sum()],
            colors=(cohort.color, colors.dark),
            autopct='%.0f%%',
        )
        if not u:
            axes[u, v].set_title(f'{cohort.name}\n(n={len(df_cohort)})')
        if not v:
            axes[u, v].set_ylabel(outcome)
        v += 1
plt.tight_layout()
pie_fig.savefig(figdir.append('ResponseRateBySite.png'))
plt.show()
```


​    
![png](.readme/output_34_0.png)
​    


#### 2.4.2 Symptom development over time
Secondly, we look at the trends by plotting the change per site


```python
trend_fig, axes = plt.subplots(1, 3, figsize=(10, 4))
for ax, score in zip(axes.reshape(-1), scoring_methods):
    for cohort in cohorts:
        # Get the values for a specific cohort and a specific symptom scoring method
        cohort_data = pd.DataFrame(data=data[data.cohort == cohort.key], columns=data.columns)
        pre = cohort_data[score + '_pre'].notna()
        post = cohort_data[score + '_post'].notna()
        subjs = cohort_data[(pre & post)]
        if not len(subjs):
            continue
        vals = subjs[[score + '_pre', score + '_post']].to_numpy().astype(int).T
        ax.plot(vals, color=cohort.color, lw=1)  # Plot the found individual lines
        ax.plot(vals.mean(axis=1), color=colors.dark, lw=5.0, zorder=3)  # And plot the mean in bold
    ax.set(xticks=(), xlim=(0, 1), title=f'{score} (n={data[score + "_pre"].notna().sum()})')
trend_fig.suptitle('Severity Scores from Time Point 1 to 2 for Each Metric')
# Format the legend
line_per_cohort = [Line2D([], [], color=c.color, label=c.name, lw=3) for c in cohorts if c.name in site_stats.index]
axes[1].legend(handles=line_per_cohort, ncol=len(site_stats.index), loc='center', bbox_to_anchor=(0.5, -.1))
trend_fig.savefig(figdir.append('ResponseTrends.png'))
```


​    
![png](.readme/output_36_0.png)
​    


### 2.5 Comparing subpopulations to the total population
#### 2.5.1 No differences in characteristics between entire population and the Extremes subpopulation
Later on in this work, we will find some interesting findings (good classification accuracy) for the "Extremes"  subpopulation, so let us take a look at these characteristics as well.

The underlying hypothesis for the formulation of this subpopulation was that - within MDD - there are different subpopulations with different a phenotype and different treatment response. However, simpler, alternative hypotheses, e.g. effects such as site, age or other clinical characteristics would be preferred. To verify the probability of the these hypotheses, we are interested in differences in population characteristics between the "Extremes" subpopulation and the total population, this is shown below. If anything stands out, this might weaken the belief in our primary hypothesis.

Under `Numel` we find that the Extremes subjects originate from all sites, and actually most sites also contributed about 50% to the Extremes subpopulation. In terms of `Age` or `is_female`, there are also no striking differences. This means that there is no clear evidence of the alternative hypotheses to our primary hypothesis.


```python
hist_fig, axes = plt.subplots(2, 2, figsize=(16, 8))
stacked_hist(data, site_stats, ax=axes[0, 0], make_legend=False, name='Age')
stacked_hist(data, site_stats, ax=axes[0, 1], make_legend=False, name='Age_of_Onset')
stacked_hist(data, site_stats, ax=axes[1, 0], make_legend=False, name='Normalized_Pretreatment_Severity')
stacked_hist(data, site_stats, ax=axes[1, 1], make_legend=True, name='Response_percentage')
axes[1, 1].set_xticklabels([f'{t:.0%}' for t in axes[1, 1].get_xticks()])
hist_fig.savefig(figdir.append('ResponseHistogram.png'))
plt.show()
```


​    
![png](.readme/output_38_0.png)
​    



```python
_, ss_means = collect_stats_per_site(data.loc[populations['Hiroshima']])
print(ss_means.T)
```

    Cohort                                    Total Hiroshima cohort
    Numel                                        92               92
    Age                                 43.0 ± 11.6      43.0 ± 11.6
    Treatment_duration                    6.0 ± 0.0        6.0 ± 0.0
    Normalized_Pretreatment_Severity   -0.31 ± 0.79     -0.31 ± 0.79
    Age_of_Onset                        39.2 ± 13.3      39.2 ± 13.3
    is_female                            47 (51.1%)       47 (51.1%)
    is_recurrent                         46 (50.0%)       46 (50.0%)
    is_responder                         45 (48.9%)       45 (48.9%)
    uses_SSRI                            91 (98.9%)       91 (98.9%)
    uses_SNRI                              0 (0.0%)         0 (0.0%)
    uses_ATYP                              1 (1.1%)         1 (1.1%)
    is_same_responder                   92 (100.0%)      92 (100.0%)
    is_extreme                           33 (35.9%)       33 (35.9%)
    ADcur                                 0.9 ± 0.6        0.9 ± 0.6
    Response_percentage               43.9% ± 30.1%    43.9% ± 30.1%
    is_remitter                          31 (33.7%)       31 (33.7%)



```python
_, rrs_means = collect_stats_per_site(data.loc[populations['SameResponders']])
print(rrs_means.T)
```

    Cohort                                    Total         AFFDIS Hiroshima cohort      Melbourne
    Numel                                       157             16               92             49
    Age                                 35.8 ± 14.9    44.1 ± 14.1      43.0 ± 11.6     19.6 ± 3.0
    Treatment_duration                    7.8 ± 2.9      5.1 ± 0.7        6.0 ± 0.0     12.0 ± 0.0
    Normalized_Pretreatment_Severity   -0.18 ± 0.89   -0.85 ± 1.20     -0.31 ± 0.79    0.28 ± 0.72
    Age_of_Onset                        31.4 ± 15.7    35.6 ± 15.7      39.2 ± 13.3     15.7 ± 2.7
    is_female                            85 (54.1%)      5 (31.2%)       47 (51.1%)     33 (67.3%)
    is_recurrent                         92 (58.6%)     15 (93.8%)       46 (50.0%)     31 (63.3%)
    is_responder                         73 (46.5%)      6 (37.5%)       45 (48.9%)     22 (44.9%)
    uses_SSRI                           149 (94.9%)      9 (56.2%)       91 (98.9%)    49 (100.0%)
    uses_SNRI                              3 (1.9%)      3 (18.8%)         0 (0.0%)       0 (0.0%)
    uses_ATYP                              9 (5.7%)      8 (50.0%)         1 (1.1%)       0 (0.0%)
    is_same_responder                  157 (100.0%)    16 (100.0%)      92 (100.0%)    49 (100.0%)
    is_extreme                           67 (42.7%)      8 (50.0%)       33 (35.9%)     26 (53.1%)
    ADcur                                 2.4 ± 6.6    10.8 ± 14.4        0.9 ± 0.6      nan ± nan
    Response_percentage               43.2% ± 30.7%  36.7% ± 35.5%    43.9% ± 30.1%  43.9% ± 30.0%
    is_remitter                          55 (35.0%)      7 (43.8%)       31 (33.7%)     17 (34.7%)



```python
extr_stats, extr_means = collect_stats_per_site(data.loc[populations['Extremes']])
print(extr_means.T)
```

    Cohort                                    Total         AFFDIS DEP-ARREST-CLIN Hiroshima cohort      Melbourne      Minnesota     Milano OSR
    Numel                                       132              8              36               33             26             11             18
    Age                                 34.3 ± 14.8    43.4 ± 13.1     32.4 ± 11.0      42.8 ± 10.4     19.8 ± 3.1     15.3 ± 2.3    50.9 ± 10.4
    Treatment_duration                    8.9 ± 3.3      5.0 ± 0.7      12.0 ± 0.0        6.0 ± 0.0     12.0 ± 0.0     10.3 ± 1.9      4.3 ± 0.7
    Normalized_Pretreatment_Severity    0.03 ± 0.98   -0.68 ± 1.02     0.72 ± 0.75     -0.54 ± 0.73    0.43 ± 0.76    0.05 ± 0.95   -0.58 ± 0.85
    Age_of_Onset                        27.2 ± 13.8    31.1 ± 14.3     25.8 ± 10.1      39.1 ± 13.7     15.6 ± 3.0     11.7 ± 2.0    34.3 ± 10.3
    is_female                            81 (61.4%)      5 (62.5%)      20 (55.6%)       19 (57.6%)     18 (69.2%)      7 (63.6%)     12 (66.7%)
    is_recurrent                         83 (62.9%)      7 (87.5%)      19 (52.8%)       18 (54.5%)     16 (61.5%)      7 (63.6%)     16 (88.9%)
    is_responder                         66 (50.0%)      2 (25.0%)      33 (91.7%)        8 (24.2%)      8 (30.8%)      6 (54.5%)      9 (50.0%)
    uses_SSRI                            91 (68.9%)      4 (50.0%)        0 (0.0%)      33 (100.0%)    26 (100.0%)    11 (100.0%)     17 (94.4%)
    uses_SNRI                            42 (31.8%)      2 (25.0%)     36 (100.0%)         0 (0.0%)       0 (0.0%)       0 (0.0%)      4 (22.2%)
    uses_ATYP                              5 (3.8%)      3 (37.5%)        0 (0.0%)         0 (0.0%)       0 (0.0%)       0 (0.0%)      2 (11.1%)
    is_same_responder                    67 (50.8%)     8 (100.0%)        0 (0.0%)      33 (100.0%)    26 (100.0%)       0 (0.0%)       0 (0.0%)
    is_extreme                         132 (100.0%)     8 (100.0%)     36 (100.0%)      33 (100.0%)    26 (100.0%)    11 (100.0%)    18 (100.0%)
    ADcur                                 1.5 ± 5.7    12.0 ± 16.0       0.0 ± 0.0        0.8 ± 0.4      nan ± nan      nan ± nan      0.9 ± 0.5
    Response_percentage               46.4% ± 46.3%  22.9% ± 44.1%   83.6% ± 21.6%    25.6% ± 40.2%  32.9% ± 34.4%  37.7% ± 65.3%  45.3% ± 50.2%
    is_remitter                          67 (50.8%)      3 (37.5%)      33 (91.7%)        8 (24.2%)      8 (30.8%)      6 (54.5%)      9 (50.0%)

```python
extremes = data.loc[populations['Extremes']]
extr_lo_stats, extr_lo_means = collect_stats_per_site(extremes[~extremes.is_responder])
extr_hi_stats, extr_hi_means = collect_stats_per_site(extremes[extremes.is_responder])
extr_non_vs_responders = pd.concat([extr_means.T.Total, extr_lo_means.T.Total, extr_hi_means.T.Total], axis=1)
extr_non_vs_responders.columns = ['Extremes Total', 'Extreme Non-responders', 'Extreme Responders']
print(extr_non_vs_responders)
```

                                     Extremes Total Extreme Non-responders Extreme Responders
    Numel                                       132                     66                 66
    Age                                 34.3 ± 14.8            35.9 ± 16.0        32.6 ± 13.3
    Treatment_duration                    8.9 ± 3.3              7.9 ± 3.2          9.8 ± 3.2
    Normalized_Pretreatment_Severity    0.03 ± 0.98           -0.29 ± 0.95        0.35 ± 0.91
    Age_of_Onset                        27.2 ± 13.8            28.3 ± 14.8        26.2 ± 12.7
    is_female                            81 (61.4%)             41 (62.1%)         40 (60.6%)
    is_recurrent                         83 (62.9%)             44 (66.7%)         39 (59.1%)
    is_responder                         66 (50.0%)               0 (0.0%)        66 (100.0%)
    uses_SSRI                            91 (68.9%)             59 (89.4%)         32 (48.5%)
    uses_SNRI                            42 (31.8%)              8 (12.1%)         34 (51.5%)
    uses_ATYP                              5 (3.8%)               2 (3.0%)           3 (4.5%)
    is_same_responder                    67 (50.8%)             49 (74.2%)         18 (27.3%)
    is_extreme                         132 (100.0%)            66 (100.0%)        66 (100.0%)
    ADcur                                 1.5 ± 5.7              2.8 ± 8.2          0.4 ± 0.8
    Response_percentage               46.4% ± 46.3%           2.9% ± 21.3%       89.9% ± 6.8%
    is_remitter                          67 (50.8%)               1 (1.5%)        66 (100.0%)



```python
print('Mean ± 1 SD treatment duration for the following populations:')
relevant_cohorts = [c for c in cohorts if c.name in site_stats.index]
for stats_desc, stats in zip(
        ['All', 'Responders', 'Non-responders', 'Extremes', 'Responding Extremes', 'Non-responding Extremes'],
        [site_stats, responder_stats, non_responder_stats, extr_stats, extr_hi_stats, extr_lo_stats]):
    print(stats_desc.ljust(24),
          f'{np.sum([stats.loc[c.name].Numel * c.treatment_duration_mu for c in relevant_cohorts if c.name in stats.index]) / stats.Numel.sum():.2f} ± '
          f'{np.sum([stats.loc[c.name].Numel * c.treatment_duration_sd for c in relevant_cohorts if c.name in stats.index]) / stats.Numel.sum():.2f}')
```

    Mean ± 1 SD treatment duration for the following populations:
    All                      8.33 ± 0.24
    Responders               8.75 ± 0.23
    Non-responders           7.77 ± 0.24
    Extremes                 8.84 ± 0.30
    Responding Extremes      9.81 ± 0.30
    Non-responding Extremes  7.88 ± 0.31



```python
print('\n'.join([f'Extremes in {c.name} with ADcur == 0:'.ljust(50) + str(
    (data[(data.cohort == c.key) & data.is_extreme].ADcur == 0).sum()) for c in relevant_cohorts]))
```

    Extremes in AFFDIS with ADcur == 0:               1
    Extremes in DEP-ARREST-CLIN with ADcur == 0:      36
    Extremes in Hiroshima cohort with ADcur == 0:     2
    Extremes in Melbourne with ADcur == 0:            0
    Extremes in Minnesota with ADcur == 0:            0
    Extremes in Milano OSR with ADcur == 0:           0


## 3. Preparation of Machine Learning setup
Our primary goal is to predict treatment response. However, we also have several sub questions for which we create five dimensions of comparisons:
1. `data_type` Do we use the `roi` data, or the `map` data?
2. `dbc_presence` Do we add `dbc` data to our chosen data type?
3. `population` Do we train on the entire population (All) or a subpopulation (i.e. Hiroshima, Extremes or Same_responders).
4. `cv_method` Which cross-validation method do we use? K-fold (`Fold`) or leave-site-out (`Site`)
5. `classifier` Which classifier do we use? Logistic regression, Support Vector Classifier (SVC) or a Gradient boosting classifier.
6. `target_label` What the label is that we are trying to predict. Although this is usually `is_responder`, there is the option to classify `is_female` as a sanity check, and also `is_remitter`, and out of curiosity we can also predict `is_extreme`.

Furthermore, we perform LASSO feature selection using an meta transformer on an L1-penalized support vector classifier. Data harmonization is performed using ComBat. The ComBat wrapper stores our `data` such that it can access covariates from withing the sklearn `pipeline`. Features are centered around the mean and scaled to unit variance. We report accuracy, balanced accuracy (bAcc) and model Effect Size and Bayes Factor.

### 3.1 data_type
* `roi` cortical thickness and the surface area for each of the 34 ROIs per hemisphere, resulting in 136 predictors (a. ROI average).
* `vec` the voxel-wise cortical surface area and cortical thickness measurements as a single one-dimensional (1D) vector generated by downsampling using spatial linear interpolation, resulting in 900 predictors (b. cortical vector).
* `2DT` cortical data representations by projecting the cortical surface thickness measurements to two-dimensional (2D) planes of 64×64 pixels using stereographic projection (c. cortical thickness projection).
* `2DA` cortical data representations by projecting the cortical surface area  measurements to two-dimensional (2D) planes of 64×64 pixels using stereographic projection (d. surface area projection).

### 3.2 dbc_presence
Here is a list of what the different `dbc_options` mean:
* `dbc_no` add nothing to cortical data
* `dbc_yes` add DBC data to cortical data (6: `is_ad`, `is_recurrent`,  `is_female`,  `Age`,  `Normalized_Pretreatment_Severity`, `Age_of_Onset`)
* `sub_add` add subcortical data to cortical data (25: `ICV`, `LLatVent`, ...,  `Rput`, `Rthal`)
* `sub_nly` add nothing to subcortical data
* `sub_dbc` add DBC data to subcortical data
* `sub_all` add subcortical and DBC data to cortical data
* `dbc_nly` only use DBC data

### 3.3 population
Here is a list of the populations with explaination:
* `All` All eligible patients from all six cohorts (n=265).
* `Hiroshima` the single largest cohort to ascertain if inter-cohort variance played a role in our prediction outcomes (I. single cohort) (n=103).
* `SameResponders` cohorts with a mean response rate below 50% (II. response rate selected cohorts) (n-168).
* `Extremes` a subpopulation consisting of the extreme subgroups of responders and non-responders, i.e., the 25% of patients showing the lowest percentage changes in depression severity and 25% responding the largest percentage changes to antidepressant treatment (III. extreme (non-)responders) (n=132).
* `LongTreated` T

### 3.4 cv_method
* `Fold`10 times repeated stratified 10-fold cross-validation implemented through `sklearn.model_selection.RepeatedStratifiedKFold`.
* `Site` leave site out cross-validation implemented through `sklearn.model_selection.LeaveOneGroupOut`.

### 3.5 classifier
* `LogisticRegression`
* `SVC`
* `GradientBoostingClassifier`

### 3.6 target_label
* `is_responder` primary outcome.
* `is_remitter` alternative outcome.
* `is_female` used as a sanity check.
* `is_extreme` briefly explored since treatment response prediction in this group works, if you can predict this group beforehand, this might still be useful. But as we will find in analysis 6.3.3.2, we can't predict it.


```python
# Some settings to our machine learning method:
target_labels = 'is_responder', 'is_remitter', 'is_female', 'is_extreme',
n_splits = 10
n_repeats = 10

# 1. Define data type for full population we need :X = train data, y = label data, c = stratification data
get_y_for_target = lambda tl: pd.Series(data=[data[tl].loc[sub.split('|')[0]] for sub in X_thc.index],
                                        index=X_thc.index, name=tl)
dtypes = {'vec': (X_thc,
                  pd.DataFrame.from_dict(
                      data={target_label: get_y_for_target(target_label) for target_label in target_labels}),
                  pd.Series(data={sub: data.loc[sub.split('|')[0]].cohort_idx for sub in X_thc.index},
                            index=X_thc.index, name='Stratification Label'),),
          'roi': (data[dkt_atlas_features],
                  pd.DataFrame.from_dict(data={target_label: data[target_label] for target_label in target_labels}),
                  data.cohort_idx,),
          }

# 2.1 Presence of Demographic, behavioural and clinical (DBC) data:
dbc_cols = ['is_ad', 'is_recurrent', 'is_female', 'Age', 'Normalized_Pretreatment_Severity', 'Age_of_Onset']
dbc_prj = data[dbc_cols].loc[[s.split('|')[0] for s in X_thc.index]]
sub_prj = data[dbc_cols].loc[[s.split('|')[0] for s in X_thc.index]]
dbc_prj.insert(loc=0, value=X_thc.index, column='SubjID|Hem')
sub_prj.insert(loc=0, value=X_thc.index, column='SubjID|Hem')
dbc_prj = dbc_prj.set_index('SubjID|Hem')
dbc_options = {
    'dbc_no': {'roi': pd.DataFrame(), 'vec': pd.DataFrame()},  # add nothing to cortical data
    'dbc_yes': {'roi': data[dbc_cols], 'vec': dbc_prj},  # add DBC data to cortical data
    'sub_add': {'roi': data[subc_rois], 'vec': None},  # add subcortical data to cortical data
    'sub_nly': {'roi': pd.DataFrame(), 'vec': None},  # add nothing to subcortical data
    'sub_dbc': {'roi': data[dbc_cols], 'vec': None},  # add DBC data to subcortical data
    'sub_all': {'roi': data[subc_rois + dbc_cols], 'vec': None},  # add subcortical and DBC data to cortical data
    'dbc_nly': {'roi': pd.DataFrame(), 'vec': None},  # only use DBC
}

# 2.2 Get indices each population, both for our roi data frame (which we already defined at 2.3) but also for projection data
get_indices = lambda key, indices: {'roi': indices, 'vec': [i for i in X_thc.index if i.split('|')[0] in indices]}
population_indices = {k: get_indices(k, i) for k, i in populations.items()}

# 3. Define the data for each of our previously defined populations
data_dict = make_empty_nest(proj_type=dtypes, dbc_presence=dbc_options, population=populations)
for dtype_name, dtype_values in dtypes.items():
    X, y, c = dtype_values
    for dbc_option, dbc_values in dbc_options.items():
        # These conditionals are a bit hacky because we replace the cortical
        # ROIs with something else (subc or dbc data), but it save a ton of code
        if dbc_option in ('sub_nly', 'sub_dbc',):
            X = data[subc_rois]
        elif dbc_option in ('dbc_nly',):
            X = data[dbc_cols]

        if dbc_values[dtype_name] is None:
            del data_dict[dtype_name][dbc_option]
            continue
        X_cort_and_dbc = pd.concat([dbc_values[dtype_name], X], axis=1)
        for pop_key, pop_idx in population_indices.items():
            get_subpop_subjs = lambda sub_df: sub_df.loc[pop_idx[dtype_name]]
            data_dict[dtype_name][dbc_option][pop_key] = [get_subpop_subjs(sub_df) for sub_df in (X_cort_and_dbc, y, c)]

# 4. Define our cross-validation methods:
cv_schemes = {'Fold': (RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=0),
                       CombatWrapper(data=data, discrete_covariates=['is_female'],
                                     continuous_covariates=['Age', 'Age2']),),
              'Site': (LeaveOneGroupOut(), None,)}

# 5. Define classifiers to use:
classifiers = \
    LogisticRegression(max_iter=500), \
    SVC(), \
    GradientBoostingClassifier(), \
    RandomForestClassifier(),

# Define other pipeline components
regressor = RegressorWrapper(data=data, continuous_covariates=['Age', 'Age2'])
imputer = PipeWrapper(KNNImputer)
selector = SelectFromModel(LinearSVC(penalty="l1", dual=False, max_iter=20000), threshold=0)
scaler = PipeWrapper(StandardScaler)

# Define Deep Learning model
resnet_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
resnet_model.fc = nn.Sequential(nn.Linear(in_features=512, out_features=2, bias=True), Squeeze())
model = TorchTrainer(subestimator=resnet_model, batch_size=32, epochs=20, verbose=False, random_state=0,
                     metrics={'balanced_accuracy': torch_balanced_accuracy, 'accuracy': torch_accuracy, })
```

#### Purge unused training options
So the code is set up to be able to run all the analyses. This will mean that about 2560 models would have to be trained.
To make life a bit quicker, we drop options that we will not use in the later analyses.
This operation in optional, and can be undone in later runs (i.e., you could first run the essential runs, and because these are loaded automatically and then skipped, in a later stage add these unused options for future applications.)


```python
# Retain only the is_responder target label
subj_classes = {'wk_1': wk1_subjs, 'wk_2': data.index}
```


```python
# Preallocate dictionary for storage
if os.path.isfile(results_file):
    results_dict = pickle_in(results_file)
    print(f'Loaded results file: {results_file}.\n The file contained {len(nested_dict_to_df(results_dict))} results')
    dl_aggregates = pickle_in(dl_aggregate_file)
    print(
        f'Loaded aggregates file: {results_file}.\n The file contained {len(nested_dict_to_df(dl_aggregates))} trained Deep Learning models')
else:
    results_dict, dl_aggregates = {}, {}
    print(f'Created new results file: {results_file}')
```

    Loaded results file: D:\repositories\ENIGMA_lean\results\20240418-181259-postRebuttalLongLong\result_dict.pkl.
     The file contained 1302 results
    Loaded aggregates file: D:\repositories\ENIGMA_lean\results\20240418-181259-postRebuttalLongLong\result_dict.pkl.
     The file contained 140 trained Deep Learning models


## 4. Train Machine Learning Models
We have set up an impressive nested dictionary of six levels, with at the deepest level data (X), labels(y) and a stratification variable (c). To train all machine learning models we loop over each item. At the deepest loop level you will find two chucks of code: the first for classical machine learning, and the second for deep learning (see `1)` and `2)`). Although it might have been nice to also create a loop for these two options, they do not share data or models, so that would not have made a lot of sense.

Most calulations are performed in the `cross_validate` and `torch_val_score` functions, which output a list of `n_splits` × `n_repeats` of test outcome values. These results are stored in the `results_dict`. For the exact storage location in this dictionary we use the same `experiment_specifier` that we used to get the data from the `data_dict`. Finally, we can simply convert this nested dictionary to a multi index Pandas DataFrame.

By default, the training loop checks if results are preexisting, to allow for continuation of training and prevent overwriting.


```python
# Print progress
n_tasks = count_values(data_dict) * len(cv_schemes) * len(classifiers) * len(target_labels)
n_tasks = 1512
pb = ProgressBar(n_tasks, desc='models trained')

scorers = 'accuracy', 'balanced_accuracy',

# For explanation on these six nested loops please see the list at 3.
with Timer():
    for subj_class, subj_idxs in subj_classes.items():
        for target_label in target_labels:
            for dtype, data_a in data_dict.items():
                for dbc_presence, data_b in data_a.items():
                    for population_name, (X, ys, c) in data_b.items():
                        subj_incl = [i for i in X.index if i.split('|')[0] in subj_idxs]
                        # Define True label
                        X_safe = X.loc[subj_incl].drop(columns=[target_label], errors='ignore')
                        y = ys.loc[subj_incl][target_label]
                        c = c.loc[subj_incl]
                        for cv_name, (cv, harmonizer) in cv_schemes.items():
                            for classifier in classifiers:
                                try:
                                    if population_name == 'Extremes' and target_label == 'is_extreme' or \
                                            len(c.unique()) == 1 and cv_name == 'Site':
                                        # Can't predict on own population or do LSO-CV on one site
                                        continue

                                    if subj_class == 'wk_1' and (
                                            dbc_presence != 'dbc_no' or
                                            population_name != 'All' or
                                            cv_name != 'Fold' or
                                            target_label != 'is_responder'):
                                        # We only want to run the sensitivity analysis of the weeks of cutoff
                                        # for runs that are relevant to the primary analysis
                                        continue

                                    # 1) Classical Machine Learning: Specify which Classical ML experiment we are going to run
                                    experiment_specifier = dtype, dbc_presence, population_name, cv_name, classifier.__class__.__name__, target_label, subj_class, 'done'

                                    previous_result = safe_dict_get(results_dict, *experiment_specifier)
                                    if previous_result['done'] is None:  # Skip if results exist
                                        # Get CV-scores, score and a priori chance
                                        pipeline = make_pipeline(imputer, regressor, harmonizer, scaler, imputer,
                                                                 selector,
                                                                 classifier, )
                                        # Duplicate Generator. (y and groups are not in use at the same time)
                                        split_score, split_null = tee(cv.split(X_safe, y=c, groups=c))
                                        cross_val_scores = cross_validate(
                                            estimator=pipeline,
                                            X=X_safe,
                                            y=y,
                                            cv=split_score,
                                            scoring=scorers
                                        )
                                        # Extract and assign the relevant scoring items
                                        for scorer in scorers:
                                            previous_result[scorer] = cross_val_scores[f'test_{scorer}']

                                        # The tricky thing with perfectly balanced samples is that with n_splits -> inf, null_acc -> 0.0
                                        previous_result['null'] = [np.mean(y[test] == y[train].mode()[0]) for
                                                                   train, test in list(split_null)]
                                        # NB: CV order: [f1r1, f2r1, f3r1, f1r2, f2r2, f3r2] for 3 fold (f) 2 repeat (r)
                                        # Save results
                                        previous_result['done'] = True
                                        pickle_out(results_dict, results_file)
                                    pb()

                                    if dbc_presence == 'dbc_no' and dtype == 'vec':
                                        for X_arr, letter in zip((X_thc_arr, X_are_arr), 'TA'):
                                            # 2) Deep Learning: Specify which Deep Learning experiment we are going to run
                                            dl_specifier = f'2D{letter}', dbc_presence, population_name, cv_name, 'ResNet', target_label, subj_class, 'done'
                                            previous_dl_result = safe_dict_get(results_dict, *dl_specifier)
                                            if previous_dl_result['done'] is None:  # Skip if results exist
                                                pop_idx = [idx for idx, subj_id in enumerate(X_thc.index) if
                                                           subj_id in y.index]
                                                # Get CV-scores, score and a priori chance
                                                torch_results = torch_val_score(model, X_arr[pop_idx], y, cv, groups=c,
                                                                                verbose=False,
                                                                                return_pipeline='aggregate')
                                                previous_dl_result['balanced_accuracy'] = np.array(
                                                    [r['balanced_accuracy'] for r in torch_results['test']])
                                                previous_dl_result['accuracy'] = np.array(
                                                    [r['accuracy'] for r in torch_results['test']])
                                                previous_dl_result['null'] = torch_results['null']

                                                dl_aggregate = safe_dict_get(dl_aggregates, *dl_specifier[:-1])
                                                dl_aggregate[dl_specifier[-2]] = torch_results['pipeline']
                                                # Save results
                                                pickle_out(results_dict, results_file)
                                                previous_dl_result['done'] = True
                                                pickle_out(dl_aggregates, dl_aggregate_file)
                                            pb()
                                except (KeyboardInterrupt, SystemExit) as e:
                                    raise e
                                except:
                                    print('A MAJOR ISSUE!!', *experiment_specifier)
                                    pass

results_table = nested_dict_to_df(results_dict)
# The table is ordered by the order of the nesting of the calculation.
# Reorder the index to be consistent with the order described in the manuscript.
results_table = results_table.reorder_levels([5, 0, 1, 4, 3, 2, 6])
```


    VBox(children=(IntProgress(value=0, bar_style='info', description='Progress:', layout=Layout(width='50%'), max…


    IOPub message rate exceeded.
    The notebook server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--NotebookApp.iopub_msg_rate_limit`.
    
    Current values:
    NotebookApp.iopub_msg_rate_limit=1000.0 (msgs/sec)
    NotebookApp.rate_limit_window=3.0 (secs)

​    


```python
results_table = nested_dict_to_df(results_dict)
# The table is ordered by the order of the nesting of the calculation.
# Reorder the index to be consistent with the order described in the manuscript.
results_table = results_table.reorder_levels([5, 0, 1, 4, 3, 2, 6])
```

Display an example of the training process of a deep learning model as a check.


```python
# Specify which DL result to look at as an deep learning configuration:
a_dl_res = dl_aggregates['2DT']['dbc_no']['Extremes']['Fold']['ResNet']['is_responder']['wk_2']  # A Deep Learning Result
# Show it
lw = 4
dl_fig, axes = plt.subplots(3, 2, figsize=(9, 6))
line_colors = colors.blue, colors.orange,
for u, (partition_name, partition) in enumerate(a_dl_res.items()):
    for v, (metric_name, metric_values) in enumerate(partition.items()):
        # Plot all learning curves
        axes[v, u].plot(metric_values.T, color='#aaa')
        # Plot the mean learning curve
        axes[v, u].plot(metric_values.mean(0), color=line_colors[u], lw=lw)
        axes[v, u].set(
            title=f'{partition_name.capitalize()}' if u else '',
            ylabel=metric_name if not u else None,
            xlabel='Batch' if v else None,
            yscale='log' if 'Log' in metric_name else 'linear',
        )
        # Only draw a legend once
        if not u and v == 2:
            axes[v, u].legend(frameon=False, handles=[
                Line2D([], [], color=line_colors[0], label='Mean loss', linewidth=lw, ),
                Line2D([], [], color=line_colors[1], label='Mean accuracy', linewidth=lw, ),
                Line2D([], [], color='#aaaaaa', label='Individual folds')],
                              )
dl_fig.suptitle('Deep Learning Training Progression')
dl_fig.tight_layout()
dl_fig.savefig(figdir.append(f'dl_training.png'))
dl_fig.show()
```


​    
![png](.readme/output_55_0.png)
​    


Let us drop the LRC since it was only for development purposes, and RandomForestClassifier because it is not included in the main analyses.


```python
results_table_with_alternate_classifiers = deepcopy(results_table)
results_table = results_table[~(results_table.index.get_level_values(3) == 'LogisticRegression')]
results_table = results_table[~(results_table.index.get_level_values(3) == 'RandomForestClassifier')]
```

## 5. Statistics
From our results we will compute statistics and add them to the data frame to create a data frame aptly called:  `results_and_stats`
These statistics are: population size, T-statisic, p-value, Bayes factor and Effect size. Then combine statistics and results tables, and separate cortical (primary) from subcortical (exploratory) findings.


```python
statistics_calculator = calc_stats_wrapper(population_indices, data_dict, n_splits)
stats_table = statistics_calculator(results_table)

# Horizontal concat the statistics table to the results table
results_and_stats = pd.concat((results_table, stats_table), axis=1)
pickle_out(results_and_stats, restats_file)

# Split our results for primary and exploratory analyses
cortical_results = results_and_stats.loc[(sN, sN, ['dbc_no', 'dbc_yes']), :]  # Limit primary analysis to cortical data
```

## 6. Results
### 6.1 Primary results
This line means: present the average K-Fold cross-validation result for predicting response in the entire population. These results include multiple data types (ROI, 1D-vector and 2D projection; multiple ML models (GBC, SVC, ResNet) and with and without behavioural, clinical and demographic data.

`is_strict` means that we limit our selected data frame to the value specified for the level on which a comparison is requested. i.e.: The level we are requesting a comparison for (6) is considered with predicted outcomes. Here we only want to see the value that we specified (`is_responder`). When is_strict were `False` (default), we would also see the other possible predicted values `is_female`, `is_remitter` and `is_extreme`. It would not have affected the printed value.


```python
distiller = TableDistiller(cortical_results, 'is_responder', None, 'dbc_no', None, 'Fold', 'All', 'wk_2',
                           is_strict=True, verbose=True, n_splits=n_splits)
distiller(1)
```

    (1) 'is_female', 'is_remitter', 'is_responder', 'is_extreme'
    (2) '2DT', 'vec', 'roi', '2DA'
    (3) 'dbc_no', 'dbc_yes'
    (4) 'ResNet', 'GradientBoostingClassifier', 'SVC'
    (5) 'Site', 'Fold'
    (6) 'All', 'Extremes', 'SameResponders', 'LongTreated', 'Hiroshima'
    (7) 'wk_2', 'wk_1'
                     |  bacc  |  sbac  |  acc   |  sac   |  null  |  snul  | p_val  | t_stat | L10BF  |
    1.is_responder   | 51.5%  |  7.9%  | 57.7%  |  8.9%  | 56.9%  |  7.7%  | 0.366  | -0.951 | -0.922 |

​    

### 6.2 Secondary Results
#### 6.2.1 Design Configurations
This line means: present the average K-Fold cross-validation result for predicting response in the entire population. Here, we split the results out three times, based on data type, based on the inclusion of clinical data and based on the machine learning method used. `is_strict` does not have an effect here, since we do not specify any value for any level we are requesting(1, 2 and 5 are `None`).


```python
distiller.is_strict = False
distiller(2, 3, 4)
```

                     |  bacc  |  sbac  |  acc   |  sac   |  null  |  snul  | p_val  | t_stat | L10BF  |
    2.No Significant difference (p_val: 0.088, F=2.2354) among:
     -2DT            | 54.6%  |  9.7%  | 57.1%  |  9.6%  | 56.9%  |  7.6%  | 0.436  | -0.815 | -1.009 |
     -vec            | 51.7%  |  6.0%  | 59.4%  |  8.1%  | 56.9%  |  7.6%  | 0.622  | -0.511 | -1.095 |
     -roi            | 49.6%  |  7.4%  | 56.6%  |  9.0%  | 56.9%  |  7.9%  | 0.173  | -1.478 | -0.679 |
     -2DA            | 51.6%  |  9.2%  | 57.0%  |  9.2%  | 56.9%  |  7.6%  | 0.385  | -0.913 | -0.973 |
    
    3.No Significant difference (p_val: 0.770, t_stat0.302 BF=-1.12e+00, d=4.24e-02) between:
     -dbc_no         | 51.5%  |  7.9%  | 57.7%  |  8.9%  | 56.9%  |  7.7%  | 0.366  | -0.951 | -0.922 |
     -dbc_yes        | 50.3%  |  6.8%  | 57.7%  |  8.6%  | 56.9%  |  7.7%  | 0.283  | -1.142 | -0.833 |
    
    4.Significant difference (p_val: 2.3e-05, F=12.2878) among:
     -ResNet         | 53.1%  |  9.6%  | 57.1%  |  9.4%  | 56.9%  |  7.6%  | 0.410  | -0.864 | -0.991 |
     -GradientBoos...| 52.0%  |  9.1%  | 56.5%  |  9.2%  | 56.9%  |  7.7%  | 0.300  | -1.099 | -0.812 |
     -SVC            | 49.3%  |  2.4%  | 59.5%  |  7.9%  | 56.9%  |  7.7%  | 0.397  | -0.890 | -0.962 |

​    

#### 6.2.2 Secondary: Effect of Leave-Site-Out CV
This line means: present the average results for predicting response in the entire population using either K-Fold os Leave-Site-Out cross validation. `is_strict` does not have an effect here, since we do not specify any value for the level we are requesting (4 is `None`)


```python
distiller(5)
```

                     |  bacc  |  sbac  |  acc   |  sac   |  null  |  snul  | p_val  | t_stat | L10BF  |
    5.No Significant difference (p_val: 0.338, t_stat-1.011 BF=-9.25e-01, d=-6.99e-01) between:
     -Site           | 50.0%  | 10.8%  | 54.6%  | 13.6%  | 65.7%  | 19.9%  | 0.347  | -0.992 | -0.930 |
     -Fold           | 51.5%  |  7.9%  | 57.7%  |  8.9%  | 56.9%  |  7.7%  | 0.366  | -0.951 | -0.922 |

​    

### 6.3 Explorative Results
#### 6.3.1 Performance in Subpopulations
Compared to previous analyses, we do not specify a population (level 3 is `None`). Thus, this line means: present the average results for predicting response in each of the subpopulations using K-Fold cross-validation and any type of data, machine learning method or inclusion of clinical data. Like earlier, `is_strict` does not have an effect here, since we do not specify any value for the level we are requesting (3 is `None`)


```python
distiller = TableDistiller(subc_compare, 'is_responder', None, 'dbc_no', None, 'Fold', None, 'wk_2', n_splits=n_splits)
distiller(6)
```

                     |  bacc  |  sbac  |  acc   |  sac   |  null  |  snul  | p_val  | t_stat | L10BF  |
    6.Significant difference (p_val: 4.4e-56, F=171.8782) among:
     -All            | 51.5%  |  7.9%  | 57.7%  |  8.9%  | 56.9%  |  7.7%  | 0.366  | -0.951 | -0.922 |
     -Extremes       | 63.8%  | 12.0%  | 63.8%  | 12.0%  | 50.0%  |  0.0%  | 0.007* | 3.440  | 1.241  |
     -SameResponders | 50.8%  | 11.4%  | 50.7%  | 12.3%  | 51.1%  | 11.2%  | 0.864  | -0.177 | -0.850 |
     -LongTreated    | 52.0%  | 10.1%  | 63.1%  | 12.7%  | 65.6%  | 11.7%  | 0.425  | -0.836 | 0.117  |
     -Hiroshima      | 46.2%  | 14.1%  | 43.3%  | 15.5%  | 44.3%  | 14.4%  | 0.798  | -0.263 | -0.811 |

​    

#### 6.3.2 Explorative II: Subcortical based Prediction


```python
added_data = results_and_stats.index.get_level_values(2)
subc_compare = results_and_stats[np.logical_or(added_data == 'dbc_no', added_data == 'sub_add')]
distiller = TableDistiller(subc_compare, 'is_responder', None, None, None, 'Fold', 'All', 'wk_2', n_splits=n_splits)
distiller(3)
```

                     |  bacc  |  sbac  |  acc   |  sac   |  null  |  snul  | p_val  | t_stat | L10BF  |
    3.No Significant difference (p_val: 0.627, t_stat0.504 BF=-1.03e+00, d=8.68e-02) between:
     -sub_add        | 50.6%  |  6.9%  | 57.6%  |  8.7%  | 56.9%  |  7.9%  | 0.214  | -1.338 | -0.776 |
     -dbc_no         | 51.5%  |  7.9%  | 57.7%  |  8.9%  | 56.9%  |  7.7%  | 0.366  | -0.951 | -0.922 |

​    

##### 6.3.3.1 Repeating our primary analyses on the Extremes Population


```python
distiller = TableDistiller(cortical_results, 'is_responder', None, 'dbc_no', None, 'Fold', 'Extremes', 'wk_2',
                           is_strict=True, spacing=15, n_splits=n_splits)
distiller(1)
distiller.is_strict = False
distiller(2, 3, 4, 5)
```

                     |  bacc  |  sbac  |  acc   |  sac   |  null  |  snul  | p_val  | t_stat | L10BF  |
    1.is_responder   | 63.8%  | 12.0%  | 63.8%  | 12.0%  | 50.0%  |  0.0%  | 0.007* | 3.440  | 1.241  |
    
                     |  bacc  |  sbac  |  acc   |  sac   |  null  |  snul  | p_val  | t_stat | L10BF  |
    2.Significant difference (p_val: 5.0e-05, F=8.2521) among:
     -2DT            | 62.9%  | 13.2%  | 62.9%  | 13.2%  | 50.0%  |  0.0%  | 0.020* | 2.807  | 0.617  |
     -vec            | 64.0%  | 11.1%  | 64.0%  | 11.1%  | 50.0%  |  0.0%  | 0.005* | 3.651  | 1.467  |
     -roi            | 66.4%  | 11.2%  | 66.4%  | 11.2%  | 50.0%  |  0.0%  | 0.002* | 4.226  | 2.000  |
     -2DA            | 58.9%  | 12.3%  | 58.9%  | 12.3%  | 50.0%  |  0.0%  | 0.067  | 2.079  | -0.105 |
    
    3.Significant difference (p_val: nan, t_statnan BF=9.49e-02, d=1.45e-02) between:
     -dbc_no         | 63.8%  | 12.0%  | 63.8%  | 12.0%  | 50.0%  |  0.0%  | 0.007* | 3.440  | 1.241  |
     -dbc_yes        | 65.0%  | 11.3%  | 65.0%  | 11.3%  | 50.0%  |  0.0%  | 0.004* | 3.906  | 1.658  |
    
    4.Significant difference (p_val: 9.2e-05, F=10.4957) among:
     -ResNet         | 60.9%  | 12.9%  | 60.9%  | 12.9%  | 50.0%  |  0.0%  | 0.037* | 2.443  | 0.256  |
     -GradientBoos...| 65.1%  | 11.9%  | 65.1%  | 11.9%  | 50.0%  |  0.0%  | 0.005* | 3.744  | 1.467  |
     -SVC            | 65.3%  | 10.6%  | 65.3%  | 10.6%  | 50.0%  |  0.0%  | 0.003* | 4.133  | 2.000  |
    
    5.No Significant difference (p_val: 0.889, t_stat0.143 BF=-8.24e-01, d=5.69e-01) between:
     -Site           | 44.0%  | 14.6%  | 40.0%  | 16.0%  | 46.5%  | 29.3%  | 0.781  | -0.287 | -0.973 |
     -Fold           | 63.8%  | 12.0%  | 63.8%  | 12.0%  | 50.0%  |  0.0%  | 0.007* | 3.440  | 1.241  |

​    

##### 6.3.3.2 Explorative III: All kinds of ROI combinations


```python
distiller = TableDistiller(results_and_stats, 'is_responder', None, None, None, 'Fold', 'All', 'wk_2')
distiller(3)
```

                     |  bacc  |  sbac  |  acc   |  sac   |  null  |  snul  | p_val  | t_stat | L10BF  |
    3.Significant difference (p_val: 2.9e-04, F=5.1764) among:
     -sub_dbc        | 54.5%  |  8.4%  | 56.9%  |  8.6%  | 56.9%  |  7.9%  | 0.856  | -0.187 | -1.150 |
     -dbc_no         | 51.5%  |  7.9%  | 57.7%  |  8.9%  | 56.9%  |  7.7%  | 0.366  | -0.951 | -0.922 |
     -sub_nly        | 53.7%  |  8.3%  | 59.6%  |  8.6%  | 56.9%  |  7.9%  | 0.730  | -0.356 | -1.133 |
     -dbc_yes        | 50.3%  |  6.8%  | 57.7%  |  8.6%  | 56.9%  |  7.7%  | 0.283  | -1.142 | -0.833 |
     -sub_all        | 53.8%  |  7.8%  | 59.8%  |  8.6%  | 56.9%  |  7.9%  | 0.809  | -0.249 | -1.132 |
     -sub_add        | 50.6%  |  6.9%  | 57.6%  |  8.7%  | 56.9%  |  7.9%  | 0.214  | -1.338 | -0.776 |
     -dbc_nly        | 52.2%  |  8.5%  | 57.7%  |  8.9%  | 56.9%  |  7.9%  | 0.425  | -0.835 | -1.002 |

​    


```python
distiller = TableDistiller(results_and_stats, 'is_responder', None, None, None, 'Fold', 'Extremes', 'wk_2')
distiller(3)
```

                     |  bacc  |  sbac  |  acc   |  sac   |  null  |  snul  | p_val  | t_stat | L10BF  |
    3.Significant difference (p_val: 2.1e-06, F=8.3331) among:
     -sub_dbc        | 62.2%  | 13.1%  | 62.2%  | 13.1%  | 50.0%  |  0.0%  | 0.025* | 2.687  | 0.490  |
     -dbc_no         | 63.8%  | 12.0%  | 63.8%  | 12.0%  | 50.0%  |  0.0%  | 0.007* | 3.440  | 1.241  |
     -sub_nly        | 61.1%  | 12.2%  | 61.1%  | 12.2%  | 50.0%  |  0.0%  | 0.028* | 2.625  | 0.433  |
     -dbc_yes        | 65.0%  | 11.3%  | 65.0%  | 11.3%  | 50.0%  |  0.0%  | 0.004* | 3.906  | 1.658  |
     -sub_all        | 60.9%  | 12.9%  | 60.9%  | 12.9%  | 50.0%  |  0.0%  | 0.039* | 2.420  | 0.205  |
     -sub_add        | 66.8%  | 11.3%  | 66.8%  | 11.3%  | 50.0%  |  0.0%  | 0.002* | 4.320  | 1.922  |
     -dbc_nly        | 56.9%  | 12.3%  | 56.9%  | 12.3%  | 50.0%  |  0.0%  | 0.039* | 2.408  | 0.210  |

​    




```python
distiller = TableDistiller(cortical_results, 'is_responder', None, 'dbc_no', None, 'Fold', 'LongTreated', 'wk_2',
                           is_strict=True, verbose=False, spacing=15, n_splits=n_splits)
distiller(1)
distiller.is_strict = False
distiller(2, 3, 4, 5, 6)
```

                     |  bacc  |  sbac  |  acc   |  sac   |  null  |  snul  | p_val  | t_stat | L10BF  |
    1.is_responder   | 52.0%  | 10.1%  | 63.1%  | 12.7%  | 65.6%  | 11.7%  | 0.425  | -0.836 | 0.117  |
    
                     |  bacc  |  sbac  |  acc   |  sac   |  null  |  snul  | p_val  | t_stat | L10BF  |
    2.No Significant difference (p_val: 0.090, F=2.2188) among:
     -2DT            | 54.0%  | 13.1%  | 61.9%  | 14.2%  | 65.6%  | 11.7%  | 0.459  | -0.773 | -0.866 |
     -vec            | 51.9%  | 10.1%  | 63.7%  | 12.5%  | 65.6%  | 11.7%  | 0.451  | -0.789 | 0.570  |
     -roi            | 51.1%  |  9.0%  | 62.5%  | 11.9%  | 65.6%  | 11.7%  | 0.220  | -1.317 | 0.687  |
     -2DA            | 51.7%  |  8.1%  | 64.2%  | 12.7%  | 65.6%  | 11.7%  | 0.653  | -0.465 | -0.946 |
    
    3.Significant difference (p_val: nan, t_statnan BF=5.15e-01, d=nan) between:
     -dbc_no         | 52.0%  | 10.1%  | 63.1%  | 12.7%  | 65.6%  | 11.7%  | 0.425  | -0.836 | 0.117  |
     -dbc_yes        | 51.7%  | 10.6%  | 63.3%  | 13.1%  | 65.6%  | 11.7%  | 0.411  | -0.863 | 0.587  |
    
    4.Significant difference (p_val: 1.3e-04, F=10.0763) among:
     -ResNet         | 52.9%  | 10.9%  | 63.1%  | 13.5%  | 65.6%  | 11.7%  | 0.551  | -0.619 | -0.906 |
     -GradientBoos...| 53.1%  | 13.4%  | 60.6%  | 12.3%  | 65.6%  | 11.7%  | 0.320  | -1.053 | -0.744 |
     -SVC            | 50.0%  |  0.0%  | 65.6%  | 11.7%  | 65.6%  | 11.7%  |  nan   |  nan   | 2.000  |
    
    5.Significant difference (p_val: nan, t_statnan BF=1.65e-02, d=nan) between:
     -Site           | 48.6%  |  8.1%  | 47.1%  | 17.8%  | 40.7%  | 18.9%  | 0.184  | 1.438  | 0.041  |
     -Fold           | 52.0%  | 10.1%  | 63.1%  | 12.7%  | 65.6%  | 11.7%  | 0.425  | -0.836 | 0.117  |
    
    6.Significant difference (p_val: 4.4e-56, F=171.8782) among:
     -All            | 51.5%  |  7.9%  | 57.7%  |  8.9%  | 56.9%  |  7.7%  | 0.366  | -0.951 | -0.922 |
     -Extremes       | 63.8%  | 12.0%  | 63.8%  | 12.0%  | 50.0%  |  0.0%  | 0.007* | 3.440  | 1.241  |
     -SameResponders | 50.8%  | 11.4%  | 50.7%  | 12.3%  | 51.1%  | 11.2%  | 0.864  | -0.177 | -0.850 |
     -LongTreated    | 52.0%  | 10.1%  | 63.1%  | 12.7%  | 65.6%  | 11.7%  | 0.425  | -0.836 | 0.117  |
     -Hiroshima      | 46.2%  | 14.1%  | 43.3%  | 15.5%  | 44.3%  | 14.4%  | 0.798  | -0.263 | -0.811 |

​    

#### 6.3.3 Explorative III: Performance of alternative classifiers


```python
all_clf_distiller = TableDistiller(pd.concat(
    (results_table_with_alternate_classifiers, statistics_calculator(results_table_with_alternate_classifiers)),
    axis=1), 'is_responder', None, 'dbc_no', 'RandomForestClassifier', 'Fold', 'All', 'wk_2', is_strict=True, n_splits=n_splits)
all_clf_distiller(1)
all_clf_distiller.is_strict = False
all_clf_distiller(4)
```

                     |  bacc  |  sbac  |  acc   |  sac   |  null  |  snul  | p_val  | t_stat | L10BF  |
    1.is_responder   | 49.3%  |  6.7%  | 56.6%  |  8.2%  | 56.9%  |  7.7%  | 0.143  | -1.603 | -0.590 |
    
                     |  bacc  |  sbac  |  acc   |  sac   |  null  |  snul  | p_val  | t_stat | L10BF  |
    4.Significant difference (p_val: 1.7e-20, F=34.2239) among:
     -GradientBoos...| 52.0%  |  9.1%  | 56.5%  |  9.2%  | 56.9%  |  7.7%  | 0.300  | -1.099 | -0.812 |
     -LogisticRegr...| 48.5%  | 10.0%  | 51.4%  | 10.3%  | 56.9%  |  7.7%  | 0.030  | -2.574 | 0.557  |
     -ResNet         | 53.1%  |  9.6%  | 57.1%  |  9.4%  | 56.9%  |  7.6%  | 0.410  | -0.864 | -0.991 |
     -RandomForest...| 49.3%  |  6.7%  | 56.6%  |  8.2%  | 56.9%  |  7.7%  | 0.143  | -1.603 | -0.590 |
     -SVC            | 49.3%  |  2.4%  | 59.5%  |  7.9%  | 56.9%  |  7.7%  | 0.397  | -0.890 | -0.962 |

​    

#### 6.3.4 Exploratory IV: Sensitivity analysis of inclusion criteria


```python
distiller = TableDistiller(cortical_results, 'is_responder', None, 'dbc_no', None, 'Fold', 'All', 'wk_1', is_strict=False,
                           n_splits=n_splits)
distiller(7)
```

                     |  bacc  |  sbac  |  acc   |  sac   |  null  |  snul  | p_val  | t_stat | L10BF  |
    7.No Significant difference (p_val: 0.917, t_stat-0.107 BF=-1.10e+00, d=-2.78e-02) between:
     -wk_2           | 51.5%  |  7.9%  | 57.7%  |  8.9%  | 56.9%  |  7.7%  | 0.366  | -0.951 | -0.922 |
     -wk_1           | 51.5%  |  7.7%  | 57.9%  |  8.9%  | 56.9%  |  8.2%  | 0.365  | -0.953 | -0.933 |

​    

### 6.4 Visuals of performance over several analyses


```python
# Human-readable Title Dictionaries
hr_data = OrderedDict({
    'dbc_no': 'cortical',
    'dbc_nly': 'clinical',
    'sub_nly': 'sub-cortical',
    'dbc_yes': 'cortical and clinical',
    'sub_dbc': 'sub-cortical and clinical',
    'sub_add': '(sub-)cortical',
    'sub_all': '(sub-)cortical and clinical',
})
hr_pop = {
    'All': 'Full\nPopulation',
    'Extremes': 'Extremes\nSubpopulation',
}

bins = np.linspace(-80, 80, 17)

queries = flatten(
    [[('is_responder', 'roi', d, 'GradientBoostingClassifier', 'Fold', p, 'wk_2',) for d in hr_data] for p in hr_pop])

n_rows = 2
n_cols = len(queries) // n_rows
xlim = 80
ylim = 0.05  # max(ax.get_ylim()) # example.population.mean() #

cs = [colors.blue] * n_cols + [colors.orange] * n_cols
bacc_fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 6))

for ax_n, query, ax, c in zip(range(n_rows * n_cols), queries, axes.reshape(-1), cs):
    # Get data
    example = results_and_stats.loc[query]
    score, null = example['balanced_accuracy'], example['null']
    if len(example.shape) > 1:
        score, null = flatten(score), flatten(null)

    # Plot
    ax.hist(np.subtract(score, null) * 100, bins, density=True, color=c, label=hr_pop[query[-2]])
    ax.plot([0, 0], [0, ylim], c='k')
    acc_explainer = '' if ax_n else '(bAcc)'
    if not ax_n % n_cols:
        ax.legend(frameon=False)
    ax.text(-xlim, ylim, f'{np.mean(score):.1%}\n{acc_explainer}', va='top', size=14)
    # Format the plot
    ax.set(
        xlim=(-xlim, xlim),
        ylim=(0, ylim),
        title=hr_data[query[2]].capitalize(),
    )
    if not ax_n % n_cols:
        ax.set_ylabel('Normalized frequency')
    ax.set_xlabel('bAcc Δ to null')
    if ax_n < n_cols:
        ax.get_xaxis().set_visible(False)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    plt.suptitle('Predicting Response Using Various Sets of ROI Data in two populations', fontsize=18)
bacc_fig.tight_layout()
bacc_fig.savefig(figdir.append(f'bAcc_histograms.png'))
```


​    
![png](.readme/output_82_0.png)
​    


## Sidenotes
### Secondary Results

Three things are worth mentioning about these results:
1. Why the null score for `Site` is < 50%
2. Why the null score for `Extremes` is < 50%
3. How the bAcc > 50% when Acc < Null

The reasons are:
1. Because sites present varying a-priori response rates, but the mean of all sites is around 50%. This implies that, when a site with dominant class A is in the test set, the dominant class in the train set will likely be B. Thus we often see a mismatch between dominant class in the train set and the test set.
2. The classes in the entire data set are balanced 50%/50%. When we use (even stratified) cross-validation, whenever there are more samples of class A in the train set, there will be more samples of B in the test set. This will incur a deviation from the 50% balance of size ~ `1/(sample_size/n_splits)` (deviates due to rounding of test set sizes).
3. Because the "mean of balanced accuracies" != "the balanced accuracy of the means"

Based on simple calculations using the subject samples and labels we can provide an estimate of the expected `null score` and compare this with the measured null score:


```python
measured_sites_accs = [np.mean(flatten(results_and_stats.loc[q]['null'])) for q in (
    ('is_responder', sN, sN, sN, 'Fold', 'Extremes'), ('is_responder', 'roi', sN, sN, 'Site', 'All')
)]
```


```python
# Retrieve the accuracy score from our results
queries = ('Fold', 'Extremes'), ('Site', 'All')
measured_sites_accs = [np.mean(flatten(results_and_stats.loc[('is_responder', sN, sN, sN, *q)]['null'])) for q in
                       queries]

print('We can say the following about the null accuracy score:')
for i, ((cv_name, (cv_scheme, _)), example_population, measured_sites_acc) in enumerate(
        zip(cv_schemes.items(), ('Extremes', 'All'), measured_sites_accs), 1):
    # Simulate simple
    Xt, yt, ct, = data_dict['roi']['dbc_no'][example_population]
    yt = yt.is_responder

    sites_acc = []
    dominant_class_mismatches, n_splits = 0, 0
    for train_idx, test_idx in cv_scheme.split(yt, y=yt, groups=ct):
        n_splits += 1
        dominant_class_in_train = yt.iloc[train_idx].mode()[0].item()
        dominant_class_in_test = yt.iloc[test_idx].mode()[0].item()
        if dominant_class_in_train != dominant_class_in_test:
            dominant_class_mismatches += 1
        test_samples = yt.iloc[test_idx]
        sites_acc.append(np.mean(test_samples == dominant_class_in_train))

    print(
        f'{i}.  In Leave-{cv_name}-Out CV of {example_population} the dominant class was mismatched {dominant_class_mismatches}/{n_splits} times:\n'
        f'    - Expected: {np.mean(sites_acc):.1%}\n'
        f'    - Measured: {measured_sites_acc:.1%}\n')

s = results_and_stats.loc['is_responder', '2DT', 'dbc_no', 'ResNet', 'Fold', 'All', 'wk_2']['accuracy']
n = results_and_stats.loc['is_responder', '2DT', 'dbc_no', 'ResNet', 'Fold', 'All', 'wk_2']['null']
print(f'3.  The mean-balanced-accuracy (bAcc) is not the same as balanced-accuracy of the means:\n'
      f'    Mean accuracy score: {np.mean(s):.1%} ± {np.std(s):.1%}, Mean null score: {np.mean(n):.1%} ± {np.std(n):.1%}\n'
      f'    - Balanced accuracy of means:  {np.mean(s) / np.mean(n) / 2:.1%}\n'
      f'    - Mean of balanced accuracies: {np.mean([i / j / 2 for i, j in zip(s, n)]):.1%}\n')
```

    We can say the following about the null accuracy score:
    1.  In Leave-Fold-Out CV of Extremes the dominant class was mismatched 80/100 times:
        - Expected: 46.9%
        - Measured: 50.0%
    
    2.  In Leave-Site-Out CV of All the dominant class was mismatched 4/6 times:
        - Expected: 44.3%
        - Measured: 65.9%
    
    3.  The mean-balanced-accuracy (bAcc) is not the same as balanced-accuracy of the means:
        Mean accuracy score: 57.1% ± 9.6%, Mean null score: 56.9% ± 7.6%
        - Balanced accuracy of means:  47.4%
        - Mean of balanced accuracies: 47.9%

​    

Now previously, I have attempted to fix this by replacing in this notebook:
```
  previous_result['null'] = [np.mean(y[test] == y[train].mode()[0]) for train, test in list(split_null)]
```
with
```
  previous_result['null'] = np.array([np.mean(y[test]) if y.mean() == 0.5 else np.mean(y[test] == y[train].mode()[0]) for train, test in list(split_null)])
```
And adding a special case in the `torch_val_score` method when `if y.mean() == 0.5`. But creates an overestimation of the performance and only makes the results harder to interpret, so I have since removed them.

## 6. Interpretarion of Cortical Results
### Calculate
We run simulations for two populations `All` and `Extremes` and two data types, . We store the coefficients in a dictionary called `coef_dict`.


```python
# Parameters of our simulation
n_splits = 40
n_repeats = 2

# Preallocate an array to which we can store the feature importance
importance_array = np.zeros((n_splits * n_repeats, len(bool_var)))

# Define the machine learning pipeline
combat = list(cv_schemes.values())[-1][-1]
pipeline = make_pipeline(imputer, combat, scaler, selector, classifiers[0], )

# Preallocate two dictionaries: One to store the simulation bAccs, one to store results
coef_bacc = make_empty_nest(['All', 'Extremes'], ['dbc_no', 'sub_all'], bottom=[])
coef_dict = deepcopy(coef_bacc)

# We use a ROS to balanced training data. Test data is balanced through RSKF-CV.
ros = RandomOverSampler()

for population_name, coef_dict_b in coef_dict.items():
    for data_type in coef_dict_b:
        X, y, _ = data_dict['roi'][data_type][population_name]
        y = y['is_responder']
        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=0)
        coef_df = pd.DataFrame(columns=X.columns)
        b_accs = []
        for train_index, test_idx in tqdm(cv.split(X, y)):
            X_train, y_train = X.iloc[train_index], y.iloc[train_index]
            X_train, y_train, = ros.fit_resample(X_train, y_train)
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
            X_test, y_test = ros.fit_resample(X_test, y_test)

            pipeline = make_pipeline(imputer, scaler, selector, classifiers[-1], )
            pipeline.fit(X_train, y_train)
            clf = pipeline[-1]
            y_pred = pipeline.predict(X_test)

            # We have already used ROS, so Acc = bAcc, but just for good measure:
            bacc = balanced_accuracy_score(y_test, y_pred)

            coef_df.loc[len(coef_df)] = clf.feature_importances_
            b_accs.append(bacc)

        coef_dict[population_name][data_type] = coef_df.mean()
        coef_bacc[population_name][data_type] = np.mean(b_accs)
```

    80it [01:44,  1.31s/it]
    80it [00:38,  2.10it/s]
    80it [00:35,  2.25it/s]
    80it [00:22,  3.49it/s]


### Cortical Show
Show fancy


```python
# Flip the atlas that matches regions to indices
inv_dkt_atlas_lut = {v: k for k, v in dkt_atlas_lut.items()}

# Assign storage location
lut_dir = get_root('data', 'fsaverage', 'manual_labels')
os.makedirs(lut_dir, exist_ok=True)

dfs = {'All': [], 'Extremes': []}
# pp stands for projection_property: area or thickness
for pop, hemi, map_prop in product(('All', 'Extremes'), ("Left", "Right"), ('surf', 'thick')):
    coef_means = coef_dict[pop]['dbc_no']
    lut_fname = lut_dir.append(f'{hemi[0]}H-LRC_{map_prop}-coefs-{pop}.pkl')
    if not os.path.isfile(lut_fname):
        idx2coef = make_coef_lut(coef_means, hm=hemi[0], property=map_prop, sd=True)
        pickle_out(idx2coef, lut_fname)
        lut_status = 'created'
    else:
        idx2coef = pickle_in(lut_fname)
        lut_status = 'loaded'
    dfs[pop].append(pd.Series({inv_dkt_atlas_lut[k]: v for k, v in idx2coef.items()}, name=f'{hemi}_{map_prop}'))
    print(f'Coefficient Look-Up-Table {lut_status} for {map_prop} in {hemi} hemisphere. ({lut_fname})')
pd.concat(dfs['Extremes'], axis=1)
```

    Coefficient Look-Up-Table created for surf in Left hemisphere. (D:\repositories\ENIGMA_lean\data\fsaverage\manual_labels\LH-LRC_surf-coefs-All.pkl)
    Coefficient Look-Up-Table created for thick in Left hemisphere. (D:\repositories\ENIGMA_lean\data\fsaverage\manual_labels\LH-LRC_thick-coefs-All.pkl)
    Coefficient Look-Up-Table created for surf in Right hemisphere. (D:\repositories\ENIGMA_lean\data\fsaverage\manual_labels\RH-LRC_surf-coefs-All.pkl)
    Coefficient Look-Up-Table created for thick in Right hemisphere. (D:\repositories\ENIGMA_lean\data\fsaverage\manual_labels\RH-LRC_thick-coefs-All.pkl)
    Coefficient Look-Up-Table created for surf in Left hemisphere. (D:\repositories\ENIGMA_lean\data\fsaverage\manual_labels\LH-LRC_surf-coefs-Extremes.pkl)
    Coefficient Look-Up-Table created for thick in Left hemisphere. (D:\repositories\ENIGMA_lean\data\fsaverage\manual_labels\LH-LRC_thick-coefs-Extremes.pkl)
    Coefficient Look-Up-Table created for surf in Right hemisphere. (D:\repositories\ENIGMA_lean\data\fsaverage\manual_labels\RH-LRC_surf-coefs-Extremes.pkl)
    Coefficient Look-Up-Table created for thick in Right hemisphere. (D:\repositories\ENIGMA_lean\data\fsaverage\manual_labels\RH-LRC_thick-coefs-Extremes.pkl)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
    
    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Left_surf</th>
      <th>Left_thick</th>
      <th>Right_surf</th>
      <th>Right_thick</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>bankssts</th>
      <td>0.225564</td>
      <td>0.400976</td>
      <td>0.725218</td>
      <td>0.318618</td>
    </tr>
    <tr>
      <th>caudalanteriorcingulate</th>
      <td>0.714526</td>
      <td>0.733274</td>
      <td>0.236932</td>
      <td>0.532695</td>
    </tr>
    <tr>
      <th>caudalmiddlefrontal</th>
      <td>0.716545</td>
      <td>0.559250</td>
      <td>0.547786</td>
      <td>0.429713</td>
    </tr>
    <tr>
      <th>cuneus</th>
      <td>0.553615</td>
      <td>0.519163</td>
      <td>0.594030</td>
      <td>0.218352</td>
    </tr>
    <tr>
      <th>fusiform</th>
      <td>0.694217</td>
      <td>0.114038</td>
      <td>0.173524</td>
      <td>0.370372</td>
    </tr>
    <tr>
      <th>inferiorparietal</th>
      <td>0.622145</td>
      <td>0.666933</td>
      <td>0.952800</td>
      <td>0.644766</td>
    </tr>
    <tr>
      <th>inferiortemporal</th>
      <td>0.262894</td>
      <td>0.000000</td>
      <td>0.310763</td>
      <td>0.412891</td>
    </tr>
    <tr>
      <th>isthmuscingulate</th>
      <td>0.348311</td>
      <td>0.566538</td>
      <td>0.361238</td>
      <td>0.514188</td>
    </tr>
    <tr>
      <th>lateraloccipital</th>
      <td>0.528657</td>
      <td>0.814262</td>
      <td>0.290009</td>
      <td>0.444446</td>
    </tr>
    <tr>
      <th>lateralorbitofrontal</th>
      <td>0.617164</td>
      <td>0.346796</td>
      <td>0.384433</td>
      <td>0.207585</td>
    </tr>
    <tr>
      <th>lingual</th>
      <td>0.745644</td>
      <td>0.606157</td>
      <td>0.585818</td>
      <td>0.394041</td>
    </tr>
    <tr>
      <th>medialorbitofrontal</th>
      <td>0.354821</td>
      <td>0.682403</td>
      <td>1.000000</td>
      <td>0.420663</td>
    </tr>
    <tr>
      <th>middletemporal</th>
      <td>0.418054</td>
      <td>0.170677</td>
      <td>0.234334</td>
      <td>0.277385</td>
    </tr>
    <tr>
      <th>parahippocampal</th>
      <td>0.968004</td>
      <td>0.117933</td>
      <td>0.101126</td>
      <td>0.127951</td>
    </tr>
    <tr>
      <th>paracentral</th>
      <td>0.407474</td>
      <td>0.131782</td>
      <td>0.209960</td>
      <td>0.545614</td>
    </tr>
    <tr>
      <th>parsopercularis</th>
      <td>0.114000</td>
      <td>0.474552</td>
      <td>0.617022</td>
      <td>0.745656</td>
    </tr>
    <tr>
      <th>parsorbitalis</th>
      <td>0.818598</td>
      <td>0.682201</td>
      <td>0.570766</td>
      <td>0.843756</td>
    </tr>
    <tr>
      <th>parstriangularis</th>
      <td>0.171510</td>
      <td>0.408061</td>
      <td>0.700506</td>
      <td>0.448635</td>
    </tr>
    <tr>
      <th>pericalcarine</th>
      <td>0.212133</td>
      <td>0.468999</td>
      <td>0.196210</td>
      <td>0.336493</td>
    </tr>
    <tr>
      <th>postcentral</th>
      <td>0.617473</td>
      <td>0.655423</td>
      <td>0.469110</td>
      <td>0.574932</td>
    </tr>
    <tr>
      <th>posteriorcingulate</th>
      <td>0.643339</td>
      <td>0.641265</td>
      <td>0.297176</td>
      <td>0.464945</td>
    </tr>
    <tr>
      <th>precentral</th>
      <td>-0.550000</td>
      <td>0.848927</td>
      <td>-0.550000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>precuneus</th>
      <td>0.764550</td>
      <td>0.486661</td>
      <td>0.892318</td>
      <td>0.585712</td>
    </tr>
    <tr>
      <th>rostralanteriorcingulate</th>
      <td>0.764884</td>
      <td>0.772472</td>
      <td>0.501688</td>
      <td>0.288294</td>
    </tr>
    <tr>
      <th>rostralmiddlefrontal</th>
      <td>0.000000</td>
      <td>0.261614</td>
      <td>0.466002</td>
      <td>0.255178</td>
    </tr>
    <tr>
      <th>superiorfrontal</th>
      <td>0.768010</td>
      <td>0.655299</td>
      <td>0.838486</td>
      <td>0.690202</td>
    </tr>
    <tr>
      <th>superiorparietal</th>
      <td>0.543888</td>
      <td>0.385158</td>
      <td>0.779848</td>
      <td>0.256075</td>
    </tr>
    <tr>
      <th>superiortemporal</th>
      <td>0.208468</td>
      <td>0.434077</td>
      <td>0.567895</td>
      <td>0.493174</td>
    </tr>
    <tr>
      <th>frontalpole</th>
      <td>0.716716</td>
      <td>0.514380</td>
      <td>0.505318</td>
      <td>0.592433</td>
    </tr>
    <tr>
      <th>temporalpole</th>
      <td>0.209870</td>
      <td>0.399616</td>
      <td>0.663377</td>
      <td>0.543110</td>
    </tr>
    <tr>
      <th>transversetemporal</th>
      <td>0.279125</td>
      <td>0.472147</td>
      <td>0.543926</td>
      <td>0.157032</td>
    </tr>
    <tr>
      <th>insula</th>
      <td>0.434086</td>
      <td>0.656469</td>
      <td>0.184733</td>
      <td>0.817245</td>
    </tr>
  </tbody>
</table>
</div>




```python
# inverse the dkt_atlas
population = 'Extremes'
dkt_atlas_lut_ud = {v: k for k, v in dkt_atlas_lut.items()}
colnames = []
feature_importances = []
for hm in 'LR':
    for letter in 'surf', 'thick':
        # load the LUT
        lut_fname = lut_dir.append(f'{hm}H-LRC_{letter}-coefs-{population}.pkl')
        feature_importances.append({dkt_atlas_lut_ud[k]: v for k, v in pickle_in(lut_fname).items()})
        colnames.append(f'{hm}H_{letter}')
print(f'Coefficients as provided for imaging of the {population} population:')
print(pd.DataFrame(feature_importances, index=colnames).T)
```

    Coefficients as provided for imaging of the Extremes population:
                               LH_surf  LH_thick   RH_surf  RH_thick
    bankssts                  0.225564  0.400976  0.725218  0.318618
    caudalanteriorcingulate   0.714526  0.733274  0.236932  0.532695
    caudalmiddlefrontal       0.716545  0.559250  0.547786  0.429713
    cuneus                    0.553615  0.519163  0.594030  0.218352
    fusiform                  0.694217  0.114038  0.173524  0.370372
    inferiorparietal          0.622145  0.666933  0.952800  0.644766
    inferiortemporal          0.262894  0.000000  0.310763  0.412891
    isthmuscingulate          0.348311  0.566538  0.361238  0.514188
    lateraloccipital          0.528657  0.814262  0.290009  0.444446
    lateralorbitofrontal      0.617164  0.346796  0.384433  0.207585
    lingual                   0.745644  0.606157  0.585818  0.394041
    medialorbitofrontal       0.354821  0.682403  1.000000  0.420663
    middletemporal            0.418054  0.170677  0.234334  0.277385
    parahippocampal           0.968004  0.117933  0.101126  0.127951
    paracentral               0.407474  0.131782  0.209960  0.545614
    parsopercularis           0.114000  0.474552  0.617022  0.745656
    parsorbitalis             0.818598  0.682201  0.570766  0.843756
    parstriangularis          0.171510  0.408061  0.700506  0.448635
    pericalcarine             0.212133  0.468999  0.196210  0.336493
    postcentral               0.617473  0.655423  0.469110  0.574932
    posteriorcingulate        0.643339  0.641265  0.297176  0.464945
    precentral               -0.550000  0.848927 -0.550000  1.000000
    precuneus                 0.764550  0.486661  0.892318  0.585712
    rostralanteriorcingulate  0.764884  0.772472  0.501688  0.288294
    rostralmiddlefrontal      0.000000  0.261614  0.466002  0.255178
    superiorfrontal           0.768010  0.655299  0.838486  0.690202
    superiorparietal          0.543888  0.385158  0.779848  0.256075
    superiortemporal          0.208468  0.434077  0.567895  0.493174
    frontalpole               0.716716  0.514380  0.505318  0.592433
    temporalpole              0.209870  0.399616  0.663377  0.543110
    transversetemporal        0.279125  0.472147  0.543926  0.157032
    insula                    0.434086  0.656469  0.184733  0.817245



```python
#show colormap
n_shades = 200
cbar_lim = -2, 2
colorbar = get_rgb_cbar(n_shades=n_shades)

# #Plot with label brain (annot)
# imgs = 'lh_annot', 'lh_surf', 'lh_thick', 'colorbar', 'rh_annot', 'rh_surf', 'rh_thick',
# imgs_fpath = r"D:\repositories\ENIGMA\documents\graphics\Blender\{}.png"
# img_titles = 'Labels', 'Surface Coefficients', 'Thickness Coefficients', '',\
#              'Labels', 'Surface Coefficients', 'Thickness Coefficients',

#Plot without label brain (annot)
imgs = 'lh_surf', 'lh_thick', 'colorbar', 'rh_surf', 'rh_thick',
imgs_fpath = figdir.append('{}.png')
img_titles = 'Surface Area', 'Cortical Thickness', '', \
    'Surface Area', 'Cortical Thickness',

n_rows = 2
n_cols = len(imgs) // n_rows
for img_pop in ('Extremes',):
    brain_fig, axes = plt.subplots(nrows=n_rows,
                                   ncols=len(imgs) // n_rows + 1,
                                   figsize=(n_cols * 3 + 2, 2 * n_rows),
                                   gridspec_kw={'width_ratios': n_cols * [1] + [0.1]})
    axes = axes.reshape(-1)
    for ax, title, img_path in zip(axes, img_titles, imgs):
        # Select and display image
        formatted_img_path = imgs_fpath.format(img_path if 'annot' in img_path else f'{img_path}_{img_pop}')
        img = colorbar if img_path == 'colorbar' else mpimg.imread(formatted_img_path)
        ax.imshow(img)

        # Set axes properties
        ax.set(xticks=[], title=title)
        if img_path == 'colorbar':
            ax.set_yticks(np.linspace(0, n_shades, 5))
            ax.grid(False)
            ax.set_yticklabels(np.linspace(cbar_lim[-1], cbar_lim[0], 5))
            ax.set_ylabel('Norm. Coefficient (σ)')
        else:
            ax.set_yticks([])
    axes[-1].axis('off')
    axes[0].set_ylabel('Left Hemisphere')
    axes[n_cols + 1].set_ylabel('Right Hemisphere')
    perf = np.mean(results_dict['roi']['dbc_no'][img_pop]['Fold']['GradientBoostingClassifier']['is_responder']['wk_2']['balanced_accuracy'])
    n_part = len(data_dict['roi']['dbc_no'][img_pop][0])
    # plt.suptitle(f'Coefficients for {img_pop} Population (n = {n_part:.0f}, bAcc = {bacc:.1%})')
    plt.suptitle(f'Coefficients for the Extreme (Non-)responders Subpopulation (n = {n_part:.0f}, bAcc = {perf:.1%})')
    brain_fig.savefig(figdir.append(f'{img_pop}-3Dbrains.tiff'))
plt.show()
```


​    
![png](.readme/output_93_0.png)
​    



```python
X_thc2, _ = load_proj_df(data, 'Thickness')

titles = 'Median Imporance', 'Median thickness', 'aparc.a2009s.annot'

map_imp = np.reshape(np.median(importance_array, axis=0), [30, 30])
map_thk = np.reshape(np.median(X_thc2, axis=0), [30, 30])
map_img = np.load(results_dir.append('aggregated_label_annot_res-32.npy'))

fig, axes = plt.subplots(1, len(titles), figsize=(10, 4))
h_thk = axes[0].imshow(map_imp, cmap='viridis', clim=(-map_imp.std() * 2, map_imp.std() * 2), )
h_imp = axes[1].imshow(map_thk, cmap='jet')
h_img = axes[2].imshow(map_img)

for ax, mappable, title in zip(axes, [h_thk, h_imp, None], titles):
    ax.grid(False)
    ax.set(title=title, xticks=[], yticks=[])
    if mappable:
        fig.colorbar(mappable, ax=ax)
fig.tight_layout()
fig.show()
```

    100%|██████████| 252/252 [00:00<00:00, 2571.51it/s]
​    


​    
![png](.readme/output_94_1.png)
​    


Note that the labeling we were provided in the ROI analysis (`aparc.DKTatlas40.annot`) is different from the labeling we were provided in the projection analysis (`aparc.a2009s.annot`) which is shown in the figure above. Also, one is not simply as subset of the other.

I Wish we would have the DKTatlas for the projection data so we could see if the importance found in the projection analysis lines up with the importance found in the ROI analysis.

### Subcortical Show
Show a table


```python
# A function to strip subcortical ROIs from their hemisphere information
rm_hem = lambda x: x[1:] if x[0] in hems else x

# Get hemisphere options, and ROI options so we can strip them from column names
hems = 'LRM'

# Features of interest, pick one
feature_set = ['sub_all', 'dbc_no'][0]

imp_dfs = {}
for population, population_coef_dict in coef_dict.items():
    bacc = np.mean(coef_bacc[population][feature_set])

    # Subcortical importances
    subc = {k: v for k, v in population_coef_dict[feature_set].items() if k in subcortical_rois}

    # Apply the subcortical strip function to the index to get all unique ROIs
    no_hem_subc_rois = set(population_coef_dict[feature_set].index.map(rm_hem))

    # Preallocate a numpty array to store our
    subc_coefs = np.empty((len(no_hem_subc_rois), len(hems)))
    subc_coefs[:] = np.nan

    # Collect coef by ROI and hemisphere, and appoint directly into a numpy array
    for i, v in subc.items():
        roi_name = rm_hem(i)
        hemi = i.replace(roi_name, '')
        hemi = hemi if hemi else 'M'
        x = list(no_hem_subc_rois).index(roi_name)
        y = hems.index(hemi)
        subc_coefs[x, y] = v
    subc_coefs = np.concatenate([subc_coefs, np.nansum(np.abs(subc_coefs), 1)[:, np.newaxis]], axis=1)

    # Convert the array to a table, and do formatting
    subc_relevance_table = pd.DataFrame(data=subc_coefs, columns=list(hems) + ['SAR'],
                                        index=no_hem_subc_rois)  # We use Sum of Absolute Relevances
    subc_relevance_table = subc_relevance_table.sort_values(by='SAR', ascending=False)  # for sorting
    subc_relevance_table = subc_relevance_table.drop(columns=['SAR'])  # and drop it after.
    subc_relevance_table = subc_relevance_table.replace(np.nan, pd.NA)
    subc_relevance_table = subc_relevance_table.rename(columns={'L': 'Left', 'R': 'Right', 'M': 'Middle / NA'})

    # Store the table
    imp_dfs[population] = subc_relevance_table
```


```python
imp_dfs['All']
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
    
    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Left</th>
      <th>Right</th>
      <th>Middle / NA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>pal</th>
      <td>0.02301</td>
      <td>0.024482</td>
      <td>0.025529</td>
    </tr>
    <tr>
      <th>amyg</th>
      <td>0.022517</td>
      <td>0.01921</td>
      <td>0.018849</td>
    </tr>
    <tr>
      <th>accumb</th>
      <td>0.024082</td>
      <td>0.016063</td>
      <td>0.020334</td>
    </tr>
    <tr>
      <th>put</th>
      <td>0.019159</td>
      <td>0.018086</td>
      <td>0.018402</td>
    </tr>
    <tr>
      <th>hippo</th>
      <td>0.01731</td>
      <td>0.018122</td>
      <td>0.015892</td>
    </tr>
    <tr>
      <th>thal</th>
      <td>0.017074</td>
      <td>0.014313</td>
      <td>0.01622</td>
    </tr>
    <tr>
      <th>caud</th>
      <td>0.014919</td>
      <td>0.014384</td>
      <td>0.014036</td>
    </tr>
    <tr>
      <th>LatVent</th>
      <td>0.015227</td>
      <td>0.013728</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>ICV</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>0.020306</td>
    </tr>
    <tr>
      <th>vent</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>0.015081</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>is_ad</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>is_recurrent</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>is_female</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>Age_of_Onset</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>Normalized_Pretreatment_Severity</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
  </tbody>
</table>
</div>




```python
imp_dfs['Extremes']
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
    
    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Left</th>
      <th>Right</th>
      <th>Middle / NA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>accumb</th>
      <td>0.021405</td>
      <td>0.030332</td>
      <td>0.026231</td>
    </tr>
    <tr>
      <th>pal</th>
      <td>0.02372</td>
      <td>0.026119</td>
      <td>0.025201</td>
    </tr>
    <tr>
      <th>amyg</th>
      <td>0.024351</td>
      <td>0.023221</td>
      <td>0.021141</td>
    </tr>
    <tr>
      <th>hippo</th>
      <td>0.019201</td>
      <td>0.018283</td>
      <td>0.018085</td>
    </tr>
    <tr>
      <th>thal</th>
      <td>0.013892</td>
      <td>0.012533</td>
      <td>0.012153</td>
    </tr>
    <tr>
      <th>put</th>
      <td>0.011277</td>
      <td>0.015345</td>
      <td>0.011658</td>
    </tr>
    <tr>
      <th>caud</th>
      <td>0.012747</td>
      <td>0.011662</td>
      <td>0.012845</td>
    </tr>
    <tr>
      <th>ICV</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>0.027722</td>
    </tr>
    <tr>
      <th>LatVent</th>
      <td>0.01307</td>
      <td>0.011915</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>vent</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>0.012293</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>is_ad</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>is_recurrent</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>is_female</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>Age_of_Onset</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>Normalized_Pretreatment_Severity</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('start time:', results_uid)
print('finish time:', datetime.now().strftime("%Y%m%d-%H%M%S"))
```

    start time: 20240418-181259-postRebuttalLongLong
    finish time: 20240423-075004
