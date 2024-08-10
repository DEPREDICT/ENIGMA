from skopt.space import Real, Integer, Categorical
from copy import deepcopy

# General parameter space
param_space = {
    # KNNImputer parameters
    'pipewrapper__knnimputer__n_neighbors': Integer(1, 20),
    'pipewrapper__knnimputer__weights': Categorical(['uniform', 'distance']),

    # RegressorWrapper parameters
    'regressorwrapper__regressor__n_estimators': Integer(10, 200),
    'regressorwrapper__regressor__learning_rate': Real(0.01, 0.3, 'log-uniform'),

    # SelectFromModel parameters
    'selectfrommodel__linearsvc__C': Real(1e-3, 1e2, 'log-uniform'),
}
svc_space = deepcopy(param_space)
gbc_space = deepcopy(param_space)
log_space = deepcopy(param_space)
rfc_space = deepcopy(param_space)

# SVC parameters
svc_space.update({
    'svc__C': Real(1e-3, 1e2, 'log-uniform'),
    'svc__kernel': Categorical(['linear', 'rbf']),
    'svc__degree': Integer(2, 5),
})

# GradientBoostingClassifier parameters
gbc_space.update({
    'xgbclassifier__n_estimators': Integer(50, 500),  # Number of boosting rounds
    'xgbclassifier__learning_rate': Real(0.001, 0.3, 'log-uniform'),  # Learning rate (eta in XGBoost)
    'xgbclassifier__max_depth': Integer(3, 15),  # Maximum depth of a tree
    'xgbclassifier__subsample': Real(0.5, 1.0),  # Subsample ratio of the training instances
})

# LogisticRegression parameters
log_space.update({
    'logisticregression__C': Real(1e-4, 1e2, 'log-uniform'),  # Regularization strength
    'logisticregression__l1_ratio': Real(0, 1)  # ElasticNet mixing parameter (only used if penalty is 'elasticnet')
})

# RandomForestClassifier parameters
rfc_space.update({
    'randomforestclassifier__n_estimators': Integer(50, 500),  # Number of trees in the forest
    'randomforestclassifier__max_depth': Integer(5, 50),  # Maximum depth of the tree
    'randomforestclassifier__max_features': Categorical(['auto', 'sqrt', 'log2', None]),  # Number of features to consider at each split
})


if __name__ == '__main__':
    '''
    A demo of BayesSearchCV in conjunction with permutation_test_score.
    '''
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from skopt import BayesSearchCV
    from sklearn.datasets import load_iris
    from sklearn.model_selection import permutation_test_score, RepeatedStratifiedKFold
    from utils import Timer

    pipeline = make_pipeline(StandardScaler(), SVC())
    X, y = load_iris(return_X_y=True)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=0)
    opt = BayesSearchCV(estimator=pipeline, search_spaces=svc_space, n_jobs=-1, cv=cv)

    with Timer():
        score, perm_scores, pvalue = permutation_test_score(pipeline, X, y, n_jobs=-1, cv=cv)
    print(f'Classification accuracy is {score:.1%}\n'
          f'Mean permutation accuracy is {perm_scores.mean():.1%}+{perm_scores.std():.1%}\n'
          f'The p-value is {pvalue:.4f}')


