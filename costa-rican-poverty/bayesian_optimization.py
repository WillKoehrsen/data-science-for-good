# numpy and pandas for data manipulation
import numpy as np
import pandas as pd 

# Clearing up memory
import gc

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer

# Modeling
import lightgbm as lgb

# Evaluation of the model
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score

from hyperopt import hp
from hyperopt import tpe
from hyperopt import Trials
from hyperopt import fmin

import csv
from hyperopt import STATUS_OK
from timeit import default_timer as timer

import json

import ast

import json

train = pd.read_csv('data/train.csv')
train_valid_labels = train.loc[train['parentesco1'] == 1, ['idhogar', 'Target']].copy()

feature_matrix = pd.read_csv('data/ft_2000.csv', low_memory = False)
feature_matrix.shape

feature_matrix.drop(columns = 'idhogar', inplace = True)

for col in feature_matrix.select_dtypes('object'):
	if col != 'idhogar':
		feature_matrix[col] = feature_matrix[col].astype(np.float32)

missing_threshold = 0.95
correlation_threshold = 0.95

feature_matrix = feature_matrix.replace({np.inf: np.nan, -np.inf:np.nan})

# One hot encoding (if necessary)
feature_matrix = pd.get_dummies(feature_matrix)
n_features_start = feature_matrix.shape[1]
print('Original shape: ', feature_matrix.shape)

# Find missing and percentage
missing = pd.DataFrame(feature_matrix.isnull().sum())
missing['fraction'] = missing[0] / feature_matrix.shape[0]
missing.sort_values('fraction', ascending = False, inplace = True)

# Missing above threshold
missing_cols = list(missing[missing['fraction'] > missing_threshold].index)
n_missing_cols = len(missing_cols)

# Remove missing columns
feature_matrix = feature_matrix[[x for x in feature_matrix if x not in missing_cols]]
print('{} missing columns with threshold: {}.'.format(n_missing_cols, missing_threshold))

# Zero variance
unique_counts = pd.DataFrame(feature_matrix.nunique()).sort_values(0, ascending = True)
zero_variance_cols = list(unique_counts[unique_counts[0] == 1].index)
n_zero_variance_cols = len(zero_variance_cols)

# Remove zero variance columns
feature_matrix = feature_matrix[[x for x in feature_matrix if x not in zero_variance_cols]]
print('{} zero variance columns.'.format(n_zero_variance_cols))

# Correlations
corr_matrix = feature_matrix.corr()

# Extract the upper triangle of the correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(np.bool))

# Select the features with correlations above the threshold
# Need to use the absolute value
to_drop = [column for column in upper.columns if any(upper[column].abs() > correlation_threshold)]

n_collinear = len(to_drop)

feature_matrix = feature_matrix[[x for x in feature_matrix if x not in to_drop]]
print('{} collinear columns removed with correlation above {}.'.format(n_collinear,  correlation_threshold))

total_removed = n_missing_cols + n_zero_variance_cols + n_collinear

print('Total columns removed: ', total_removed)
print('Shape after feature selection: {}.'.format(feature_matrix.shape))

# Remove columns derived from the Target
drop_cols = []
for col in feature_matrix:
    if col == 'Target':
        pass
    else:
        if 'Target' in col:
            drop_cols.append(col)

feature_matrix = feature_matrix[[x for x in feature_matrix if x not in drop_cols]]    

# Extract out training and testing data
train = feature_matrix[feature_matrix['Target'].notnull()]
test = feature_matrix[feature_matrix['Target'].isnull()]

train_labels = np.array(train.pop('Target')).reshape((-1, ))

def macro_f1_score(labels, predictions):
    # Reshape the predictions as needed
    predictions = predictions.reshape(len(np.unique(labels)), -1 ).argmax(axis = 0)
    
    metric_value = f1_score(labels, predictions, average = 'macro')
    
    # Return is name, value, is_higher_better
    return 'macro_f1', metric_value, True

def model_valid(model, features, labels, nfolds = 5, return_preds = False):
    """Model using the GBM and cross validation.
       Trains with early stopping on each fold.
       Hyperparameters probably need to be tuned."""

    # Using stratified kfold cross validation
    strkfold = StratifiedKFold(n_splits = nfolds, shuffle = True)
    
    # Convert to arrays for indexing
    features = np.array(features)
    labels = np.array(labels).reshape((-1 ))
    
    valid_scores = []
    best_estimators = []
    # Iterate through the folds
    for i, (train_indices, valid_indices) in enumerate(strkfold.split(features, labels)):
        # Training and validation data
        X_train = features[train_indices]
        X_valid = features[valid_indices]
        y_train = labels[train_indices]
        y_valid = labels[valid_indices]
        
        # Train with early stopping
        model.fit(X_train, y_train, early_stopping_rounds = 100, 
                  eval_metric = macro_f1_score,
                  eval_set = [(X_train, y_train), (X_valid, y_valid)],
                  eval_names = ['train', 'valid'],
                  verbose = 200)
        
        # Record the validation fold score
        valid_scores.append(model.best_score_['valid']['macro_f1'])
        best_estimators.append(model.best_iteration_)
        
    best_estimators = np.array(best_estimators)
    valid_scores = np.array(valid_scores)
    return valid_scores, best_estimators

def objective(hyperparameters):
    """Objective function for Gradient Boosting Machine Hyperparameter Optimization.
       Writes a new line to `outfile` on every iteration"""
    
    # Keep track of evals
    global ITERATION
    
    ITERATION += 1
    
    # Using early stopping to find number of trees trained
    if 'n_estimators' in hyperparameters:
        del hyperparameters['n_estimators']
    
    # Retrieve the subsample
    subsample = hyperparameters['boosting_type'].get('subsample', 1.0)
    
    # Extract the boosting type and subsample to top level keys
    hyperparameters['boosting_type'] = hyperparameters['boosting_type']['boosting_type']
    hyperparameters['subsample'] = subsample
    
    # Make sure parameters that need to be integers are integers
    for parameter_name in ['num_leaves', 'subsample_for_bin', 'min_child_samples']:
        hyperparameters[parameter_name] = int(hyperparameters[parameter_name])

    # Will be selected with early stopping
    hyperparameters['n_estimators'] = 10000
    hyperparameters['class_weight'] = 'balanced'
    # hyperparameters['device'] = 'gpu'
    model = lgb.LGBMClassifier(**hyperparameters)
    
    start = timer()
    valid_scores, best_estimators = model_valid(model, train, train_labels)
    run_time = timer() - start
    
    # Extract the best score
    best_score = valid_scores.mean()
    best_std = valid_scores.std()
    
    # Loss must be minimized
    loss = 1 - best_score
    
    # Boosting rounds that returned the highest cv score
    n_estimators = int(best_estimators.mean())
    
    # Add the number of estimators to the hyperparameters
    hyperparameters['n_estimators'] = n_estimators

    # Write to the csv file ('a' means append)
    of_connection = open(OUT_FILE, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([loss, hyperparameters, ITERATION, run_time, best_score, best_std])
    of_connection.close()

    # Dictionary with information for evaluation
    return {'loss': loss, 'hyperparameters': hyperparameters, 'iteration': ITERATION,
            'train_time': run_time, 'status': STATUS_OK}

"""
Search Domain
"""

# Define the search space
space = {
    'boosting_type': hp.choice('boosting_type', 
                                            [{'boosting_type': 'gbdt', 'subsample': hp.uniform('gdbt_subsample', 0.5, 1)}, 
                                             {'boosting_type': 'dart', 'subsample': hp.uniform('dart_subsample', 0.5, 1)},
                                             {'boosting_type': 'goss', 'subsample': 1.0}]),
    'num_leaves': hp.quniform('num_leaves', 5, 50, 1),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.015), np.log(0.5)),
    'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),
    'min_child_samples': hp.quniform('min_child_samples', 5, 60, 3),
    'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
    'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
    'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0)
}

# Record results
trials = Trials()

# Create a file and open a connection
OUT_FILE = 'optimization/optimization1.csv'
of_connection = open(OUT_FILE, 'w')
writer = csv.writer(of_connection)

MAX_EVALS = 1000
N_FOLDS = 5
ITERATION = 0

# Write column names
headers = ['loss', 'hyperparameters', 'iteration', 'runtime', 'score', 'std']
writer.writerow(headers)
of_connection.close()

print("Running Optimization for {} Trials.".format(MAX_EVALS))

# Run optimization
best = fmin(fn = objective, space = space, algo = tpe.suggest, trials = trials,
            max_evals = MAX_EVALS)

import json

# Save the trial results
with open('optimization/trials1.json', 'w') as f:
    f.write(json.dumps(trials))

print(best)

results = pd.read_csv(OUT_FILE, index_col = 0)
results = results.sort_values('score', ascending = False)
results.to_csv('optimization/sorted_optimization1.csv', index = False)