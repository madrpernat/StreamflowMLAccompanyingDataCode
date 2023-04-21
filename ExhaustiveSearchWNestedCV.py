import itertools
import csv
import json
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor


def main():
    # only putting one configuration file name in this list, but the code is set up so that you can run as many config
    # files as you want. See the .json files in the repository to see what information they include.
    config_list = ['Animas_River_config_svr.json']

    # Scoring object used later for hyperparameter tuning
    nse_scorer = make_scorer(calc_nse, greater_is_better=True)

    for config_file in config_list:
        # loading the config (.json) file
        with open(config_file) as file:
            configs = json.load(file)

        # reading in this config's respective data file. See the .csv files in the repository to see what a data file
        # includes.
        df = pd.read_csv(configs['data_file'])

        # For each 'regressor'/'combo' pair (where 'combo' is feature combination), this dictionary stores all of its
        # out-of-sample performance scores across 3 metrics (NSE, RRMSE, R-squared).
        results = {'regressor': [], 'number_of_features': [], 'combo': [], 'test_nse_scores': [], 'avg_test_nse': [],
                   'test_rrmse_scores': [], 'avg_test_rrmse': [], 'test_r2_scores': [], 'avg_test_r2': []}

        # Cycle through each model type ('regressor')
        for i in range(len(configs['regressors'])):
            regressor_name = configs['regressors'][i]

            # set up pipeline object and hyperparameter grid for this model type
            pipe, param_grid = set_pipeline(regressor_name)

            # feature_list contains the 10 features previously identified by forward feature selection
            feature_list = configs['features'][i]

            X = df[feature_list]  # features (ie. predictor variables)
            y = df['Streamflow']  # target variable

            # get a list of every possible feature combination
            all_feature_combinations = create_all_combos(feature_list)

            # Iterate through all feature combinations. Use a Nested CV approach for model evaluation: split the entire
            # dataset into 5 folds (n_splits). One is held out as the outer "testing set" while the remaining 4 are
            # used for inner cross validation (using the GridSearchCV object), where optimal hyperparameters are
            # determined. Then, using the optimal hyperparameters, test the model on the testing set. Repeat for the
            # other 4 folds. This results in 5 performance estimates. This procedure is repeated 3 times (n_repeats) to
            # result in 15 performance estimates.
            for feature_combo in all_feature_combinations:
                print(feature_combo)
                outer_X = X.loc[:, feature_combo]

                # repeated k-fold instance
                rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)

                # create list empty lists to fill with the 15 performance estimates
                test_nse_score_list = []
                test_rrmse_score_list = []
                test_r2_score_list = []

                # looping through the 15 different outer/inner splits
                for p, (train_idx, test_idx) in enumerate(rkf.split(outer_X)):
                    inner_X_train, inner_X_test = outer_X.loc[train_idx, :], outer_X.loc[test_idx, :]
                    inner_y_train, inner_y_test = y.loc[train_idx], y.loc[test_idx]

                    grid = GridSearchCV(pipe, param_grid, scoring=nse_scorer, n_jobs=-1)  # inner CV for hp tuning

                    # Using best hyperparameters, retrain model on full inner dataset
                    grid.fit(inner_X_train, inner_y_train)
                    # and then use it to predict the testing set
                    y_preds = grid.predict(inner_X_test)

                    # calculate performance metrics (ie. predicted vs. observed)
                    test_nse_score = calc_nse(inner_y_test, y_preds)
                    test_rrmse_score, test_r2_score = calc_other_metrics(inner_y_test, y_preds)

                    test_nse_score_list.append(test_nse_score)
                    test_rrmse_score_list.append(test_rrmse_score)
                    test_r2_score_list.append(test_r2_score)

                # input regressor name, feature combination, and performance scores into the results dictionary
                add_info(results, regressor_name, feature_combo)
                add_scores(results, "nse", test_nse_score_list)
                add_scores(results, "rrmse", test_rrmse_score_list)
                add_scores(results, "r2", test_r2_score_list)

        # store results dictionary into pickle file so you can use the results later if you want
        with open(configs['basin_name'] + "_results.pkl", 'wb') as pickle_file:
            pickle.dump(results, pickle_file)

        # and then also write the result dictionary out into a csv file easy view of results and/or manipulation in
        # excel or other software
        with open(configs['basin_name'] + "_results.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['regressor', 'number_of_features', 'combo', 'test_nse_scores', 'avg_test_nse',
                             'test_rrmse_scores', 'avg_test_rrmse', 'test_r2_scores', 'avg_test_r2'])

            for k in range(len(results['combo'])):
                writer.writerow([results['regressor'][k], results['number_of_features'][k], results['combo'][k],
                                 results['test_nse_scores'][k], results['avg_test_nse'][k],
                                 results['test_rrmse_scores'][k], results['avg_test_rrmse'][k],
                                 results['test_r2_scores'][k], results['avg_test_r2'][k]])


def set_pipeline(regressor_name):
    transformer = StandardScaler()

    if regressor_name == 'multiple_linear_regression':
        pipe = Pipeline([("transformer", transformer), ('regressor', LinearRegression())])
        param_grid = {"regressor__fit_intercept": [True, False]}

    elif regressor_name == "principal_component_regression":
        pipe = Pipeline(
            [('transformer', transformer), ('pca', PCA(n_components=0.99)), ('regressor', LinearRegression())])
        param_grid = {"regressor__fit_intercept": [True, False]}

    elif regressor_name == 'random_forest':
        pipe = Pipeline([('transformer', transformer), ('regressor', RandomForestRegressor(random_state=1))])
        param_grid = {"regressor__max_depth": [2, 5, 10, None],
                      "regressor__max_features": [1, 0.6]}

    elif regressor_name == 'extremely_randomized_trees':
        pipe = Pipeline([('transformer', transformer), ('regressor', ExtraTreesRegressor(random_state=1))])
        param_grid = {"regressor__max_depth": [2, 5, 10, None],
                      "regressor__max_features": [1, 0.6]}

    elif regressor_name == 'support_vector_regression':
        pipe = Pipeline([('transformer', transformer), ('regressor', SVR())])
        param_grid = {'regressor__C': [0.1, 1, 10, 100],
                      'regressor__gamma': [1, 0.1, 0.01, 0.001],
                      'regressor__epsilon': [0.01, 0.1, 1],
                      'regressor__kernel': ['linear', 'poly', 'rbf']}

    return pipe, param_grid


# Given a list of features, this function creates a list of all possible combinations of all the features in list
def create_all_combos(features_list: list):
    combos = []
    for L in range(len(features_list) + 1):
        for subset in itertools.combinations(features_list, L):
            combos.append(list(subset))
    return combos[1:]


# Calculates the Nash_Sutcliffe Efficiency given observed and predicted streamflow values
def calc_nse(y_true, y_pred):
    y_true = list(y_true)
    mean_observed = np.mean(y_true)
    length = len(y_true)
    numerator = 0
    denominator = 0
    for i in range(length):
        numerator += (y_true[i] - y_pred[i]) ** 2
        denominator += (y_true[i] - mean_observed) ** 2
    nse = 1 - (numerator / denominator)
    return nse


def calc_other_metrics(y_true, y_pred):
    y_true = list(y_true)
    length = len(y_true)
    mean_true = np.mean(y_true)

    # RRMSE calculation
    summation0 = 0
    for q in range(length):
        summation0 += (y_true[q] - y_pred[q]) ** 2
    rrmse = (np.sqrt((1 / length) * summation0)) / mean_true

    # R-squared calculation
    mean_pred = np.mean(y_pred)
    summation1 = 0
    summation2 = 0
    summation3 = 0
    for q in range(length):
        summation1 += (y_true[q] - mean_true) * (y_pred[q] - mean_pred)
        summation2 += (y_true[q] - mean_true) ** 2
        summation3 += (y_pred[q] - mean_pred) ** 2
    r2 = (summation1 / (np.sqrt(summation2) * np.sqrt(summation3))) ** 2

    return rrmse, r2


def add_info(results, regressor_name, feature_combo):
    results['regressor'].append(regressor_name)
    results['number_of_features'].append(len(feature_combo))
    results['combo'].append(feature_combo)

def add_scores(results, metric, test_score_list):
    results["test_" + metric + "_scores"].append(test_score_list)
    results["avg_test_" + metric].append(np.mean(test_score_list))

if __name__ == "__main__":
    main()
