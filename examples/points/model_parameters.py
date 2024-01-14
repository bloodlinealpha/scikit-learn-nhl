import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import metrics 

def create_points_model_with_parameters():
    # Read the 5 year stats CSV file into a DataFrame
    stats_5_year_df = pd.read_csv("csv/Top-100-NHL-5-year-Stats.csv")
    #print(stats_5_year_df.head())

    # remove the columns that are not needed for the model
    drop_columns_1 = ['lastName', 'penaltyMinutes', 'faceoffWinPct', 'playerId', 'seasonId', 'shootsCatches', 'skaterFullName', 'teamAbbrevs']

    #remove the columns related to points - prevents data leakage and overfitting
    drop_columns_2 = ['goals', 'assists', 'evGoals', 'evPoints','pointsPerGame', 'ppGoals', 'ppPoints']

    # merge the two lists
    drop_columns_1.extend(drop_columns_2)

    # drop the columns
    stats_5_year_df_1 = stats_5_year_df.drop(drop_columns_1, axis=1)
    # print(stats_5_year_df.head().to_string())

    # check for missing values
    #print(stats_5_year_df_1.isnull().sum())

    #one-hot encode positionCode
    stats_5_year_df_1 = pd.get_dummies(stats_5_year_df_1, columns=['positionCode'])

    # set features and target
    features = stats_5_year_df_1.drop(['points'], axis=1)
    target = stats_5_year_df_1['points']

    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Get a list of all columns that were created by one-hot encoding
    encoded_cols = [col for col in X_train.columns if 'positionCode_' in col]

    # Calculate min and max values from the training set
    min_values = X_train.min()
    max_values = X_train.max()

    # Normalize the test and training data independently to prevent data leakage and overfitting
    # Normalize training data using min_values and max_values
    X_train = X_train.apply(lambda x: x if x.name in encoded_cols else (x - min_values[x.name]) / (max_values[x.name] - min_values[x.name]))

    # Normalize testing data using min_values and max_values
    X_test = X_test.apply(lambda x: x if x.name in encoded_cols else (x - min_values[x.name]) / (max_values[x.name] - min_values[x.name]))

    # create linear regression model parameters options
    linear_model_parameters = {
        'fit_intercept': [True, False],
    }

    # create random forest model hyperparameter options
    random_forest_model_parameters = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }

    # create gradient boost model hyperparameter options
    gradient_boost_model_parameters = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5],
        'min_samples_split': [2, 4, 6],
        'min_samples_leaf': [1, 2, 3],
        'subsample': [0.8, 0.9, 1.0],
        'max_features': ['sqrt', 'log2', None]
    }

    #initialize the GridSearchCV objects
    linear_model_grid = GridSearchCV(LinearRegression(), linear_model_parameters, cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
    random_forest_model_grid = GridSearchCV(RandomForestRegressor(), random_forest_model_parameters, cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
    gradient_boost_model_grid = GridSearchCV(GradientBoostingRegressor(), gradient_boost_model_parameters, cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)

    print("fitting models...")

    # fit the model to the training data
    linear_model_grid.fit(X_train, y_train)
    print("Linear Regression Model Complete!")

    random_forest_model_grid.fit(X_train, y_train)
    print("Random Forest Model Complete!")

    gradient_boost_model_grid.fit(X_train, y_train)
    print("Gradient Boost Model Complete!")

    # get the best models
    linear_model = linear_model_grid.best_estimator_
    random_forest_model = random_forest_model_grid.best_estimator_
    gradient_boost_model = gradient_boost_model_grid.best_estimator_

    # print the best parameters for each model
    print("Linear Regression Model")
    print(linear_model_grid.best_params_)
    print("\n")
    
    print("Random Forest Model")
    print(random_forest_model_grid.best_params_)
    print("\n")

    print("Gradient Boost Model")
    print(gradient_boost_model_grid.best_params_)
    print("\n")


    # make predictions using the testing set
    linear_predictions = linear_model.predict(X_test)
    random_forest_predictions = random_forest_model.predict(X_test)
    gradient_boost_predictions = gradient_boost_model.predict(X_test)

    # print the metrics for the linear regression model
    print("Linear Regression Model")
    print("Mean Absolute Error (MAE):", metrics.mean_absolute_error(y_test, linear_predictions))
    print("Mean Squared Error (MSE):", metrics.mean_squared_error(y_test, linear_predictions))
    print("Root Mean Squared Error (RMSE):", np.sqrt(metrics.mean_squared_error(y_test, linear_predictions)))
    print("R2 Score:", metrics.r2_score(y_test, linear_predictions))
    print("\n")

    # print the metrics for the random forest model
    print("Random Forest Model")
    print("Mean Absolute Error (MAE):", metrics.mean_absolute_error(y_test, random_forest_predictions))
    print("Mean Squared Error (MSE):", metrics.mean_squared_error(y_test, random_forest_predictions))
    print("Root Mean Squared Error (RMSE):", np.sqrt(metrics.mean_squared_error(y_test, random_forest_predictions)))
    print("R2 Score:", metrics.r2_score(y_test, random_forest_predictions))
    print("\n")

    # print the metrics for the gradient boost model
    print("Gradient Boost Model")
    print("Mean Absolute Error (MAE):", metrics.mean_absolute_error(y_test, gradient_boost_predictions))
    print("Mean Squared Error (MSE):", metrics.mean_squared_error(y_test, gradient_boost_predictions))
    print("Root Mean Squared Error (RMSE):", np.sqrt(metrics.mean_squared_error(y_test, gradient_boost_predictions)))
    print("R2 Score:", metrics.r2_score(y_test, gradient_boost_predictions))
    print("\n")

    return drop_columns_1, linear_model, random_forest_model, gradient_boost_model, min_values, max_values


if __name__ == '__main__':
    create_points_model_with_parameters()