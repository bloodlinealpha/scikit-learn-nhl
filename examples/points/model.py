import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from joblib import dump  

# helper functions
def drop_columns():
    """ Return a list of the columns that are not needed for the model. """
    # remove the columns that are not needed for the model
    drop_columns_1 = ['lastName', 'penaltyMinutes', 'faceoffWinPct', 'playerId', 'seasonId', 'shootsCatches', 'skaterFullName', 'teamAbbrevs']

    #remove the columns related to points - prevents data leakage and overfitting
    drop_columns_2 = ['goals', 'assists', 'evGoals', 'evPoints','pointsPerGame', 'ppGoals', 'ppPoints']

    # merge the two lists
    drop_columns_1.extend(drop_columns_2)

    return drop_columns_1

def load_and_prepare_data():
    """ Load the data from the CSV file and prepare it for the model. """

    # Read the 5 year stats CSV file into a DataFrame
    stats_5_year_df = pd.read_csv("csv/Top-100-NHL-5-year-Stats.csv")
    #print(stats_5_year_df.head())

    # drop the columns that are not needed for the model
    drop_columns_1 = drop_columns()

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

    return features, target, drop_columns_1

def create_test_train_split(features, target):
    """ Create a test and training split of the data. Normalize the data. """

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

    return X_train, X_test, y_train, y_test, min_values, max_values

def test_output_points_model(model, X_test, y_test, name, folder):
    """ Test the model and output the metrics to console and to a txt file."""

    # make predictions using the testing set
    linear_predictions = model.predict(X_test)

    # get the metrics for the model
    MAE = metrics.mean_absolute_error(y_test, linear_predictions)
    MSE = metrics.mean_squared_error(y_test, linear_predictions)
    RMSE = np.sqrt(metrics.mean_squared_error(y_test, linear_predictions))
    R2 = metrics.r2_score(y_test, linear_predictions)

    # print the metrics for the model
    print(name)
    print("Mean Absolute Error (MAE):", MAE)
    print("Mean Squared Error (MSE):", MSE)
    print("Root Mean Squared Error (RMSE):", RMSE)
    print("R2 Score:", R2)
    print("\n")

    feature_importances = False
    if name in ["Linear_Regression_Model", "Linear_Regression_Model_Tuned", ]:
        feature_importances = pd.DataFrame(model.coef_, index = X_test.columns, columns=['importance']).sort_values('importance', ascending=False)
    else:
        feature_importances = pd.DataFrame(model.feature_importances_, index = X_test.columns, columns=['importance']).sort_values('importance', ascending=False)
    print("Feature Importances: ")
    print(feature_importances)

    
    # 
    # overwite and save to a txt file
    with open(f"{folder}/console/{name}_output_results.txt", "w") as f:
        f.write(name + "\n")
        f.write("Mean Absolute Error (MAE): " + str(MAE) + "\n")
        f.write("Mean Squared Error (MSE): " + str(MSE) + "\n")
        f.write("Root Mean Squared Error (RMSE): " + str(RMSE) + "\n")
        f.write("R2 Score: " + str(R2) + "\n")
        if feature_importances is not False:
            f.write("\n")
            f.write("Feature Importances: " + "\n")
            f.write(feature_importances.to_string())
        f.write("\n")
        f.write("\n"+ "Output Date and Time" + "\n")
        f.write(str(pd.Timestamp.now()) + "\n")

def output_tuned_model(linear_model_grid, random_forest_model_grid, gradient_boost_model_grid):
    """ Output the best parameters for each model. Outputs to console and to a txt file."""

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

    # write the best parameters to a single txt file
    with open("output_tuned/console/best_model_parameters.txt", "w") as f:
        f.write("Linear Regression Model" + "\n")
        f.write(str(linear_model_grid.best_params_) + "\n")
        f.write("\n")
        f.write("Random Forest Model" + "\n")
        f.write(str(random_forest_model_grid.best_params_) + "\n")
        f.write("\n")
        f.write("Gradient Boost Model" + "\n")
        f.write(str(gradient_boost_model_grid.best_params_) + "\n")
        f.write("\n")
        f.write("\n"+ "Output Date and Time" + "\n")
        f.write(str(pd.Timestamp.now()) + "\n")

def save_model(model, min_values, max_values, name, folder):
    """ Save the model to a joblib file for later use. Allows for the model to be used again without having to retrain it."""

    # save the model to a joblib file
    dump(model, f"{folder}/joblib/{name}_model.joblib")

    # save the min and max values using joblib
    dump(min_values, f"{folder}/joblib/min_values.joblib")
    dump(max_values, f"{folder}/joblib/max_values.joblib")


# default functions
def create_default_points_model():
    """
    Create a default points model using linear regression, random forest, and gradient boost. Without modifying the model parameters.
    """

    # load and prepare the data
    features, target, drop_columns_1 = load_and_prepare_data()

    # split the data into training and testing sets
    X_train, X_test, y_train, y_test, min_values, max_values = create_test_train_split(features, target)
    
    # create a linear regression model
    linear_model = LinearRegression()
    random_forest_model = RandomForestRegressor(n_estimators=100, max_depth=10)
    gradient_boost_model = GradientBoostingRegressor(n_estimators=100, max_depth=10)

    # fit the model to the training data
    linear_model.fit(X_train, y_train)
    random_forest_model.fit(X_train, y_train)
    gradient_boost_model.fit(X_train, y_train)

    # test each model - outputs to console
    test_output_points_model(linear_model, X_test, y_test, "Linear_Regression_Model", "output")
    test_output_points_model(random_forest_model, X_test, y_test, "Random_Forest_Model", "output")
    test_output_points_model(gradient_boost_model, X_test, y_test, "Gradient_Boost_Model", "output")

    # save the models
    save_model(linear_model, min_values, max_values,"linear_regression_default", "output")
    save_model(random_forest_model, min_values, max_values,"random_forest_default", "output")
    save_model(gradient_boost_model, min_values, max_values, "gradient_boost_default", "output")

    return drop_columns_1, linear_model, random_forest_model, gradient_boost_model, min_values, max_values

def create_tuned_points_model():
    """
    Create a points model using linear regression, random forest, and gradient boost. Tuned using the best parameters found by GridSearchCV.
    """

    # load and prepare the data
    features, target, drop_columns_1 = load_and_prepare_data()

    # split the data into training and testing sets
    X_train, X_test, y_train, y_test, min_values, max_values = create_test_train_split(features, target)

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

    # output the best parameters for each model
    output_tuned_model(linear_model_grid, random_forest_model_grid, gradient_boost_model_grid)

    # get the best models
    linear_model = linear_model_grid.best_estimator_
    random_forest_model = random_forest_model_grid.best_estimator_
    gradient_boost_model = gradient_boost_model_grid.best_estimator_

    # test each model - outputs to console
    test_output_points_model(linear_model, X_test, y_test, "Linear_Regression_Model_Tuned", "output_tuned")
    test_output_points_model(random_forest_model, X_test, y_test, "Random_Forest_Model_Tuned", "output_tuned")
    test_output_points_model(gradient_boost_model, X_test, y_test, "Gradient_Boost_Model_Tuned","output_tuned")

    # save the models
    save_model(linear_model, min_values, max_values, "linear_regression_tuned", "output_tuned")
    save_model(random_forest_model, min_values, max_values, "random_forest_tuned", "output_tuned")
    save_model(gradient_boost_model, min_values, max_values, "gradient_boost_tuned", "output_tuned")

    return drop_columns_1, linear_model, random_forest_model, gradient_boost_model, min_values, max_values

if __name__ == '__main__':
    # uncomment the function you want to run
    # create_default_points_model()
    create_tuned_points_model()