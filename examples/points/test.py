import pandas as pd
from joblib import load
from model import drop_columns

def points_current_test(drop_columns_1, linear_model, random_forest_model, gradient_boost_model, min_values, max_values, extrapolate=False):
    """Create predictions for the current season data. Returns a dataframe with the results."""    

    # load the current season points csv
    current_season_df = pd.read_csv("csv/Top-100-NHL-20232024-Jan-6-2023.csv")
    # print(current_season_df.head().to_string())

    # one-hot encode positionCode for all rows
    current_season_df = pd.get_dummies(current_season_df, columns=['positionCode'])

    # create empty list to save results to a csv file
    results = []

    # create empty list to save the player stats
    player_stats_list = []

    # loop through each row and predict the points for the player
    for index, row in current_season_df.iterrows():
        # get the player name
        player_name = row['skaterFullName']

        # get points for the player
        points = row['points']

        # get games played for the player
        games_played = row['gamesPlayed']

        # extrapolate the points if necessary
        if extrapolate:
            # extrapolate the points assuming the player plays 82 games
            points = (points / games_played) * 82
            # update games played to 82
            games_played = 82

        # create a dataframe with the player stats
        player_stats = pd.DataFrame(row).transpose()
        
        # drop the columns that are not needed as they were not used when training the model
        player_stats = player_stats.drop(drop_columns_1, axis=1)

        # drop points as it is the target variable
        player_stats = player_stats.drop(['points'], axis=1)

        # Get a list of all columns that were created by one-hot encoding
        encoded_cols = [col for col in player_stats.columns if 'positionCode_' in col]

        #columns to ignore when extrapolating
        extrapolate_ignore_cols = ['shootingPct', 'timeOnIcePerGame']

        # merge the two lists so that the encoded columns are not extrapolated either
        extrapolate_ignore_cols.extend(encoded_cols)
        if extrapolate:
            # find the columns that are not in the extrapolate_ignore_cols list
            extrapolate_cols = [col for col in player_stats.columns if col not in extrapolate_ignore_cols]
            
            # extrapolate the columns assuming the player plays 82 games
            player_stats[extrapolate_cols] = player_stats[extrapolate_cols].apply(lambda x: (x / x['gamesPlayed']) * (82), axis=1)


        # create a copy of the player stats to save to the player_stats_list
        player_stats_copy = player_stats.copy()

        # add the player name to the start of player_stats_copy and remove the first column
        player_stats_copy.insert(0, 'skaterFullName', player_name)
        player_stats_copy = player_stats_copy.drop(player_stats_copy.columns[1], axis=1)

        # remove encoded columns
        player_stats_copy = player_stats_copy.drop(encoded_cols, axis=1)

        # save the player stats before normalization
        player_stats_list.append(player_stats_copy)
            
        # Normalize data using min_values and max_values
        player_stats = player_stats.apply(lambda x: x if x.name in encoded_cols else (x - min_values[x.name]) / (max_values[x.name] - min_values[x.name]))

        # predict the points
        predicted_points_lr = linear_model.predict(player_stats)
        predicted_points_rf = random_forest_model.predict(player_stats)
        predicted_points_gb = gradient_boost_model.predict(player_stats)

        # add the prediction results to the results list
        results.append([player_name, predicted_points_lr[0], predicted_points_rf[0], predicted_points_gb[0], points, games_played])
        
        # print the results
        # print(f"{player_name} is predicted to score {predicted_points_lr[0]:.0f} points this season. Actual points: {points}")
        # print(f"{player_name} is predicted to score {predicted_points_rf[0]:.0f} points this season. Actual points: {points}")
        # print(f"{player_name} is predicted to score {predicted_points_gb[0]:.0f} points this season. Actual points: {points}")
        # print("\n")

    # create a dataframe with the results
    results_df = pd.DataFrame(results, columns=['Player Name', 'Predicted Points (LR)', 'Predicted Points (RF)', 'Predicted Points (GB)', 'Actual Points', 'Games Played'])
    print(results_df.to_string())

    # create a dataframe with the player stats
    player_stats_df = pd.concat(player_stats_list)

    # save the player stats to a excel file
    if extrapolate:
        player_stats_df.to_excel("output/player_stats_extrapolated.xlsx")
    else:
        player_stats_df.to_excel("output/player_stats.xlsx")

    return results_df

# test that creates predictions for the current season point totals
def run_tests_default_models():
    """Run the tests using the default models for the current season. Loads the models from the joblib files and create predictions for the current season data."""

    # get columns to be dropped
    drop_columns_1 = drop_columns()

    # load joblib files for model and min/max values
    linear_model = load('output/joblib/linear_regression_default_model.joblib')
    random_forest_model = load('output/joblib/random_forest_default_model.joblib')
    gradient_boost_model = load('output/joblib/gradient_boost_default_model.joblib')
    min_values = load('output/joblib/min_values.joblib')
    max_values = load('output/joblib/max_values.joblib')

    # run the current season test
    print("Current Season Test")
    results_df_current = points_current_test(drop_columns_1, linear_model, random_forest_model, gradient_boost_model, min_values, max_values)
    print('*******************************************************************************************************')

    print("Current Season Test Extrapolated")
    results_df_extrapolated = points_current_test(drop_columns_1, linear_model, random_forest_model, gradient_boost_model, min_values, max_values, extrapolate=True)
    print('*******************************************************************************************************')

    # export the results to an excel file
    results_df_current.to_excel("output/results_current.xlsx")
    results_df_extrapolated.to_excel("output/results_extrapolated.xlsx")

def run_tests_tuned_models():
    """Run the tests using the tuned models for the current season. Loads the tuned models from the joblib files and create predictions for the current season data."""

    # get columns to be dropped
    drop_columns_1 = drop_columns()

    # load joblib files for model and min/max values
    linear_model = load('output_tuned/joblib/linear_regression_tuned_model.joblib')
    random_forest_model = load('output_tuned/joblib/random_forest_tuned_model.joblib')
    gradient_boost_model = load('output_tuned/joblib/gradient_boost_tuned_model.joblib')
    min_values = load('output_tuned/joblib/min_values.joblib')
    max_values = load('output_tuned/joblib/max_values.joblib')


    # run the current season test
    print("Tuned - Current Season Test")
    results_df_current = points_current_test(drop_columns_1, linear_model, random_forest_model, gradient_boost_model, min_values, max_values)
    print('*******************************************************************************************************')

    print("Tuned - Current Season Test Extrapolated")
    results_df_extrapolated = points_current_test(drop_columns_1, linear_model, random_forest_model, gradient_boost_model, min_values, max_values, extrapolate=True)
    print('*******************************************************************************************************')

    # export the results to an excel file
    results_df_current.to_excel("output_tuned/results_current_tuned.xlsx")
    results_df_extrapolated.to_excel("output_tuned/results_extrapolated_tuned.xlsx")

if __name__ == "__main__":
    run_tests_default_models()
    run_tests_tuned_models()



