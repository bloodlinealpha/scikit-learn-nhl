import pandas as pd
from model import create_points_model

def points_current_test(drop_columns_1, linear_model, random_forest_model, gradient_boost_model, min_values, max_values, extrapolate=False):
    # test with current season stats for each player
    #import current season csv
    current_season_df = pd.read_csv("csv/Top-100-NHL-20232024-Jan-6-2023.csv")
    #print(current_season_df.head().to_string())

    # one-hot encode positionCode
    current_season_df = pd.get_dummies(current_season_df, columns=['positionCode'])

    # create empty list to save results to a csv file
    results = []

    # create empty list to save the player stats
    player_stats_list = []
    # for each row in the current season df, predict the points
    for index, row in current_season_df.iterrows():
        # if index != 1:
        #     continue
        # get the player name
        player_name = row['skaterFullName']
        # get points for the player
        points = row['points']
        # get games played for the player
        games_played = row['gamesPlayed']
        if extrapolate:
            # extrapolate points
            points = (points / games_played) * 82
            games_played = 82

        # create a dataframe with the player stats
        player_stats = pd.DataFrame(row).transpose()
        
        # drop the columns
        player_stats = player_stats.drop(drop_columns_1, axis=1)
        # drop points
        player_stats = player_stats.drop(['points'], axis=1)

        # Get a list of all columns that were created by one-hot encoding
        encoded_cols = [col for col in player_stats.columns if 'positionCode_' in col]

        #columns to ignore when extrapolating
        extrapolate_ignore_cols = ['shootingPct', 'timeOnIcePerGame']

        # merge the two lists
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

        # print the player stats
        # print(f"Player Stats for {player_name}")
        # print(player_stats.to_string())

        # predict the points
        predicted_points_lr = linear_model.predict(player_stats)
        predicted_points_rf = random_forest_model.predict(player_stats)
        predicted_points_gb = gradient_boost_model.predict(player_stats)

        # Denormalize predictions
        predicted_points_lr = (predicted_points_lr * (max_values['points'] - min_values['points'])) + min_values['points']
        predicted_points_rf = (predicted_points_rf * (max_values['points'] - min_values['points'])) + min_values['points']
        predicted_points_gb = (predicted_points_gb * (max_values['points'] - min_values['points'])) + min_values['points']

        # append the results to the list
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
        player_stats_df.to_excel("output/test-1.xlsx")
    else:
        player_stats_df.to_excel("output/test-2.xlsx")

    return results_df

def run_tests():
    # create the model
    drop_columns_1, linear_model, random_forest_model, gradient_boost_model, min_values, max_values = create_points_model()

    # run the current season test
    print("Current Season Test")
    results_df_current = points_current_test(drop_columns_1, linear_model, random_forest_model, gradient_boost_model, min_values, max_values)
    print('*******************************************************************************************************')

    print("Current Season Test Extrapolated")
    results_df_extrapolated = points_current_test(drop_columns_1, linear_model, random_forest_model, gradient_boost_model, min_values, max_values, extrapolate=True)
    print('*******************************************************************************************************')

    # export the results to an excel file
    results_df_current.to_excel("output/test-3.xlsx")
    results_df_extrapolated.to_excel("output/test-4.xlsx")

    return results_df_current, results_df_extrapolated

if __name__ == "__main__":
    run_tests()



