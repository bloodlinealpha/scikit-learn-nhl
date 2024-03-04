import pandas as pd
from joblib import load
from model import create_features


# create the current season features
def create_current_season_features():
    # load the current season schedule for remaining games
    current_schedule = pd.read_csv('csv/NHL_teams_schedule_20232024_2024-02-26.csv')

    #print(current_schedule.head())

    # load the current season data and created features
    current_season_data, team_data_cleaned = create_features('NHL_teams_historical_stats_20232024.csv')

    # save to csv
    #current_season_data.to_csv('csv/test/current_season_features.csv', index=False)

    #print(current_season_data.head())

    # find all numeric columns in team_data_cleaned
    # these columns will be used as an average value for predictions, ideally these would be updated after each game or predicted as well.
    numeric_columns = ["faceoffWinPct", "penaltyKillPct", "powerPlayPct", "shotsAgainstPerGame", "shotsForPerGame", "winStreak" ]

    # find the average of each numeric column of each fullTeamName
    current_team_averages = team_data_cleaned.groupby(['teamId', 'teamFullName'], as_index=False )[numeric_columns].mean()

    # add rows for homeRoad
    current_schedule['homeRoad'] = "H"

    # duplicate all rows and change homeRoad to "R" and swap home_id and away_id and homeTeam and awayTeam
    current_schedule_road = current_schedule.copy()
    current_schedule_road['homeRoad'] = "R"
    current_schedule_road = current_schedule_road.rename(columns={'home_id': 'away_id', 'away_id': 'home_id', 'homeTeam': 'awayTeam', 'awayTeam': 'homeTeam'})

    # append the two dataframes
    current_schedule = current_schedule._append(current_schedule_road)

    # update column names to teamId
    current_schedule = current_schedule.rename(columns={'home_id': 'teamId', "away_id": "opponentTeamId", "awayTeam": "opponentTeamAbbrev"})

    # create new feature column of days since last game for each team
    current_schedule['gameDate'] = pd.to_datetime(current_schedule['gameDate'])
    current_schedule = current_schedule.sort_values(by=['teamId', 'gameDate'])
    current_schedule['daysSinceLastGame'] = current_schedule.groupby('teamId')['gameDate'].diff().dt.days
    current_schedule['daysSinceLastGame'] = current_schedule['daysSinceLastGame'].fillna(0).astype(int)

    # update the homeTeam column to teamFullName
    current_schedule = current_schedule.rename(columns={'homeTeam': 'teamFullName'})

    # create a dictionary from current_team_averages
    team_name_dict = current_team_averages.set_index('teamId')['teamFullName'].to_dict()

    # replace the homeTeam values
    current_schedule['teamFullName'] = current_schedule['teamId'].map(team_name_dict)

    # remove the teamId column so it is not duplicated
    current_team_averages = current_team_averages.drop(columns='teamId')

    # merge the current_schedule with current_team_averages
    current_schedule = current_schedule.merge(current_team_averages, on='teamFullName')

    # get the count of gamesPlayed for each team from team_data_cleaned
    games_played = team_data_cleaned.groupby('teamFullName')['gamesPlayed'].max().reset_index()

    # add gamesPlayed column to current_schedule
    current_schedule["gamesPlayed"] = current_schedule['teamFullName'].map(games_played.set_index('teamFullName')['gamesPlayed'])

    # Create a new column 'increment' with all values set to 1
    current_schedule['increment'] = 1

    # Sort by 'gameDate'
    current_schedule = current_schedule.sort_values('gameDate')

    # Group by 'teamFullName' and increment 'gamesPlayed' for each subsequent date
    current_schedule['gamesPlayed'] = current_schedule.groupby('teamFullName').apply(lambda x: x['gamesPlayed'] + x['increment'].cumsum(), include_groups=False).reset_index(level=0, drop=True)

    # Drop the 'increment' column as it's no longer needed
    current_schedule = current_schedule.drop(columns=['increment'])

    current_schedule_cleaned = current_schedule.copy()

    # save the DataFrame to a CSV file
    current_schedule_cleaned.to_csv('csv/test/current_schedule_cleaned.csv', index=False)

    # encode homeRoad, opponentTeamAbbrev, and teamFullName as integers
    current_schedule = pd.get_dummies(current_schedule, columns=['homeRoad', 'opponentTeamAbbrev', 'teamFullName'], drop_first=True, dtype=int)

    # check for missing values, should be none as we added the missing columns
    # null_counts = current_schedule.isnull().sum()
    # print(null_counts[null_counts > 0])

    # reorder the columns to match the training data (current_season_data)
    # drop 'opponentTeamId' as it is not in the training data
    current_schedule = current_schedule.drop(['opponentTeamId'], axis=1)

    # find missing columns for current schedule data
    missing_columns = current_season_data.columns.difference(current_schedule.columns)

    # add missing columns to current schedule data
    for column in missing_columns:
        current_schedule[column] = 0

    # reorder the columns, columns need to be in the same order as the training data
    current_schedule = current_schedule[current_season_data.columns]

    # save to csv
    current_schedule.to_csv('csv/test/current_schedule_features.csv', index=False)

    return current_schedule, current_schedule_cleaned

# test/predict the current season (up to Feb 26, 2024) data using the default models
def test_current_season():
    """
    Tests the current season (2023/2024) data using the default models. The end date is 2024-02-26.
    Models are trained on data from the 2013/2014 season to the 2022/2023 season.
    This way we can see how the models are performing on the current season data.
    This is best case scenario as we are using actual game stats for the features, as compared to making predictions for future games where we would need to use predicted stats.

    """
    # load the current season data
    current_season_data_original = pd.read_csv('csv/NHL_teams_historical_stats_20232024.csv')

    # create the features for the current season data
    current_season_data, current_season_data_cleaned = create_features('NHL_teams_historical_stats_20232024.csv')

    # load training data features to compare columns
    team_data, team_data_cleaned = create_features('NHL_teams_historical_stats_20132014_to_20222023.csv')

    # find missing columns for current season data
    missing_columns = team_data.columns.difference(current_season_data.columns)

    # add missing columns to current season data
    for column in missing_columns:
        current_season_data[column] = 0

    # reorder the columns
    current_season_data = current_season_data[team_data.columns]

    # load the default models
    rfc_model = load('joblib/rfc_model.joblib')
    svm_model = load('joblib/svm_model.joblib')
    mlp_model = load('joblib/mlp_model.joblib')

    # store the wins
    y = current_season_data['wins']

    results = []
    # make predictions using the default models for each row in the current season data
    for index, row in current_season_data.iterrows():
        # create a dataframe from the row
        current_season = pd.DataFrame(row).transpose()

        # drop the gameDate and wins columns
        current_season = current_season.drop(['gameDate', 'wins'], axis=1)
        
        rfc_prediction = rfc_model.predict(current_season)
        svm_prediction = svm_model.predict(current_season)
        mlp_prediction = mlp_model.predict(current_season)
        print(f"RFC Prediction: {rfc_prediction[0]}, SVM Prediction: {svm_prediction[0]}, MLP Prediction: {mlp_prediction[0]}, Actual Wins: {y[index]} \n\n")

        # save the predictions to a list
        results.append([current_season_data_original.iloc[index]['gameDate'],
                        current_season_data_original.iloc[index]['gameId'],
                        current_season_data_original.iloc[index]['teamFullName'],
                        current_season_data_original.iloc[index]['opponentTeamAbbrev'],
                        y[index],
                        rfc_prediction[0],
                        int(rfc_prediction[0] == y[index]), 
                        svm_prediction[0],
                        int(svm_prediction[0] == y[index]),
                        mlp_prediction[0],
                        int(mlp_prediction[0] == y[index])
                        ])

    # create a dataframe from the results
    results_df = pd.DataFrame(results, columns=['gameDate', 'gameId', 'teamFullName', 'opponentTeamAbbrev', 'actualWins', 'rfcPrediction', 'rfcCorrect', 'svmPrediction', 'svmCorrect', 'mlpPrediction', 'mlpCorrect'])

    # save the results to a csv file
    results_df.to_csv('csv/test/2023_2024_02-26_predictions.csv', index=False)
    
# predict using the current season remaining schedule (Feb 27, 2024 - Apr 19, 2024) using the default models 
def predict_current_season():
    # create the current season features
    current_features, current_features_cleaned = create_current_season_features()

    # load the default models
    rfc_model = load('joblib/rfc_model.joblib')
    svm_model = load('joblib/svm_model.joblib')
    mlp_model = load('joblib/mlp_model.joblib')

    # create dataframe from the current_features_cleaned to append the predictions
    results = current_features_cleaned.copy()

    # drop columns
    results = results.drop(["faceoffWinPct", "penaltyKillPct", "powerPlayPct", "shotsAgainstPerGame", "shotsForPerGame", "winStreak", "daysSinceLastGame", "gamesPlayed" ], axis=1)

    # make predictions using the default models for each row in the current season features
    for index, row in current_features.iterrows():
        # create a dataframe from the row
        current_season_row = pd.DataFrame(row).transpose()

        # drop the gameDate and wins columns
        current_season_row = current_season_row.drop(['gameDate','wins'], axis=1)
        
        # make predictions
        rfc_prediction = rfc_model.predict(current_season_row)
        svm_prediction = svm_model.predict(current_season_row)
        mlp_prediction = mlp_model.predict(current_season_row)

        print(f"RFC Prediction: {rfc_prediction[0]}, SVM Prediction: {svm_prediction[0]}, MLP Prediction: {mlp_prediction[0]} \n\n")

        
        # save the predictions to the results dataframe
        results.at[index, 'rfcPrediction'] = rfc_prediction[0]
        results.at[index, 'svmPrediction'] = svm_prediction[0]
        results.at[index, 'mlpPrediction'] = mlp_prediction[0]

        

    # map the predictions to win or loss
    mapping = {0: 'loss', 1: 'win'}
    results['rfcPrediction'] = results['rfcPrediction'].map(mapping)
    results['svmPrediction'] = results['svmPrediction'].map(mapping)
    results['mlpPrediction'] = results['mlpPrediction'].map(mapping)
    
    # save the results to a csv file
    results.to_csv('csv/test/current_season_predictions.csv', index=False)


if __name__ == "__main__":
    # uncomment the function to run
    # test_current_season()
    predict_current_season()
