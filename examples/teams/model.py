import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from joblib import dump  

"""
- Read the NHL teams historical stats into a DataFrame
- Remove the columns that are not needed
- Create new feature columns for days since last game and win streak
- Save the DataFrame to a CSV file
- Create a RandomForestClassifier model to predict the outcome of a game
- Create a Support Vector Machine model to predict the outcome of a game
- Create a Multi-layer Perceptron model to predict the outcome of a game

"""

# create a function to read the NHL teams historical stats into a DataFrame
def create_features(file_name):
    # create a DataFrame from the CSV file
    team_data = pd.read_csv(f'csv/{file_name}')
    # print(team_data.head())
    # print(team_data.columns)
    # print(team_data.shape)
    # print(team_data.info())
    # print(team_data.describe())

    # remove the columns that are not needed
    features_to_remove = ["gamesPlayed", "gameId", "points", "goalsFor", "goalsAgainstPerGame", "goalsAgainst", "goalsForPerGame", "losses", "otLosses", "penaltyKillNetPct", "pointPct", "powerPlayNetPct", "regulationAndOtWins", "ties", "winsInRegulation", "winsInShootout"]

    # remove the columns that are not needed
    team_data = team_data.drop(features_to_remove, axis=1)

    # update all PHX opponetTeamAbbrev to ARI
    team_data['opponentTeamAbbrev'] = team_data['opponentTeamAbbrev'].replace('PHX', 'ARI')

    # update all Phoenix Coyotes teamFullName to Arizona Coyotes
    team_data['teamFullName'] = team_data['teamFullName'].replace('Phoenix Coyotes', 'Arizona Coyotes')

    # create new feature column of days since last game for each team
    team_data['gameDate'] = pd.to_datetime(team_data['gameDate'])
    team_data = team_data.sort_values(by=['teamId', 'gameDate'])
    team_data['daysSinceLastGame'] = team_data.groupby('teamId')['gameDate'].diff().dt.days
    team_data['daysSinceLastGame'] = team_data['daysSinceLastGame'].fillna(0).astype(int)

    # create a new feature column for the current win streak for each team.
    # sort the data by teamId and gameDate
    team_data = team_data.sort_values(by=['teamId', 'gameDate']) 

    def win_streak(series):
        win_streak = 0
        win_streak_list = []
        for i in range(len(series)):
            if i != 0 and series.iloc[i-1]['wins'] == 1 and series.iloc[i]['daysSinceLastGame'] < 50:
                win_streak += 1
            else:
                win_streak = 0
            win_streak_list.append(win_streak)
        return pd.Series(win_streak_list, index=series.index)

    team_data['winStreak'] = team_data.groupby(['teamId']).apply(win_streak, include_groups=False).reset_index(level=0, drop=True)

    # add feature for games played in the season
    team_data = team_data.sort_values(by=['teamId', 'gameDate']) 

    def games_played(series):
        games_played = 0
        games_played_list = []
        for i in range(len(series)):
            if i != 0 and series.iloc[i]['daysSinceLastGame'] < 50:
                games_played += 1
            else:
                games_played = 0
            games_played_list.append(games_played)
        return pd.Series(games_played_list, index=series.index)

    team_data['gamesPlayed'] = team_data.groupby(['teamId']).apply(games_played, include_groups=False).reset_index(level=0, drop=True)

    # create a copy of the DataFrame to clean the data
    team_data_cleaned = team_data.copy()

    # save the DataFrame to a CSV file
    team_data_cleaned.to_csv(f'csv/{file_name}_cleaned.csv', index=False)

    # encode homeRoad, opponentTeamAbbrev, and teamFullName as integers
    team_data = pd.get_dummies(team_data, columns=['homeRoad', 'opponentTeamAbbrev', 'teamFullName'], drop_first=True, dtype=int)

    # check for missing values
    # null_counts = team_data.isnull().sum()
    # print(null_counts[null_counts > 0])

    # fill in missing values with 0
    team_data = team_data.fillna(0)

    return team_data, team_data_cleaned

# create a function that prints the accuracy of the classifier models
def print_model_accuracy(model_name , model, X_test, y_test):
    # print the score of the model
    print(f"{model_name} model accuracy score: {model.score(X_test, y_test)}")

    # print the confusion matrix of the model
    print(f"{model_name} model confusion matrix: \n{confusion_matrix(y_test, model.predict(X_test))}")

    # print the classification report of the model
    print(f"{model_name} model classification report: \n{classification_report(y_test, model.predict(X_test))}")

    # print the cross validation score of the model
    print(f"{model_name} model cross validation score: {cross_val_score(model, X_test, y_test, cv=5)}")

# create the RandomForestClassifier model
def RandomForestClassifier_model(team_data):
    X = team_data.drop(['gameDate', 'wins'], axis=1)
    y = team_data['wins']

    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # create the RandomForestClassifier model
    rfc_model = RandomForestClassifier(random_state=42)

    # fit the model to the training data
    rfc_model.fit(X_train, y_train)

    # print the accuracy of the model
    print_model_accuracy("RandomForestClassifier", rfc_model, X_test, y_test)
    return rfc_model

# create a SVM model
def SVM_model(team_data):

    # create the features
    X = team_data.drop(['gameDate', 'wins'], axis=1)
    # create the target
    y = team_data['wins']

    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # create the SVM model
    svm_model = SVC(random_state=42)

    # fit the model to the training data
    svm_model.fit(X_train, y_train)

    # print the accuracy of the model
    print_model_accuracy("SVM", svm_model, X_test, y_test)
    return svm_model

# create a Multi-layer Perceptron model
def MLP_model(team_data):

    X = team_data.drop(['gameDate', 'wins'], axis=1)
    y = team_data['wins']

    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # create the MLP model
    mlp_model = MLPClassifier(random_state=42)

    # fit the model to the training data
    mlp_model.fit(X_train, y_train)

    # print the accuracy of the model
    print_model_accuracy("MLP", mlp_model, X_test, y_test)
    return mlp_model

# create the default models
def create_default_models():

    # run create_feature data
    team_data, team_data_cleaned = create_features('NHL_teams_historical_stats_20132014_to_20222023.csv')

    # create the RandomForestClassifier model
    rfc_model = RandomForestClassifier_model(team_data)

    # create the SVM model
    svm_model = SVM_model(team_data)

    # create the MLP model
    mlp_model = MLP_model(team_data)

    # save the models to a file
    dump(rfc_model, 'joblib/rfc_model.joblib')
    dump(svm_model, 'joblib/svm_model.joblib')
    dump(mlp_model, 'joblib/mlp_model.joblib')

    return 


if __name__ == "__main__":
    create_default_models()









