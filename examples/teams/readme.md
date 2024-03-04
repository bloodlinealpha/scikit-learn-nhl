# NHL Game Classification - Game Win/Loss for 2023/24 Season

This project aims to classify NHL games as either a win or a loss by team, for the 2023/24 season.


## Introduction
In this project, we will analyze historical NHL game data and build machine learning models (Classification) to predict the outcome of games for the 2023/24 season. The models will classify each game as either a win or a loss based on various features.

- Data is obtained using the NHL API

### Run Locally
1.) Create and Activate the Virtual Environment
- Open a terminal and navigate to the points directory:
    ```shell
    cd .\examples\teams\
- Create the virtual environment
    ```shell
    python -m venv env or python3 -m venv env
- Activate the virtual environment
    ```shell
    cd .\env\Scripts\activate
- You should see a (env) in yout terminal

2.) Install the Packages 
- Navigate back to the root directory
    ```shell
        cd ..
        cd ..
- Install the pip packages from the requirements.txt
    ```shell
    pip install -r requirements.txt


## Steps Taken

1. **Created and Ran [init.py](init.py) which:**
    - Creates the CSV: [NHL_teams_historical_stats_20132014_to_20222023.csv](csv\NHL_teams_historical_stats_20132014_to_20222023.csv) - used for creating and training the model.
    - Creates the CSV: [NHL_teams_historical_stats_20232024.csv](csv\NHL_teams_historical_stats_20232024.csv) - used for testing the model and prediction.
    - Creates the CSV: [NHL_teams_schedule_20232024_2024-02-26.csv](csv\NHL_teams_schedule_20232024_2024-02-26.csv) - used for prediction.

2. **Created and ran [model.py](model.py), which creates three Classification models:**
    - Random Forest (RFC)
    - Support Vector Machines (SVM/SVC)
    - Multilayer Perceptron Classifier (MLP)

    **For each model:**
    - Imported the `NHL_teams_historical_stats_20132014_to_20222023.csv`
    - Dropped columns that are not needed and those that cause data leakage.
    - Updated PHX to ARI as they have changed their team name (special case).
    - Created a new feature column for days since last game for each team.
    - Created a new feature column for the current win streak for each team.
    - Created a new feature column for the games played for each team.
    - Saved the cleaned data to a csv: [NHL_teams_historical_stats_20132014_to_20222023.csv_cleaned.csv](csv\NHL_teams_historical_stats_20132014_to_20222023.csv_cleaned.csv)
    - One-hot encoded `'homeRoad', 'opponentTeamAbbrev', 'teamFullName'`
    - Fill in missing or N/A values with 0.
    - Split the data into train and test sets.
    - Created the 3 classifier models (RFC, SVM, MLP).
    - Ran the test set, which prints model metrics to the console.
    - Saved the prediction model and normalization values for each model to a .joblib file in the respective output folder (allows it to be used in the future without retraining).

4. **Created and ran [test.py](test.py) which:**
    - Tests the model against:
        - current season completed games (up to: feb 26, 2024).
        - remaining games for the current season (feb 27 - Apr 18).
    
5. **Results are output to [csv\test](csv\test)**
    - [csv\test\current_season_predictions.csv](csv\test\current_season_predictions.csv)
        - current season completed game results.
        - this is unseen data for each model.
        - accuracy is very similiar to the training accuracy.
    - [csv\test\2023_2024_02-26_predictions.csv](csv\test\2023_2024_02-26_predictions.csv)
        - predicts game results for each remaining game left in the season
        - feature values are calcualted using averages where necessary


