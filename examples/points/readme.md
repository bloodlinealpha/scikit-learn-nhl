# Points Regression - Top 100 NHL Players 2023/24 Season

This example highlights a simple example using the NHL API, [Scikit-Learn](https://scikit-learn.org/stable/), and Python. Using the top 100 NHL players by points (as of Jan 6, 2023) and Scikit-Learn I created 3 regression models (Linear, Random Forest, and Gradient Boost) to predict player point totals. 

The models are trained on data from the past 5 seasons (not including the current season = 2023/2024). This allows us to:
- test the models using the current season stats (as of Jan 6, 2023)
- extrapolate current season stats to 82 games and predict end of the season point totals

To visualize the data I created a graph and datatable using [Plotly](https://dash.plotly.com/).
## Visualizer

### Live Example
See it live and try the interactive visualizer at: [https://bloodlinealpha.com/nhl/points-prediction/](https://bloodlinealpha.com/nhl/points-prediction/)

### Run Locally
1.) Create and Activate the Virtual Environment
- Open a terminal and navigate to the points directory:
    ```shell
    cd .\examples\points\
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
3.) Run the visualizer
- Ensure you are in the [points](examples\points) folder
    ```shell
    python visualizer.py or python3 visualizer.py 
- Open your browser and navigate to: [http://127.0.0.1:8050/](http://127.0.0.1:8050/)


## Steps Taken
1.) I created the [Top-100-NHL-20232024-Jan-6-2023.json](Top-100-NHL-20232024-Jan-6-2023.json) using the below link...
- [Top 100 Players 2023/2024](https://api.nhle.com/stats/rest/en/skater/summary?isAggregate=false&isGame=false&sort=%5B%7B%22property%22:%22points%22,%22direction%22:%22DESC%22%7D,%7B%22property%22:%22playerId%22,%22direction%22:%22ASC%22%7D%5D&start=0&limit=100&factCayenneExp=gamesPlayed%3E=1&cayenneExp=gameTypeId=2%20and%20seasonId%3C=20232024%20and%20seasonId%3E=20232024)


2.) Created and Ran [init.py](init.py) which:
- Creates the CSV of the json file: ([Top-100-NHL-20232024-Jan-6-2023.csv](csv\Top-100-NHL-20232024-Jan-6-2023.csv))
- Queries the stats from the past 5 seasons for each current top 100 player
- Creates the [Top-100-NHL-5-year-Stats.csv](csv\Top-100-NHL-5-year-Stats.csv).

3.) 
Created and ran [model.py](model.py), which creates a:
- **Default model** - default settings
- **Tuned model** - tuned parameters using GridSearchCV
    - It finds the best combination of hyperparameters for a machine learning model.

**For each model I:**
- Imported the Top-100-NHL-5-year-Stats.csv
- Dropped columns that are not needed and those that cause data leakage
- One-hot encodes the positionCode
- Split the data into train and test sets
- Normalized the data based on max and min values
- Created 3 prediction models (*default* and *tuned* for each):
    - 1.) **Linear Regression Model & Linear Regression Model - Tuned**
    - 2.) **Random Forest Regressor Model & Random Forest Regressor Model - Tuned**
    - 3.) **Gradient Boosting Regressor Model & Gradient Boosting Regressor Model - Tuned**
- Ran the test set, which prints and saves the metrics for each model to the [console](output/console) and [console_tuned](output_tuned/console/) folder
- Saved the prediction model and normalization values for each model to a .joblib file in the respective output folder (allows it to be used in the future without retraining)

4.) Created and ran [test.py](test.py) which:
- Takes in the Top-100-NHL-20232024-Jan-6-2023.csv
- One-hot encodes the positionCode
- Loops through each row (Player's stats)
- Applies the same data manipulation as the model training (normalization, encoding)
- Combines all players stats and saves **player_stats.xlsx** to [output](output)
- Predicts the points for each player, combines all players and saves **results_current.xlsx** to [output](output) and **results_current_tuned.xlsx** to [output_tuned](output_tuned)
- Extrapolates the necessary data to a full 82 game season
- Combines all players and saves **player_stats_extrapolated.xlsx** to [output](output)
- Predicts the points for each player, combines all players and saves **results_extrapolated.xlsx** to [output](output) and **results_extrapolated_tuned.xlsx** to [output_tuned](output_tuned)

5.) Created and ran [visualizer.py](visualizer.py) which:
- Uses [Plotly](https://dash.plotly.com/) to create a simple graph and data table
- Uses the [output](output) and [output_tuned](output_tuned) excel files directly to load the results data.
- Steps:
    - 1.) Select a player from the drop down list
    - 2.) Select a model (default or tuned)
    - 3.) Toggle the lines from the legend
    - 4.) Click directly on the graph to view predictions for each line
    - 5.) Reset The legend if necessary
    - 6.) Scroll down to see the tabular data for each line on the graph


## NHL API
1.) API Calls
- [Top 100 Players 2023/2024](https://api.nhle.com/stats/rest/en/skater/summary?isAggregate=false&isGame=false&sort=%5B%7B%22property%22:%22points%22,%22direction%22:%22DESC%22%7D,%7B%22property%22:%22playerId%22,%22direction%22:%22ASC%22%7D%5D&start=0&limit=100&factCayenneExp=gamesPlayed%3E=1&cayenneExp=gameTypeId=2%20and%20seasonId%3C=20232024%20and%20seasonId%3E=20232024)
- [Player Stats for Past Seasons](https://api.nhle.com/stats/rest/en/skater/summary?isAggregate=false&isGame=false&sort=%5B%7B%22property%22:%22points%22,%22direction%22:%22DESC%22%7D,%7B%22property%22:%22playerId%22,%22direction%22:%22ASC%22%7D%5D&start=0&limit=100&factCayenneExp=gamesPlayed%3E=1&cayenneExp=gameTypeId=2%20and%20seasonId%3C=20232024%20and%20seasonId-2%3E=20182019%20and%20playerId=8477492) 
    - where the playerId, seasonId ,and seasonId-2 can be updated.