# Scikit Learn - Top 100 NHL Players 2023/24 Season

This example highlights the a simple example using the NHL API, [Scikit-Learn](https://scikit-learn.org/stable/), and Python. Using the top 100 NHL players by points (as of Jan 6, 2023) and Scikit-Learn I created 3 regression model (Linear, Random Forest, and Gradient Boost) to predict player point totals. 

The models are trained on data from the past 5 seasons (not including the current season = 2023/2024). This allows us to test the models using current players current stats (as of Jan 6, 2023), which we can also extrapolate to 82 games, so we can predict end of the season point totals.

To visualize the data I created a graph and datatable using [Plotly](https://dash.plotly.com/).


## Steps Taken
1.) I created the [Top-100-NHL-20232024-Jan-6-2023.json](Top-100-NHL-20232024-Jan-6-2023.json) using the below link...
- [Top 100 Players 2023/2024](https://api.nhle.com/stats/rest/en/skater/summary?isAggregate=false&isGame=false&sort=%5B%7B%22property%22:%22points%22,%22direction%22:%22DESC%22%7D,%7B%22property%22:%22playerId%22,%22direction%22:%22ASC%22%7D%5D&start=0&limit=100&factCayenneExp=gamesPlayed%3E=1&cayenneExp=gameTypeId=2%20and%20seasonId%3C=20232024%20and%20seasonId%3E=20232024)


2.) Created and Ran [init.py](init.py) which:
- Creates the CSV of the json file: ([Top-100-NHL-20232024-Jan-6-2023.csv](csv\Top-100-NHL-20232024-Jan-6-2023.csv))
- Queries the stats from the past 5 seasons for each current top 100 player
- Creates the [Top-100-NHL-5-year-Stats.csv](csv\Top-100-NHL-5-year-Stats.csv).

3.) Created and ran [model.py](examples\points\model.py), which:
- Takes in the Top-100-NHL-5-year-Stats.csv
- Drops columns that are not needed and cause cause leakage
- One-hot encodes the positionCode
- Normalizes the data based on max and min values
- Splits the data into train and test sets
- Creates 3 prediction models:
    - 1.) Linear Regression Model
    - 2.) Random Forest Regressor Model
    - 3.) Gradient Boosting Regressor Model
- Runs the test set and prints the metrics for each model

4.) Created and ran [test.py](examples/points/test.py) which:
- Takes in the Top-100-NHL-20232024-Jan-6-2023.csv
- One-hot encodes the positionCode
- Loops through each row (Player's stats)
- Applies the same data manipulation as the model training
- Save each row to a list, combine all players and output to [player_stats.xlsx](examples\points\output\player_stats.xlsx)
- Predict the Points for each player, combine all players and output to [results_current.xlsx](examples\points\output\results_current.xlsx)
- Extrapolate the necessary data to a full 82 game season
- Save each row to a list, combine all players and output to [player_stats_extrapolated.xlsx](examples\points\output\player_stats_extrapolated.xlsx)
- Predict the Points for each player, combine all players and output to [results_extrapolated.xlsx](examples\points\output\results_extrapolated.xlsx)

4.) Created and ran [visualizer.py](examples\points\visualizer.py) which:
- Uses [Plotly](https://dash.plotly.com/) to create a simple graph and data table
- Steps:
    - 1.) Select a player from the drop down list
    - 2.) Toggle the lines from the legend
    - 3.) Click directly on the graph to view predictions for each line
    - 4.) Reset The legend if necessary
    - 5.) Scroll down to see the tabular data for each line on the graph


## NHL API
1.) API Calls
- [Top 100 Players 2023/2024](https://api.nhle.com/stats/rest/en/skater/summary?isAggregate=false&isGame=false&sort=%5B%7B%22property%22:%22points%22,%22direction%22:%22DESC%22%7D,%7B%22property%22:%22playerId%22,%22direction%22:%22ASC%22%7D%5D&start=0&limit=100&factCayenneExp=gamesPlayed%3E=1&cayenneExp=gameTypeId=2%20and%20seasonId%3C=20232024%20and%20seasonId%3E=20232024)
- [Player Stats for Past Seasons](https://api.nhle.com/stats/rest/en/skater/summary?isAggregate=false&isGame=false&sort=%5B%7B%22property%22:%22points%22,%22direction%22:%22DESC%22%7D,%7B%22property%22:%22playerId%22,%22direction%22:%22ASC%22%7D%5D&start=0&limit=100&factCayenneExp=gamesPlayed%3E=1&cayenneExp=gameTypeId=2%20and%20seasonId%3C=20232024%20and%20seasonId-2%3E=20182019%20and%20playerId=8477492) - where the playerId, seasonId ,and seasonId-2 can be updated.

