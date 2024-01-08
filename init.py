import pandas as pd
from io import StringIO
from http_request import get_request

# Read the current top 100 players JSON file into a DataFrame
# this allows us to get the player IDs for the next step
def read_json_to_df(file_name):
    '''
    This function takes a JSON file and reads it into a DataFrame
    :param file_name: name of the JSON file
        :type file_name: str
    :return: DataFrame
    '''
    df = pd.read_json(file_name)
    data_df = pd.json_normalize(df['data'])
    return data_df

# Create a CSV file with the historical stats for the top 100 players
def create_historical_csv(player_ids, start, end, filename):
    """
    Generates a CSV file containing historical stats of players.

    Parameters:
    ----------
    player_ids : list
        List of player IDs to fetch the historical stats for.
    start : str
        Start year for the historical stats in 'YYYYYYYY' format (e.g., '20232024').
    end : str
        End year for the historical stats in 'YYYYYYYY' format (e.g., '20182019').
    filename : str
        Name of the CSV file to be created.

    Returns:
    -------
    None
    """
    # Create an empty list to store the player stats DataFrames
    player_stats_dfs = []

    # Loop through the player IDs
    for count, player_id in enumerate(player_ids, start=1):
        print(f"Getting stats for player {count} of {len(player_ids)}")
        # Get the player stats endpoint for the current player ID
        player_stats_endpoint = f"https://api.nhle.com/stats/rest/en/skater/summary?isAggregate=false&isGame=false&sort=%5B%7B%22property%22:%22points%22,%22direction%22:%22DESC%22%7D,%7B%22property%22:%22playerId%22,%22direction%22:%22ASC%22%7D%5D&start=0&limit=100&factCayenneExp=gamesPlayed%3E=1&cayenneExp=gameTypeId=2%20and%20seasonId%3C={start}%20and%20seasonId%3E={end}%20and%20playerId={player_id}"

        # make the http request
        player_stats_response = get_request(player_stats_endpoint)

        # Read the JSON response from the API endpoint into a DataFrame
        player_stats_df = read_json_to_df(StringIO(player_stats_response.text))

        # Append the DataFrame to the list of DataFrames
        player_stats_dfs.append(player_stats_df)

    # Concatenate the list of DataFrames into a single DataFrame
    player_stats_df = pd.concat(player_stats_dfs)

    # Write the DataFrame to a CSV file
    player_stats_df.to_csv(f"csv/{filename}.csv", index=False)

# Read the current top 100 players JSON file into a DataFrame
init_100_json = "Top-100-NHL-20232024-Jan-6-2023.json"
init_100_df = read_json_to_df(init_100_json)

# save it to a csv
init_100_df.to_csv("csv/Top-100-NHL-20232024-Jan-6-2023.csv", index=False)

# get the list of columns
# columns = init_100_df.columns.tolist()

# Get the player IDs from the DataFrame
init_100_ids = init_100_df["playerId"].tolist()

# Create a CSV file with the historical stats for the top 100 players
create_historical_csv(init_100_ids, '20222023', '20182019', "Top-100-NHL-5-year-Stats")



