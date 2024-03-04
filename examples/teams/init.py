import pandas as pd
from io import StringIO
import json
from http_request import get_request

# Function to read a JSON file into a DataFrame
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

# Function to build and read the schedule JSON file into a DataFrame
def read_schedule_json_to_df(file_name):
    '''
    This function takes a JSON file and reads it into a DataFrame
    :param file_name: name of the JSON file
        :type file_name: str
    :return: DataFrame
    '''
    # read the string into a json object
    data = json.load(file_name)

    # check if the json object is empty
    if not data or not data['gameWeek'] or not data['gameWeek'][0]:
        return None

    # create an empty list to store the schedule
    schedule = []
    # loop through the dates in the json object
    current = data['gameWeek'][0]

    # loop through the games in the date
    for game in current['games']:
        # create a dictionary to store the game information
        game = {
            "gameDate": current['date'],
            "home_id": game['homeTeam']["id"],
            "homeTeam": game['homeTeam']["abbrev"],
            "away_id": game['awayTeam']["id"],
            "awayTeam": game['awayTeam']["abbrev"],
        }
        # append the game to the schedule list
        schedule.append(game)

    # convert the list of games into a DataFrame
    schedule = pd.DataFrame(schedule)

    return schedule

# Create a CSV file with the historical stats for the top 100 players
def create_historical_csv(team_ids, start, end, filename):
    """
    Generates a CSV file containing historical stats of teams.

    Parameters:
    ----------
    player_ids : list
        List of team IDs to fetch the historical stats for.
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
    team_stats_dfs = []
    # NHL API only allows 100 records to be returned at a time, so we need to loop through the data. 10 seasons is roughly 900 games so we will use 1000 as the last step
    query_step = 0
    # Loop through the player IDs
    for count, team_id in enumerate(team_ids, start=1):
        print(f"Getting stats for team {count} of {len(team_ids)}")
        while query_step < 1000:
            print(f"Query step {query_step} of 1000")
            # Get the player stats endpoint for the current player ID
            team_stats_api_endpoint = f"https://api.nhle.com/stats/rest/en/team/summary?isAggregate=false&isGame=true&sort=%5B%7B%22property%22:%22gameDate%22,%22direction%22:%22DESC%22%7D,%7B%22property%22:%22teamId%22,%22direction%22:%22ASC%22%7D%5D&start={query_step}&limit=100&factCayenneExp=gamesPlayed%3E=1&cayenneExp=franchiseId%3D{team_id}%20and%20gameTypeId=2%20and%20seasonId%3C={start}%20and%20seasonId%3E={end}"

            # make the http request
            team_stats_response = get_request(team_stats_api_endpoint)

            # Read the JSON response from the API endpoint into a DataFrame
            team_stats_df = read_json_to_df(StringIO(team_stats_response.text))

            # Append the DataFrame to the list of DataFrames
            team_stats_dfs.append(team_stats_df)

            # Increment the step
            query_step += 100

        query_step = 0

    # Concatenate the list of DataFrames into a single DataFrame
    all_teams_stats_df = pd.concat(team_stats_dfs)

    # Write the DataFrame to a CSV file
    all_teams_stats_df.to_csv(f"csv/{filename}.csv", index=False)

    return None

# Function to get the historical stats for all teams
def get_historical_team_stats():
    # Get the team IDs for all teams that have played in the NHL
    team_query = "https://api.nhle.com/stats/rest/en/franchise?sort=fullName&include=lastSeason.id&include=firstSeason.id"

    team_query_results = get_request(team_query)
    # Read the JSON response from the API endpoint into a DataFrame
    team_query_df = read_json_to_df(StringIO(team_query_results.text))
    # Filter the results to only include the team ID where lastSeason is null
    team_ids = team_query_df[team_query_df['lastSeason.id'].isnull()]
    # filter so only team ids are included
    team_ids = team_ids['id'].tolist()


    # pick which seasons to get the historical stats for and uncomment the line to run the function

    # Create a CSV file with the historical stats for the all teams back to 2013/2014 season
    create_historical_csv(team_ids, '20222023', '20132014', "NHL_teams_historical_stats_20132014_to_20222023")

    # Create a CSV file with the historical stats for the all teams the current season
    #create_historical_csv(team_ids, '20232024', '20232024', "NHL_teams_historical_stats_20232024")

# Function to get the current schedule for the 2023/2024 season
def get_current_schedule():
    # Get the current schedule for the 2023/2024 season
    initial_start_date = "2024-02-26"
    start_date = "2024-02-26"
    end_date = "2024-04-19"

    # create an empty DataFrame to store the schedule
    schedule_query_dfs = []
    # Loop through all the dates between the start and end dates
    while start_date < end_date:
        print(f"Getting schedule for {start_date}")
        # Get the schedule for the current date
        schedule_query = f"https://api-web.nhle.com/v1/schedule/{start_date}"
        # Increment the date by 1 day
        start_date = pd.to_datetime(start_date) + pd.DateOffset(days=1)
        # Format the date as a string
        start_date = start_date.strftime("%Y-%m-%d")

        # make the http request
        schedule_query_results = get_request(schedule_query)

        # Read the JSON response from the API endpoint into a DataFrame
        schedule_query_df = read_schedule_json_to_df(StringIO(schedule_query_results.text))

        if schedule_query_df is None:
            continue

        # Append the DataFrame to the list of DataFrames
        schedule_query_dfs.append(schedule_query_df)
    
    # Concatenate the list of DataFrames into a single DataFrame
    schedule_query_dfs = pd.concat(schedule_query_dfs)

    # Write the DataFrame to a CSV file
    schedule_query_dfs.to_csv(f"csv/NHL_teams_schedule_20232024_{initial_start_date}.csv", index=False)

    return None


if __name__ == "__main__":
    # get_historical_team_stats()
    get_current_schedule()