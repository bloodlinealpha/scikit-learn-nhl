import requests

def get_request(url):
    """
    Makes a GET request to the specified URL and returns the response

    Parameters
    ----------
    url : str
        The URL to make the GET request to
    """
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response
        else:
            print(f"Request failed with status code {response.status_code}")
            return None
    except requests.RequestException as e:
        print(f"An error occurred: {e}")
        return None
