
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
from app.connection import create_conn_240

def authenticate_google_sheets(credentials_file='app/attemp-rate-7478cf3b26b2.json'):
    """
    Authenticate the Google Sheets API using service account credentials.
    """
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(credentials_file, scope)
    client = gspread.authorize(creds)
    return client

# Fetch data from Google Sheets, including multiple tabs (worksheets)
def fetch_data_from_google_sheet(spreadsheet_name='Maturity_Curve', tabs=None):
    """
    Fetch data from a Google Sheet with multiple tabs.
    
    Args:
    - spreadsheet_name (str): Name of the Google Spreadsheet to fetch data from.
    - tabs (list): List of tab (worksheet) names to fetch data from. If None, fetches data from all tabs.
    
    Returns:
    - dict: A dictionary of Pandas DataFrames, where keys are tab names and values are corresponding DataFrames.
    """
    # Authenticate Google Sheets
    client = authenticate_google_sheets()
    
    # Open the spreadsheet by name
    spreadsheet = client.open(spreadsheet_name)
    
    # If no specific tabs are provided, fetch all tabs
    if tabs is None:
        tabs = [worksheet.title for worksheet in spreadsheet.worksheets()]
    
    # Create a dictionary to hold the DataFrames
    data = {}
    
    # Loop through each tab and fetch its data
    for tab in tabs:
        worksheet = spreadsheet.worksheet(tab)
        # Get the data from the worksheet
        data_from_tab = worksheet.get_all_records()  # Get all rows as a list of dictionaries
        
        # Convert to Pandas DataFrame
        df = pd.DataFrame(data_from_tab)
        
        # Store the DataFrame in the dictionary with the tab name as the key
        data[tab] = df
    
    return data


def fetch_table(table_name, conn):
    """
    Fetch data from a single MySQL table
    """
    query = f"SELECT * FROM {table_name}"
    return pd.read_sql(query, conn)

def resolve_table_key(table_name, table_key_mapping):
    """
    Resolve dictionary key for a table
    """
    if table_key_mapping and table_name in table_key_mapping:
        return table_key_mapping[table_name]
    return table_name


def fetch_tables_as_dict(table_list, table_key_mapping=None):
    """
    Fetch multiple tables and return a dictionary of DataFrames
    """
    conn = create_conn_240()
    data_dict = {}

    try:
        for table in table_list:
            key = resolve_table_key(table, table_key_mapping)
            data_dict[key] = fetch_table(table, conn)
    finally:
        conn.close()

    return data_dict
