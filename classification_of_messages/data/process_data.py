# import packages
import sys
import pandas as pd
from sqlalchemy import create_engine

DEBUG = True
def debug_message(message):
    '''
    Print debug messages (if activated)

    Args:
        message: the message to be printed

    Returns:
        None
    '''
    
    if DEBUG == True:
        print("Debug: {0}".format(message))

def extract_column_names(row):
    '''
    The function retrieves column names in provided row.
    
    Column names must be string before first occurance of '-'
    
    Args:
        row
    
    Returns:
        column_names
    '''
    
    column_names = []
    
    for c in row:
        s = c.split('-') 
        column_names.append(s[0])
        
    return column_names

def load_data(messages_path, categories_path):
    '''
    The function loads data
        
    Args:
        messages_path: path to the file containing messages
        categories_path: path to the file containing categories
    
    Returns:
        df: pandas dataframe containing merged datasets
    '''
    
    debug_message("run_ETL_pipeline entry (messages_path: {} | categories_path: {})".format(messages_path, categories_path))

    # Load datasets
    messages = pd.read_csv(messages_path)
    categories = pd.read_csv(categories_path)
    
    # Merge datasets
    df = pd.merge(messages, categories, how="left", on="id")
    
    debug_message("load_data exit")

    return df

def clean_data(df):
    '''
    The function cleans data so they can be later applied for machine learning
        
    Args:
        df: pandas dataframe containing merged dataset
    
    Returns:
        df: cleaned pandas dataframe prepared to be used in machine learning
    '''

    # Split categories into separate category columns.
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(";", expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # extract a list of new column names for categories.

    #TODO: Can be optimized by executing vectorized operation instead of one by one... Crude looping in Pandas, or That Thing You Should Never Ever Do
    category_colnames = extract_column_names(row)

    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split('-').str[1]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    # Replace categories column in df with new category columns
    # drop the original categories column from `df`
    df = df.drop(columns = ["categories"])

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # Remove duplicates.
    
    # check number of duplicates
    duplicated = df[df.duplicated(subset = 'message')]
    debug_message("Number of duplicates before removing them: {}".format(duplicated.shape[0]))
    
    # drop duplicates
    df = df.drop_duplicates(subset = 'message')
    
    # check number of duplicates
    duplicated = df[df.duplicated(subset = 'message')]
    debug_message("Number of duplicates after removing them: {}".format(duplicated.shape[0]))
    
    debug_message("clean_data exit")

    return df
    
def run_ETL_pipeline(messages_path, categories_path, db_path):
    '''
    The function Orchestrates ETL pipeline run
        
    Args:
        messages_path: path to the file containing messages
        categories_path: path to the file containing categories
        db_path: path where to save the result
    
    Returns:
        None
    '''
    
    debug_message("run_ETL_pipeline entry (messages_path: {} | categories_path: {})".format(messages_path, categories_path))
    
    # 1. Load datasets
    df = load_data(messages_path, categories_path)

    # 2. clean data
    df = clean_data(df)
    
    # 3. stores cleaned data to database
    save_cleaned_data(df, db_path)
    
    debug_message("run_ETL_pipeline exit")


def save_cleaned_data(df, db_path):
    '''
    Saves scleaned data into the file
    
    Args:
        model: the model to be saved
        db_path: the location where model will be saved

    Returns:
        None
    '''
    
    debug_message("save model enter")

    # Export model as a pickle file
    sql_path = 'sqlite:///{}'.format(db_path)
    engine = create_engine(sql_path)
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')
    
    debug_message("save model exit")


def main():
    '''
    Main function orchestrating the execution
    '''
    
    if len(sys.argv) == 4:
        messages_path, categories_path, db_path  = sys.argv[1:]
            
        run_ETL_pipeline(messages_path, categories_path, db_path)  # run ETL pipeline  
        
    else:
        print('Please provide: \n'\
              '-the filepath of the disaster messages file as the first argument \n'\
              '-the filepath of the disaster categories file as the second argument \n'\
              '-the name of file where you want to save cleaned dataset \n'\
              '\nExample: python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db')
        


if __name__ == '__main__':
    main()
