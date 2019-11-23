# import packages
import sys
import pandas as pd
from sqlalchemy import create_engine

DEBUG = True
def debug_message(message):
    if DEBUG == True:
        print("Debug: {0}".format(message))

def extract_column_names(row):
    '''
    The function retrieves column names in provided row.
    
    Column names must be string before first occurance of '-'
    
    Input:
    row
    
    Output:
    column_names
    '''
    
    column_names = []
    
    for c in row:
        s = c.split('-') 
        column_names.append(s[0])
        
    return column_names

def run_ETL_pipeline(messages_path, categories_path, db_path):
    
    debug_message("run_ETL_pipeline entry (messages_path: {} | categories_path: {})".format(messages_path, categories_path))
    # 1. === read in file
    
    # 1. Load datasets
    messages = pd.read_csv(messages_path)
    categories = pd.read_csv(categories_path)
    
    # 2. Merge datasets
    df = pd.merge(messages, categories, how="left", on="id")

    # 2. === clean data
    
    #3. Split categories into separate category columns.
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(";", expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # extract a list of new column names for categories.

    #TODO: Can be optimized by executing vectorized operation instead of one by one... Crude looping in Pandas, or That Thing You Should Never Ever Do
    category_colnames = extract_column_names(row)

    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # 4. Convert category values to just numbers 0 or 1.
    # Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split('-').str[1]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    # 5. Replace categories column in df with new category columns
    # drop the original categories column from `df`
    df = df.drop(columns = ["categories"])

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # 6. Remove duplicates.
    
    # check number of duplicates
    duplicated = df[df.duplicated(subset = 'message')]
    debug_message("Number of duplicates before removing them: {}".format(duplicated.shape[0]))
    
    # drop duplicates
    df = df.drop_duplicates(subset = 'message')
    
    # check number of duplicates
    duplicated = df[df.duplicated(subset = 'message')]
    debug_message("Number of duplicates after removing them: {}".format(duplicated.shape[0]))
    
    
    # 3. === load to database
    sql_path = 'sqlite:///{}'.format(db_path)
    engine = create_engine(sql_path)
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')

    # 4. === define features and label arrays

    X = df['message']
    y = df.drop(columns=['message', 'original', 'genre'])
    
    debug_message("run_ETL_pipeline exit")
                  
    return X, y



def export_model(model):
    debug_message("export model enter")

    # Export model as a pickle file
    
    debug_message("export model exit")


def main():
    
    
    if len(sys.argv) == 4:
        messages_path, categories_path, db_path  = sys.argv[1:]
            
        X, y = run_ETL_pipeline(messages_path, categories_path, db_path)  # run ETL pipeline  
        
    else:
        print('Please provide: \n'\
              '-the filepath of the disaster messages file as the first argument \n'\
              '-the filepath of the disaster categories file as the second argument \n'\
              '-the name of file where you want to save cleaned dataset \n'\
              '\nExample: python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db')
        
    '''
    
    '''


if __name__ == '__main__':
    main()
