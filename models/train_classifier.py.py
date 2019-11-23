# import libraries
import sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import string
import pandas as pd 
from sqlalchemy import create_engine
import re
import pickle



import nltk
from nltk.tokenize import word_tokenize

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

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
   

def load_data(database_filepath):
    '''
    Load cleaned input data for the training

    Args:
        database_filepath: the path to the cleaned DB file

    Returns:
        X: input messages
        Y: output categories
    '''
    
    debug_message("load_data entry ({})".format(database_filepath))
    
    sql_path = "sqlite:///{0}".format(database_filepath)
    engine = create_engine(sql_path)

    query = "SELECT * From DisasterResponse"
    df = pd.read_sql_query(query, engine,)

    X = df['message'].values
    Y = df.drop(columns=['id', 'message', 'original', 'genre'])

    debug_message("load_data exit")

    return X, Y, 

stop_words = nltk.corpus.stopwords.words("english")
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
def tokenize(text):
    '''
    Text tokenization
    - Normalization to lowercase
    - Remove punctuation characters
    - Tokenization, lemmatization, and stop word removal
    
    Args:
        text: text as string
    
    Returns:
        tokens: a list of tockenized text
    '''
    
    #normalize text and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize andremove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return tokens


def build_model():
    '''
    Build model for training

    Args:
        None

    Returns:
        Model
    '''
    
    debug_message("build_model entry")

    model = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('forest', MultiOutputClassifier( RandomForestClassifier() ))
    ])
    
    debug_message("build_model exit")
    
    return model


def evaluate_model(model, x_test, y_test):
    '''
    Evaluate model accuracy and basic statistics as f-score

    Args:
        model:  model to be evaluated
        x_test: input data
        y_test: expected results

    Returns:
        None. Statistics is printed into terminal.
    '''
    
    debug_message("evaluate_model entry")


    y_predicted = model.predict(x_test)
    y_predicted_df = pd.DataFrame(y_predicted, columns = y_test.columns)

    overall_accuracy = (y_predicted == y_test).mean().mean()
    print("Overall accuracy: {0} \n".format(overall_accuracy))

    for column in y_test.columns:
        print('Feature: {}\n'.format(column))
        print('Accuracy: ', accuracy_score(y_test[column], y_predicted_df[column]))
        print(classification_report(y_test[column],y_predicted_df[column]))
        print('------------------------------------------------------\n')

    debug_message("evaluate_model exit")
    
def save_model(model, model_filepath):
    '''
    Saves trained model into the file

    Args:
        model:  model to be saved
        model_filepath: location where will be model saved

    Returns:
        None
    '''
    
    debug_message("save_model entry")
    
    with open(model_filepath, 'wb') as pickle_file:
        pickle.dump(model, pickle_file)
        
    debug_message("save_model exit")


def main():
    '''
    Main function orchestrating the execution
    '''
    
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state =30, test_size =0.33)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()