# Disaster Response ML Pipeline Web App, Natural Language Processing
![Web Interface](project_screen.png)

## Project description
Projects as this helps to emergency services by classifying incoming messages 
so requests for help are quickly adressed to the corresponding agencies. 
That is especially important during natural disasters when emergency workers are overloaded and time is crucial.

- I've build ETL pipeline that loads database of real messages received during natural disasters, cleans them, and saves result prepared for Machine Learning into SQLite database.
- Then I've splitted data into training set and test set and created ML pipeline using Natural Language Processing library NLTK and scikit-learn's Pipeline. The output is Multi-Output Supervised Learning Model categorizing incoming messages. I've optimized the model futher by using GridSearchCV to fine-tune model parameters. The resulting model accuracy is in average 84%. The model is exported into pickle file to be futher used.
- In the last step I've integrated Web App where you can input a message and get classification result.

The project works with real messages provided by [Figure Eight](https://www.figure-eight.com/) that were received during natural disasters.

## Usage
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to [http://127.0.0.1:3003/](http://127.0.0.1:3003) and enter messages you want to classify.

## Libraries used
Python 3
- pandas
- sqlalchemy 
- sklearn
- nltk
- flask
- plotly

## Files in the repository
- ‘data\process_data.py’: The ETL pipeline that process, cleans and stores prepared data for Machine Learning in a database.
- ‘models\trains_classifier.py’: The Machine Learning pipeline that fits, tunes, evaluates and exports the model to a pickle file.
- ‘app\run.py’: Starts the Flask server for the web app. User can in the web interface write messages he wants to classify.
- ‘app\templates\*.html’: HTML templates for the web app.