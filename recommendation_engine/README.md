# Recommendation Engine for IBM Watson Studio's data platform

## Project description
In the project I've build recommendation engine providing users of [IBM Watson Studio](https://www.ibm.com/cloud/watson-studio) with articles they migh like.
I've completed this project as part of Udacity Data Science Nanodegree.

The recommendation engine can make recommendation in a following ways:
1. Rank Based Recommendations
- Recommends the highest ranking articles in the community
2. Collaborative Filtering
- Recommends are the highest ranking articles read by similar users
3. SVD Matrix Factorization Recommendations
4. Content Based Recommendation
- Content Based Recommendation with corresponding web application is an intended part, for which implementation I will need to find a time.

Test-Drive Development approach is followed and proper implementation is verified by unit tests.

![IBM Watson Studio Screenshot](IBM_Watson_Studio_Screenshot.png)
## Usage
- To follow the process, open `Recommendations_with_IBM.ipynb`
- Corresponding web application allowing direct usage is an intended part, for which implementation I will need to find a time.

## Libraries used
Python 3
- pandas
- numpy
- matplotlib

## Files in the repository
- `Recommendations_with_IBM.ipynb`: Source code for the recommendation engine
- 'Data\articles_community.csv': dataset containing informations about articles
- 'Data\user-item-interactions.csv' dataset containing interaction between users and articles