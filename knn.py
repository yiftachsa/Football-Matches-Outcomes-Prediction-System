import sqlite3
import pandas as pd
import numpy as np
import seaborn as sns
import itertools
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer
from time import time
from sklearn.decomposition import PCA, FastICA
from sklearn.pipeline import Pipeline
import warnings
import statistics as stats
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.inspection import permutation_importance

warnings.simplefilter("ignore")


## Loading all functions
def get_match_label(match):
    ''' Derives a label for a given match. '''

    # Define variables
    home_goals = match['home_team_goal']
    away_goals = match['away_team_goal']

    label = pd.DataFrame()
    label.loc[0, 'match_api_id'] = match['match_api_id']

    # Identify match label
    if home_goals > away_goals:
        label.loc[0, 'label'] = "Win"
    if home_goals == away_goals:
        label.loc[0, 'label'] = "Draw"
    if home_goals < away_goals:
        label.loc[0, 'label'] = "Defeat"

    # Return label
    return label.loc[0]


def get_fifa_stats(match, player_stats):
    ''' Aggregates fifa stats for a given match. '''

    # Define variables
    match_id = match.match_api_id
    date = match['date']
    players = ['home_player_1', 'home_player_2', 'home_player_3', "home_player_4", "home_player_5",
               "home_player_6", "home_player_7", "home_player_8", "home_player_9", "home_player_10",
               "home_player_11", "away_player_1", "away_player_2", "away_player_3", "away_player_4",
               "away_player_5", "away_player_6", "away_player_7", "away_player_8", "away_player_9",
               "away_player_10", "away_player_11"]
    player_stats_new = pd.DataFrame()
    names = []

    # Loop through all players
    for player in players:

        # Get player ID
        player_id = match[player]

        # Get player stats
        stats = player_stats[player_stats.player_api_id == player_id]

        # Identify current stats
        current_stats = stats[stats.date < date].sort_values(by='date', ascending=False)[:1]

        if np.isnan(player_id) == True:
            overall_rating = pd.Series(0)
        else:
            current_stats.reset_index(inplace=True, drop=True)
            overall_rating = pd.Series(current_stats.loc[0, "overall_rating"])

        # Rename stat
        name = "{}_overall_rating".format(player)
        names.append(name)

        # Aggregate stats
        player_stats_new = pd.concat([player_stats_new, overall_rating], axis=1)

    player_stats_new.columns = names
    player_stats_new['match_api_id'] = match_id

    player_stats_new.reset_index(inplace=True, drop=True)

    # Return player stats
    return player_stats_new.loc[0]


def get_fifa_data(matches, player_stats, path=None, data_exists=False):
    ''' Gets fifa data for all matches. '''

    # Check if fifa data already exists
    if data_exists == True:

        fifa_data = pd.read_pickle(path)

    else:

        print("Collecting fifa data for each match...")
        start = time()

        # Apply get_fifa_stats for each match
        fifa_data = matches.apply(lambda x: get_fifa_stats(x, player_stats), axis=1)

        end = time()
        print("Fifa data collected in {:.1f} minutes".format((end - start) / 60))

    # Return fifa_data
    return fifa_data


def get_overall_fifa_rankings(fifa, get_overall=False):
    ''' Get overall fifa rankings from fifa data. '''

    temp_data = fifa

    # Check if only overall player stats are desired
    if get_overall == True:

        # Get overall stats
        data = temp_data.loc[:, (fifa.columns.str.contains('overall_rating'))]
        data.loc[:, 'match_api_id'] = temp_data.loc[:, 'match_api_id']
    else:

        # Get all stats except for stat date
        cols = fifa.loc[:, (fifa.columns.str.contains('date_stat'))]
        temp_data = fifa.drop(cols.columns, axis=1)
        data = temp_data

    # Return data
    data = extract_team_players_aggregated_features(data)
    return data


def extract_team_players_aggregated_features(data):
    home_players = data.loc[:, (data.columns.str.contains('home_player'))]
    # data['home_players_team_mean_ovr'] = data[home_players.columns].mean(axis=1)
    # data['home_players_team_std_ovr'] = data[home_players.columns].std(axis=1)
    data['home_players_team_mean_ovr'] = home_players.mean(axis=1)
    data['home_players_team_std_ovr'] = home_players.std(axis=1)
    data = data.drop(home_players.columns, axis=1)

    away_players = data.loc[:, (data.columns.str.contains('away_player'))]
    # data['away_players_team_mean_ovr'] = data[away_players.columns].mean(axis=1)
    # data['away_players_team_std_ovr'] = data[away_players.columns].std(axis=1)
    data['away_players_team_mean_ovr'] = away_players.mean(axis=1)
    data['away_players_team_std_ovr'] = away_players.std(axis=1)
    data = data.drop(away_players.columns, axis=1)

    return data


def get_last_matches(matches, date, team, x=10):
    ''' Get the last x matches of a given team. '''

    # Filter team matches from matches
    team_matches = matches[(matches['home_team_api_id'] == team) | (matches['away_team_api_id'] == team)]

    # Filter x last matches from team matches
    last_matches = team_matches[team_matches.date < date].sort_values(by='date', ascending=False).iloc[0:x, :]

    # Return last matches
    return last_matches


def get_last_matches_against_eachother(matches, date, home_team, away_team, x=10):
    ''' Get the last x matches of two given teams. '''

    # Find matches of both teams
    home_matches = matches[(matches['home_team_api_id'] == home_team) & (matches['away_team_api_id'] == away_team)]
    away_matches = matches[(matches['home_team_api_id'] == away_team) & (matches['away_team_api_id'] == home_team)]
    total_matches = pd.concat([home_matches, away_matches])

    # Get last x matches
    try:
        last_matches = total_matches[total_matches.date < date].sort_values(by='date', ascending=False).iloc[0:x, :]
    except:
        last_matches = total_matches[total_matches.date < date].sort_values(by='date', ascending=False).iloc[
                       0:total_matches.shape[0], :]

        # Check for error in data
        if (last_matches.shape[0] > x):
            print("Error in obtaining matches")

    # Return data
    return last_matches


def get_goals(matches, team):
    ''' Get the goals of a specfic team from a set of matches. '''

    # Find home and away goals
    home_goals = int(matches.home_team_goal[matches.home_team_api_id == team].sum())
    away_goals = int(matches.away_team_goal[matches.away_team_api_id == team].sum())

    total_goals = home_goals + away_goals

    # Return total goals
    return total_goals


def get_goals_conceided(matches, team):
    ''' Get the goals conceided of a specfic team from a set of matches. '''

    # Find home and away goals
    home_goals = int(matches.home_team_goal[matches.away_team_api_id == team].sum())
    away_goals = int(matches.away_team_goal[matches.home_team_api_id == team].sum())

    total_goals = home_goals + away_goals

    # Return total goals
    return total_goals


def get_wins(matches, team):
    ''' Get the number of wins of a specfic team from a set of matches. '''

    # Find home and away wins
    home_wins = int(matches.home_team_goal[
                        (matches.home_team_api_id == team) & (matches.home_team_goal > matches.away_team_goal)].count())
    away_wins = int(matches.away_team_goal[
                        (matches.away_team_api_id == team) & (matches.away_team_goal > matches.home_team_goal)].count())

    total_wins = home_wins + away_wins

    # Return total wins
    return total_wins


def get_match_features(match, matches, x=10):
    ''' Create match specific features for a given match. '''

    # Define variables
    date = match.date
    home_team = match.home_team_api_id
    away_team = match.away_team_api_id

    # Get last x matches of home and away team
    matches_home_team = get_last_matches(matches, date, home_team, x=10)
    matches_away_team = get_last_matches(matches, date, away_team, x=10)

    # Get last x matches of both teams against each other
    last_matches_against = get_last_matches_against_eachother(matches, date, home_team, away_team, x=3)

    # Create goal variables
    home_goals = get_goals(matches_home_team, home_team)
    away_goals = get_goals(matches_away_team, away_team)
    home_goals_conceided = get_goals_conceided(matches_home_team, home_team)
    away_goals_conceided = get_goals_conceided(matches_away_team, away_team)

    # Define result data frame
    result = pd.DataFrame()

    # Define ID features
    result.loc[0, 'match_api_id'] = match.match_api_id
    result.loc[0, 'league_id'] = match.league_id

    # Create match features
    result.loc[0, 'home_team_goals_difference'] = home_goals - home_goals_conceided
    result.loc[0, 'away_team_goals_difference'] = away_goals - away_goals_conceided
    result.loc[0, 'games_won_home_team'] = get_wins(matches_home_team, home_team)
    result.loc[0, 'games_won_away_team'] = get_wins(matches_away_team, away_team)
    result.loc[0, 'games_against_won'] = get_wins(last_matches_against, home_team)
    result.loc[0, 'games_against_lost'] = get_wins(last_matches_against, away_team)

    # Return match features
    return result.loc[0]


def create_feables(matches, fifa, bookkeepers, get_overall=False, horizontal=True, x=10, verbose=True):
    ''' Create and aggregate features and labels for all matches. '''

    # Get fifa stats features
    fifa_stats = get_overall_fifa_rankings(fifa, get_overall)

    if verbose == True:
        print("Generating match features...")
    start = time()

    # Get match features for all matches
    match_stats = matches.apply(lambda x: get_match_features(x, matches, x=10), axis=1)

    # Create dummies for league ID feature
    dummies = pd.get_dummies(match_stats['league_id']).rename(columns=lambda x: 'League_' + str(x))
    match_stats = pd.concat([match_stats, dummies], axis=1)
    match_stats.drop(['league_id'], inplace=True, axis=1)

    end = time()
    if verbose == True:
        print("Match features generated in {:.1f} minutes".format((end - start) / 60))

    if verbose == True:
        print("Generating match labels...")
    start = time()

    # Create match labels
    labels = matches.apply(get_match_label, axis=1)
    end = time()
    if verbose == True:
        print("Match labels generated in {:.1f} minutes".format((end - start) / 60))

    if verbose == True:
        print("Generating bookkeeper data...")
    start = time()

    # Get bookkeeper quotas for all matches
    bk_data = get_bookkeeper_data(matches, bookkeepers, horizontal=True)
    bk_data.loc[:, 'match_api_id'] = matches.loc[:, 'match_api_id']
    end = time()
    if verbose == True:
        print("Bookkeeper data generated in {:.1f} minutes".format((end - start) / 60))

    # Merges features and labels into one frame
    features = pd.merge(match_stats, fifa_stats, on='match_api_id', how='left')
    features = pd.merge(features, bk_data, on='match_api_id', how='left')
    feables = pd.merge(features, labels, on='match_api_id', how='left')

    # Drop NA values
    feables.dropna(inplace=True)

    # Return preprocessed data
    return feables


def train_classifier(clf, dm_reduction, X_train, y_train, cv_sets, params, scorer, jobs, use_grid_search=True,
                     best_components=None, best_params=None):
    ''' Fits a classifier to the training data. '''

    # Start the clock, train the classifier, then stop the clock
    start = time()

    # Check if grid search should be applied
    if use_grid_search == True:

        # Define pipeline of dm reduction and classifier
        estimators = [('dm_reduce', dm_reduction), ('clf', clf)]
        pipeline = Pipeline(estimators)

        # Grid search over pipeline and return best classifier
        grid_obj = model_selection.GridSearchCV(pipeline, param_grid=params, scoring=scorer, cv=cv_sets, n_jobs=jobs)
        grid_obj.fit(X_train, y_train)
        best_pipe = grid_obj.best_estimator_
    else:

        # Use best components that are known without grid search
        estimators = [('dm_reduce', dm_reduction(n_components=best_components)), ('clf', clf(best_params))]
        pipeline = Pipeline(estimators)
        best_pipe = pipeline.fit(X_train, y_train)

    end = time()

    # Print the results
    print("Trained {} in {:.1f} minutes".format(clf.__class__.__name__, (end - start) / 60))

    # Return best pipe
    return best_pipe


def predict_labels(clf, best_pipe, features, target):
    ''' Makes predictions using a fit classifier based on scorer. '''

    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(best_pipe.named_steps['dm_reduce'].transform(features))
    end = time()

    # Print and return results
    print("Made predictions in {:.4f} seconds".format(end - start))
    return accuracy_score(target.values, y_pred)


def train_calibrate_predict(clf, dm_reduction, X_train, y_train, X_calibrate, y_calibrate, X_test, y_test, cv_sets,
                            params, scorer, jobs,
                            use_grid_search=True, **kwargs):
    ''' Train and predict using a classifer based on scorer. '''

    # Indicate the classifier and the training set size
    print("Training a {} with {}...".format(clf.__class__.__name__, dm_reduction.__class__.__name__))

    # Train the classifier
    best_pipe = train_classifier(clf, dm_reduction, X_train, y_train, cv_sets, params, scorer, jobs)

    # Calibrate classifier
    print("Calibrating probabilities of classifier...")
    start = time()
    clf = CalibratedClassifierCV(best_pipe.named_steps['clf'], cv='prefit', method='isotonic')
    clf.fit(best_pipe.named_steps['dm_reduce'].transform(X_calibrate), y_calibrate)
    end = time()
    print("Calibrated {} in {:.1f} minutes".format(clf.__class__.__name__, (end - start) / 60))

    # Print the results of prediction for both training and testing
    print("Score of {} for training set: {:.4f}.".format(clf.__class__.__name__,
                                                         predict_labels(clf, best_pipe, X_train, y_train)))
    print("Score of {} for test set: {:.4f}.".format(clf.__class__.__name__,
                                                     predict_labels(clf, best_pipe, X_test, y_test)))

    # Return classifier, dm reduction, and label predictions for train and test set
    return clf, best_pipe.named_steps['dm_reduce'], predict_labels(clf, best_pipe, X_train, y_train), predict_labels(
        clf, best_pipe, X_test, y_test)


def convert_odds_to_prob(match_odds):
    ''' Converts bookkeeper odds to probabilities. '''

    # Define variables
    match_id = match_odds.loc[:, 'match_api_id']
    bookkeeper = match_odds.loc[:, 'bookkeeper']
    win_odd = match_odds.loc[:, 'Win']
    draw_odd = match_odds.loc[:, 'Draw']
    loss_odd = match_odds.loc[:, 'Defeat']

    # Converts odds to prob
    win_prob = 1 / win_odd
    draw_prob = 1 / draw_odd
    loss_prob = 1 / loss_odd

    total_prob = win_prob + draw_prob + loss_prob

    probs = pd.DataFrame()

    # Define output format and scale probs by sum over all probs
    probs.loc[:, 'match_api_id'] = match_id
    probs.loc[:, 'bookkeeper'] = bookkeeper
    probs.loc[:, 'Win'] = win_prob / total_prob
    probs.loc[:, 'Draw'] = draw_prob / total_prob
    probs.loc[:, 'Defeat'] = loss_prob / total_prob

    # Return probs and meta data
    return probs


def get_bookkeeper_data(matches, bookkeepers, horizontal=True):
    ''' Aggregates bookkeeper data for all matches and bookkeepers. '''

    # bk_data = pd.DataFrame()
    # in order to calc the mean of all of the bookkeepers' odds
    num_of_bks = 0
    avg_bk_data = pd.DataFrame()

    # Loop through bookkeepers
    for bookkeeper in bookkeepers:

        num_of_bks = num_of_bks + 1
        # Find columns containing data of bookkeeper
        temp_data = matches.loc[:, (matches.columns.str.contains(bookkeeper))]
        temp_data.loc[:, 'bookkeeper'] = str(bookkeeper)
        temp_data.loc[:, 'match_api_id'] = matches.loc[:, 'match_api_id']

        # Rename odds columns and convert to numeric
        cols = temp_data.columns.values
        cols[:3] = ['Win', 'Draw', 'Defeat']
        temp_data.columns = cols
        temp_data.loc[:, 'Win'] = pd.to_numeric(temp_data['Win'])
        temp_data.loc[:, 'Draw'] = pd.to_numeric(temp_data['Draw'])
        temp_data.loc[:, 'Defeat'] = pd.to_numeric(temp_data['Defeat'])

        # Check if data should be aggregated horizontally
        if (horizontal == True):

            # Convert data to probs
            temp_data = convert_odds_to_prob(temp_data)
            temp_data.drop('match_api_id', axis=1, inplace=True)
            temp_data.drop('bookkeeper', axis=1, inplace=True)

            # Rename columns with bookkeeper names
            # win_name = bookkeeper + "_" + "Win"
            # draw_name = bookkeeper + "_" + "Draw"
            # defeat_name = bookkeeper + "_" + "Defeat"
            # bk home win, bk draw, bk away win
            win_name = "BKH"
            draw_name = "BKD"
            defeat_name = "BKA"
            temp_data.columns.values[:3] = [win_name, draw_name, defeat_name]

            # temp_data is now containing win, draw, lose probabilities of the match, according to 1 bk
            # Aggregate data
            if avg_bk_data.empty:
                avg_bk_data = pd.concat([avg_bk_data, temp_data], axis=1)
            else:  # sum every cell in avg_bk_data with the suitable cell from temp_data
                avg_bk_data = avg_bk_data + temp_data
            # bk_data = pd.concat([bk_data, temp_data], axis=1)
        # else:
        # Aggregate vertically
        # bk_data = bk_data.append(temp_data, ignore_index=True)

    # If horizontal add match api id to data
    if (horizontal == True):
        temp_data.loc[:, 'match_api_id'] = matches.loc[:, 'match_api_id']

    # Calculate the average probabilities by dividing each cell by the num of bks
    if num_of_bks > 1:
        avg_bk_data = avg_bk_data / num_of_bks
    # Return average bookkeeper data
    return avg_bk_data


def get_bookkeeper_probs(matches, bookkeepers, horizontal=False):
    ''' Get bookkeeper data and convert to probabilities for vertical aggregation. '''

    # Get bookkeeper data
    data = get_bookkeeper_data(matches, bookkeepers, horizontal=False)

    # Convert odds to probabilities
    probs = convert_odds_to_prob(data)

    # Return data
    return probs


def explore_data(features, inputs, path):
    ''' Explore data by plotting KDE graphs. '''

    # Define figure subplots
    fig = plt.figure(1)
    fig.subplots_adjust(bottom=-1, left=0.025, top=2, right=0.975)

    # Loop through features
    i = 1
    for col in features.columns:
        # Set subplot and plot format
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale=0.5, rc={"lines.linewidth": 1})
        plt.subplot(7, 7, 0 + i)
        j = i - 1

        # Plot KDE for all labels
        sns.distplot(inputs[inputs['label'] == 'Win'].iloc[:, j], hist=False, label='Win')
        sns.distplot(inputs[inputs['label'] == 'Draw'].iloc[:, j], hist=False, label='Draw')
        sns.distplot(inputs[inputs['label'] == 'Defeat'].iloc[:, j], hist=False, label='Defeat')
        plt.legend();
        i = i + 1

    # Define plot format
    DefaultSize = fig.get_size_inches()
    fig.set_size_inches((DefaultSize[0] * 1.2, DefaultSize[1] * 1.2))

    plt.show()

    # Compute and print label weights
    labels = inputs.loc[:, 'label']
    class_weights = labels.value_counts() / len(labels)
    print(class_weights)

    # Store description of all features
    feature_details = features.describe().transpose()

    # Return feature details
    return feature_details


def fill_na(dataframe):
    """
    This Functions is getting a pandas Dataframe that might contains na fields
    and returns the Dataframe with no na fields
    :param dataframe: the pandas dataframe to fill na values to
    :return: pandas dataframe with no na values
    """
    dtypes = dataframe.dtypes

    empty_cols = [col for col in dataframe.columns if dataframe[col].isnull().any()]
    for col in empty_cols:
        if dtypes[col] == 'float64' or dtypes[col] == 'int64':
            dataframe[col].fillna(dataframe[col].mean(), inplace=True)
        else:
            dataframe[col].fillna(stats.mode(dataframe[col]), inplace=True)
    return dataframe


def z_score_standardization(data_frame):
    """
    Receives a dataframe and standardize all the values in the numeric columns.
    :param data_frame: a dataframe with numeric columns containing no missing values
    ":returns: a standardize dataframe
    """
    columns = list(data_frame.select_dtypes('number').columns.values)
    for column_name in columns:
        data_frame[column_name] = (data_frame[column_name] - data_frame[column_name].mean()) / data_frame[
            column_name].std(ddof=0)
    return data_frame


start = time()
## Fetching data
# Connecting to database
path = "../input/"  # Insert path here

# Defining the number of jobs to be run in parallel during grid search
n_jobs = 1  # Insert number of parallel jobs here
player_stats_data = pd.DataFrame()


def preprocess(match_path):
    global player_stats_data
    # Fetching required data tables
    match_data = pd.read_excel(match_path, sheet_name='match')
    print("finish match")

    if player_stats_data.empty:
        player_stats_data = pd.read_excel('./all_data.xlsx', sheet_name='player_attribute')
    print("finish team_attribute")

    # Reduce match data to fulfill run time requirements
    rows = ["country_id", "league_id", "season", "stage", "date", "match_api_id", "home_team_api_id",
            "away_team_api_id", "home_team_goal", "away_team_goal", "home_player_1", "home_player_2",
            "home_player_3", "home_player_4", "home_player_5", "home_player_6", "home_player_7",
            "home_player_8", "home_player_9", "home_player_10", "home_player_11", "away_player_1",
            "away_player_2", "away_player_3", "away_player_4", "away_player_5", "away_player_6",
            "away_player_7", "away_player_8", "away_player_9", "away_player_10", "away_player_11"]

    match_data.dropna(subset=rows, inplace=True)

    # TO DELETE
    # match_data = match_data.head(1000)

    ## Generating features, exploring the data, and preparing data for model training
    # Generating or retrieving already existant FIFA data
    fifa_data = get_fifa_data(match_data, player_stats_data, data_exists=False)

    # Creating features and labels based on data provided
    bk_cols = ['B365', 'BW', 'IW', 'LB', 'PS', 'WH', 'SJ', 'VC', 'GB', 'BS']
    bk_cols_selected = ['B365', 'BW']
    feables = create_feables(match_data, fifa_data, bk_cols_selected, get_overall=True)

    # inputs is the aggregated dataframe of all the matches' data
    inputs = feables.drop('match_api_id', axis=1)

    # Exploring the data and creating visualizations
    labels = inputs.loc[:, 'label']
    features = inputs.drop(['label', 'League_1.0'], axis=1)
    # remove columns that starts with "Leagu_"
    unwanted = inputs.columns[inputs.columns.str.startswith('League_')]
    features = inputs.drop(unwanted, axis=1, inplace=True)
    # feature_details = explore_data(features, inputs, path)
    features = z_score_standardization(features)
    return features, labels

# ** only in first run **
# x_train, y_train = preprocess('./train.xlsx')
# x_train.to_excel('./x_train_after_preprocess.xlsx')
# y_train.to_excel('./y_train_after_preprocess.xlsx')
#
# x_test, y_test = preprocess('./test.xlsx')
# x_test.to_excel('./x_test_after_preprocess.xlsx')
# y_test.to_excel('./y_test_after_preprocess.xlsx')

# ** only in second+ run, load the files **
x_train = pd.read_excel("./x_train_after_preprocess.xlsx")
y_train = pd.read_excel("./y_train_after_preprocess.xlsx")
x_test = pd.read_excel("./x_test_after_preprocess.xlsx")
y_test = pd.read_excel("./y_test_after_preprocess.xlsx")
# Remove index column
x_train = x_train.drop(x_train.columns[0], axis=1)
y_train = y_train.drop(y_train.columns[0], axis=1)
x_test = x_test.drop(x_test.columns[0], axis=1)
y_test = y_test.drop(y_test.columns[0], axis=1)


# **** RandomForestClassifier ****
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Train the model
# n_estimators=25, min_samples_split=25, max_depth=7
randomModel = RandomForestClassifier(n_estimators=200, random_state=1, class_weight='balanced')
randomModel.fit(x_train, y_train)

# Evaluate the training set, get accuracy
prediction = randomModel.predict(x_train)
# mean_accuracy = accuracy_score(y_test, prediction)
accuracy = metrics.accuracy_score(prediction, y_train)
print("Train set accuracy : %s" % "{0:.3%}".format(accuracy))

# Evaluate the model, get accuracy
prediction = randomModel.predict(x_test)
# mean_accuracy = accuracy_score(y_test, prediction)
accuracy = metrics.accuracy_score(prediction, y_test)
print("Test set accuracy : %s" % "{0:.3%}".format(accuracy))

# *** Classification Report ****
target_names = ['Win', 'Draw', 'Defeat']
print("Classification Report")
print(classification_report(y_test, prediction, target_names=target_names))


# choose the top 10 feature which are the most important and only leave them in the model.
featimp = pd.Series(randomModel.feature_importances_, index=x_train.columns).sort_values(ascending=False)
print("features : ")
print(featimp)
print("feature importances - top 10 ")
topFeatures = featimp.head(10)
print(topFeatures)

# Train the model again based on the important features
predictor_var = topFeatures.keys()
# Create the model again.
randomModel = RandomForestClassifier(n_estimators=200, random_state=1, class_weight='balanced')
randomModel.fit(x_train[predictor_var], y_train)

# Evaluate the training set again, get accuracy
prediction = randomModel.predict(x_train[predictor_var])
# mean_accuracy = accuracy_score(y_test, prediction)
accuracy = metrics.accuracy_score(prediction, y_train)
print("Train set accuracy after feature selection: %s" % "{0:.3%}".format(accuracy))

# Evaluate the model again based on the important features
prediction = randomModel.predict(x_test[predictor_var])
accuracy = metrics.accuracy_score(prediction, y_test)
print("Test set accuracy after feature selection: %s" % "{0:.3%}".format(accuracy))

# *** Classification Report ****
target_names = ['Win', 'Draw', 'Defeat']
print("Classification Report after selecting top features")
print(classification_report(y_test, prediction, target_names=target_names))

# # ********** KNN ************
# model = KNeighborsClassifier()
# model.fit(x_train, y_train)
# # Predicting result
# prediction = model.predict(x_test)
# # Evaluate the model, get accuracy
# accuracy = metrics.accuracy_score(prediction, y_test)
# print("Model accuracy : %s" % "{0:.3%}".format(accuracy))
# print(accuracy)
#
# # perform permutation importance
# results = permutation_importance(model, x_train, y_train, scoring='accuracy')
# # get importance
# importance = results.importances_mean
# # summarize feature importance
# featimp = {}
# for i, v in enumerate(importance):
#     # print('Feature: %0d, Score: %.5f' % (i,v))
#     print('Feature: %0d, Score: %.5f' % (i, v) + ' Feature name is ' + x_train.columns[i])
#     featimp[x_train.columns[i]] = v
#
# featimp = {k: v for k, v in sorted(featimp.items(), reverse=True, key=lambda x: x[1])}
# print("********* featimp sorted ************")
# print(featimp)
# # Train the model again based on the important features
# # choose the top 10 feature which are the most important and only leave them in the model.
# predictor_var = list(featimp.keys())[:8]
# print("********* the top features selected: ************")
# print(predictor_var)
#
# # Create the model again.
# model = KNeighborsClassifier()
# model.fit(x_train[predictor_var], y_train)
# # Evaluate the model again based on the important features
# prediction = model.predict(x_test[predictor_var])
# accuracy = metrics.accuracy_score(prediction, y_test)
# print("Model accuracy after feature selection: %s" % "{0:.3%}".format(accuracy))

