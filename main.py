import pandas as pd
# import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns
from datetime import datetime as dt
import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
folder = 'kaggleDB/'
database='kaggleDB/all_data.xlsx'
# read_file = pd.read_excel(r'kaggleDB/all_data.xlsx')
# read_file.to_csv (r'kaggleDB/all_data.csv', index = None, header=True)

# Create dfInfo of the matches' information.
# From dfInfo, create yTrain which contains the final result of the match.
# From dfInfo, create xTrain which will contain info about the teams in every match.
# X_train - features of all matches from the dfInfo except for season 15/16.
# X_test - features of all matches from the season 15/16.
# y_train - label column of all matches except for seasons 15/16.
# y_train contains the values - "Home win", "Draw", "Away team".
# y_test - label column of all matches from the season 15/16.

# model = RandomForestClassifier(criterion='gini',
                         #    n_estimators=700, # num of trees in the forest
                         #    min_samples_split=10,
                         #    min_samples_leaf=1,
                         #    max_features='auto',
                         #    oob_score=True,
                         #    random_state=1,
                         #    n_jobs=-1)

model = KNeighborsClassifier()

model.fit(X_train, y_train)
# Predicting result
Y_pred = model.predict(X_test)
# calculate mean accuracy
mean_accuracy = accuracy_score(y_test, Y_pred)
print("hi")