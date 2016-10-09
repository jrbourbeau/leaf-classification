#!/usr/bin/env python3

from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, log_loss, make_scorer
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, cross_val_score
from sklearn.model_selection import validation_curve, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

def preprocess(train_df, test_df=None):
    # Convert species string labels to numerical labels
    label_encoder = LabelEncoder().fit(train_df.species)
    num_labels = label_encoder.transform(train_df.species)
    # train_classes = list(label_encoder.classes_)
    # Get training features
    train_features = train_df.drop(['species', 'id'], axis=1)

    return train_features, num_labels

if __name__ == "__main__":

    # Load and preprocess training data
    train_df = pd.read_csv('../data/train.csv')
    train_features, train_classes = preprocess(train_df)
    # print(train_features.head(3))
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(train_features, train_classes)
    name = clf.__class__.__name__

    print("="*30)
    print(name)

    print('****Results****')
    train_predictions = clf.predict_proba(train_features)
    ll = log_loss(train_classes, train_predictions)
    print("Log Loss: {}".format(ll))

    sss = StratifiedShuffleSplit(n_splits=20, test_size=0.1, random_state=23)
    scores = cross_val_score(clf, train_features, train_classes, cv=sss, scoring='neg_log_loss')
    print("Log loss: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    scores = cross_val_score(clf, train_features, train_classes, cv=10, scoring='neg_log_loss')
    print("Log loss: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    param_grid = {'max_depth': [70,80,90],
                  'max_features':['sqrt', 'log2', 10]}
    grid = GridSearchCV(clf, param_grid=param_grid, cv=sss, verbose=3, scoring='neg_log_loss')
    # grid = GridSearchCV(clf, param_grid=param_grid, cv=sss, verbose=3, scoring=make_scorer(log_loss,
    #                                greater_is_better=False,
    #                                needs_proba=True))
    grid.fit(train_features, train_classes)
    print(grid.best_score_)
    print(grid.best_params_)
