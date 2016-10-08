#!/usr/bin/env python3

from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedShuffleSplit, ShuffleSplit
from sklearn.model_selection import cross_val_score

def preprocess(train_df, test_df=None):
    # Convert species string labels to numerical labels
    label_encoder = LabelEncoder().fit(train_df.species)
    num_labels = label_encoder.transform(train_df.species)
    # train_targets = list(label_encoder.classes_)
    # Get training features
    train_features = train_df.drop(['species', 'id'], axis=1)

    return train_features, num_labels

if __name__ == "__main__":

    # Load and preprocess training data
    train_df = pd.read_csv('../data/train.csv')
    train_features, train_targets = preprocess(train_df)
    # print(train_features.head(3))
    clf = RandomForestClassifier()
    scores = cross_val_score(clf, train_features, train_targets, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    sss = StratifiedShuffleSplit(train_targets, 10, test_size=0.2, random_state=23)
    scores = cross_val_score(clf, train_features, train_targets, cv=sss)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
