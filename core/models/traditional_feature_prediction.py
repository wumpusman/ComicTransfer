import numpy as np
import pandas as pd
import os
import sys
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ShuffleSplit

sys.path.append("../..")


class FeaturePredictionTraditional():
    """
    Attributes:
        _model: model used for predicting typically font size or bounding box area
        _scale: does basic rescaling of continious values
        _x: input features of data
        _y: ground truth that model is being trained to predict
        _x_names: names of the corresponding _x features
        _y_names: names of the corresponding _y features
        _raw_data: raw data typically DataFrame that contains all the various features used for training
    """
    def __init__(self):
        """
        basic wrapper for handling predictions for models, typically sklearn models
        """

        self._model = None  # typically a sklearn model
        self._scale = StandardScaler()
        self._x = None
        self._y = None
        self._x_names = None
        self._y_names = None  # helpful for deciphering what's going on in terms of last iteration
        self._raw_data = None  # base

    def preprocess(self, x: list, fit_it: bool = False):
        if fit_it:
            self._scale.fit(x)
        return self._scale.transform(x)

    def set_data(self, data: object):
        self._raw_data = data

    def set_model(self, model: object):
        self._model = model

    def set_features(self,
                     x_names: list = ["width_jp", "height_jp", "text_jp_len"],
                     y_names: list = ["font-size_en"]):
        """
        define the features you're going to select, this assumes *optionally* multi-outputs
        Args:
            x_names: column names that are used as input
            y_names: column names that are used as output

        Returns:
            none
        """
        self._x = self._raw_data[x_names].values

        self._y = self._raw_data[y_names].values
        self._y_names = y_names
        self._x_names = x_names
        print(self._y.shape)
        if self._y.shape[1] == 1:
            self._y = self._y.ravel()

    def fit(self, x, y, preprocess: bool = False):
        if preprocess:
            x = self.preprocess(x, True)
        self._model.fit(x, y)

    def predict(self, data: list, preprocess: bool = False) -> list:
        """
            makes general predictions on unprocessed data, make sure to call preprocess
        Args:
            data: a multi dimensional array representing data points, it assumes it hasn't been scaled or processed
            preprocess: should you normalize and scale the data according to how this handles data
        Returns:
            list
        """
        if preprocess:
            data = self.preprocess(data)
        return self._model.predict(data)

    def score_cv(self) -> list:
        """
            get general aggregate scores on dataset for train and dev [[tr],[dev]]
        Returns:
            list
        """

        ss = ShuffleSplit(n_splits=4, test_size=0.25, random_state=0)

        total_scores_tr = []
        total_scores_dev = []
        x = self._x
        y = self._y

        for tr_indices, dev_indices in ss.split(x):
            tr_x = x[tr_indices]
            tr_y = y[tr_indices]
            dev_x = x[dev_indices]
            dev_y = y[dev_indices]

            tr_x = self.preprocess(tr_x, True)
            dev_x = self.preprocess(dev_x)

            self.fit(tr_x, tr_y, False)  # this optionally prepocesses the data

            tr_score = self.score(self.predict(tr_x), tr_y)
            dev_score = self.score(self.predict(dev_x), dev_y)
            total_scores_tr.append(tr_score)
            total_scores_dev.append(dev_score)

        return total_scores_tr, total_scores_dev

    def score(self, predicted, ground_truth) -> float:
        """
            calculates a scoring metric defined by the user
        Args:
            predicted: predicted scores, whatever those may be
            ground_truth: ground truth, but could be a bounding box, or a mask

        Returns:
            float
        """
        return mean_squared_error(predicted, ground_truth, squared=False)


def save(model: FeaturePredictionTraditional, save_path: str):
    """
    wraps the whole file, except for the data itself
    Args:
        save_path: a path to save the file

    Returns:
        None
    """
    x = model._x
    y = model._y
    raw_data = model._raw_data

    model._x = None
    model._y = None
    model._raw_data = None

    joblib.dump(model, save_path)
    model.set_data(raw_data)
    model._x = x
    model._y = y


def load(load_path: str) -> object:
    """
    loads model that is saved, useful for when incorporated into the actual system
    Args:
        load_path: path of the model to be loaded

    Returns:
        object
    """
    return joblib.load(load_path)
