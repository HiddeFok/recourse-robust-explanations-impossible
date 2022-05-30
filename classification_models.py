from typing import Tuple, Callable
from copy import copy

import pandas as pd
import numpy as np
from joblib import dump, load

from matplotlib.image import imsave
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

from data_loader import *


# General functions
def optimal_threshold(data: pd.DataFrame, values: np.array, n: int = 200) -> Tuple[float, float]:
    """
    Finds the optimal threshold, when classifying a picture using the quadratic difference between the
    mean value of the background and the mean value of the person. The labels are chosen in such a way that
    the optimal threshold gives a perfect classifier.
    """
    min_value, max_value = np.min(values), np.max(values)
    thresholds = np.linspace(min_value, max_value, num=n)
    accuracy = 0
    optimal_treshold = 0
    y = data["Label"]
    for threshold in thresholds:
        y_hat = (values >= threshold).astype(int)
        new_accuracy = np.mean(y_hat == y)
        if new_accuracy > accuracy:
            accuracy = new_accuracy
            optimal_treshold = threshold
    return optimal_treshold, accuracy


def predict_threshold(values: np.array, threshold: float) -> np.array:
    """Given a threshold, predict if the given values should be accepted or rejected"""
    y_hat = (values >= threshold).astype(int)
    return y_hat


def evaluation_threshold(data: pd.DataFrame, values: np.array, threshold: float) -> pd.Series:
    """Utility function that displays a classification report for the data and a given threshold value"""
    y = data["Label"]
    y_hat = (values >= threshold).astype(int)
    imgs_wrong = data["Filename"].loc[y_hat != y]
    print(classification_report(y, y_hat, target_names=["rejected", "accepted"]))
    return imgs_wrong


def create_zero_attribution(data: pd.DataFrame, indices: Tuple[np.array, np.array]) -> pd.DataFrame:
    """This function creates two images that have a zero gradient for the quadratic differences classifier. However,
    the person is still noticeable"""
    img_1 = copy(data["img_gray"].iloc[0])
    img_2 = copy(data["img_gray"].iloc[25])

    img_1 = img_1.astype(float)
    img_1 = img_1.astype(float)

    mean_1 = np.mean(img_1[indices[0]])
    mean_2 = np.mean(img_2[indices[0]])

    img_1[indices[0]] = mean_1
    img_2[indices[0]] = mean_2 - 43

    n, m = img_1.shape
    for j in range(m):
        if j < 128:
            img_1[indices[1][:, j], j] = 255 - j * (255 / 128)
            img_2[indices[1][:, j], j] = j * (55 / 128) + 200
        else:
            img_1[indices[1][:, j], j] = j * (255 / 128) - 255
            img_2[indices[1][:, j], j] = 310 - j * (55 / 128)

    mean_new_1 = np.mean(img_1[indices[1]])
    mean_new_2 = np.mean(img_2[indices[1]])

    img_1[indices[1]] = img_1[indices[1]] * (mean_1 / mean_new_1)
    img_2[indices[1]] = img_2[indices[1]] * ((mean_2 - 43) / mean_new_2)

    img_1 = img_1 / 255.
    img_2 = img_2 / 255.

    imsave("pictures/user-icon-zero-attr-0.png", img_1, vmin=0, vmax=1, cmap="gray")
    imsave("pictures/user-icon-zero-attr-1.png", img_2, vmin=0, vmax=1, cmap="gray")

    data = data.append(
        pd.DataFrame({
            "Filename": ["user-icon-zero-attr-0.png", "user-icon-zero-attr-1.png"],
            "Label": [0, 0],
            "img": [img_1, img_2],
            "img_gray": [img_1, img_2]
        }),
        ignore_index=True)

    return data


# quadratic difference
def quadratic_differences(data: pd.DataFrame, indices: Tuple[np.array, np.array]) -> np.array:
    """Calculates the quadratic difference of the mean value of the background and the mean value of the person"""
    imgs = data["img_gray"]
    person_values = imgs.map(lambda x: x[indices[0]])
    background_values = imgs.map(lambda x: x[indices[1]])
    mean_value_person = person_values.apply(np.mean)
    mean_value_background = background_values.apply(np.mean)
    return np.power(mean_value_person - mean_value_background, 2)


def quadratic_diff_proba(imgs: np.array, indices: Tuple[np.array, np.array], threshold: float) -> np.array:
    """For LIME and SHAP, a function that gives the probability per class is needed. The quadratic difference
    classifier is a perfect classifier in this case, by construction. So, only probabilities of 1 and 0 ar given"""
    if len(imgs.shape) == 3:
        # Add batch dimension
        imgs = imgs[np.newaxis, :, :, :]
    if len(imgs.shape) == 2:
        # add batch dimension and color channel dimension
        imgs = imgs[np.newaxis, :, :, np.newaxis]
    person_values = imgs[:, indices[0], :]
    background_values = imgs[:, indices[1], :]
    features = np.power(np.mean(person_values[:, :, 0], axis=1) - np.mean(background_values[:, :, 0], axis=1), 2)
    y_hat_0 = (features < threshold).astype(int)
    y_hat_1 = (features >= threshold).astype(int)
    y_hat = np.array([y_hat_0, y_hat_1]).T

    return y_hat



