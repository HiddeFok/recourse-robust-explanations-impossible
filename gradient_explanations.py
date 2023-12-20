import os

from typing import Tuple, Callable

import numpy as np
import pandas as pd

from matplotlib.pyplot import imsave

from data_loader import *
from classification_models import *


# Utility functions
def bulk_attribution_method(data: pd.DataFrame, func: Callable, **kwargs) -> pd.Series:
    """Utility function that applies a given gradient method to all pictures in a pd.DataFrame column"""
    imgs = data["img_gray"]
    attributions = imgs.map(lambda x: func(x, **kwargs))
    return attributions


#####################
# Vanilla Gradients #
#####################
# Quadratic differences
def grad_qd(img: np.array, indices: Tuple[np.array, np.array]) -> np.array:
    person_values = img[indices[0]]
    background_values = img[indices[1]]
    mean_value_person = person_values.mean()
    mean_value_background = background_values.mean()

    diff = mean_value_background - mean_value_person
    n_per = np.sum(indices[0])
    n_back = np.sum(indices[1])

    n, m = img.shape
    grad = np.zeros((n, m))

    grad[indices[0]] = 2 / n_per * img[indices[0]] * diff
    grad[indices[1]] = -2 / n_back * img[indices[1]] * diff

    return grad


###########################################
# SmoothGrad, can be analytically derived #
###########################################
# which shows that it is equivalent to vanilla gradients
def smoothgrad_qd(img: np.array, indices: Tuple[np.array, np.array]) -> np.array:
    person_values = img[indices[0]]
    background_values = img[indices[1]]
    mean_value_person = person_values.mean()
    mean_value_background = background_values.mean()

    diff = mean_value_background - mean_value_person
    n_per = np.sum(indices[0])
    n_back = np.sum(indices[1])

    n, m = img.shape
    grad = np.zeros((n, m))

    grad[indices[0]] = 2 / n_per * img[indices[0]] * diff
    grad[indices[1]] = -2 / n_back * img[indices[1]] * diff
    return grad


########################
# Integrated Gradients #
########################
# This is a general implementation of IG. For our examples, we can explicitly calculate the attribution, which we do
# later on
def interpolate_img(img: np.array, x_0: np.array, t_path: np.array):
    t_path_x = t_path[:, np.newaxis, np.newaxis]
    x_0_x = np.expand_dims(x_0, axis=0)
    img_x = np.expand_dims(img, axis=0)
    interpolated = x_0_x + (img_x - x_0_x) * t_path_x
    return interpolated


def integrated_gradients(img: np.array, x_0: np.array, func: Callable, num: int = 50, **kwargs):
    t_path = np.linspace(0, 1, num=(num + 1))

    interpolated_imgs = interpolate_img(img, x_0, t_path)

    interpolated_grads = np.zeros((len(t_path), 256, 256))
    for i in range(len(t_path)):
        interpolated_grads[i] = func(interpolated_imgs[i, :, :], **kwargs)

    ig = (img - x_0) * np.mean(interpolated_imgs, axis=0)
    return ig


# Methods that can be analytically derived
def ig_qd(img: np.array, x_0: np.array, indices: Tuple[np.array, np.array]) -> np.array:
    n, m = img.shape
    average = 0.5 * (img + x_0)
    ig = np.zeros((n, m))

    person_values = average[indices[0]]
    background_values = average[indices[1]]
    diff = np.mean(person_values) - np.mean(background_values)

    n_per = np.sum(indices[0])
    n_back = np.sum(indices[1])

    ig[indices[0]] = 2 / n_per * diff
    ig[indices[1]] = -2 / n_back * diff

    return (img - x_0) * ig


def add_hatch_fill(img, full_f_name):
    n, m = img.shape
    x = np.linspace(0, 1, n)
    y = np.linspace(0, -1, m)

    fig, ax = plt.subplots(figsize=(2.56, 2.56))
    result = ax.contourf(
        x, y, 
        img, 
        vmin=-1, vmax=1, 
        cmap='seismic', 
    )
    print(full_f_name)
    for hatch, lvl in zip(result.collections, result.levels):
        if lvl < - 0.01:
            hatch.set_hatch('/')
        elif lvl > 0.01:
            hatch.set_hatch('o')
        print(lvl)
    ax.set_axis_off()
    ax.set_aspect("equal")

    fig.tight_layout(pad=0)
    fig.savefig(full_f_name + ".pdf", pad_inches=0)
    fig.savefig(
        full_f_name + ".png", 
        dpi=600, 
        transparent=False, 
        bbox_inches="tight",
        pad_inches=0
    )
    fig.clear()
    plt.close(fig)
    


def save_results(data: pd.DataFrame, scores: np.array, attributions: pd.Series, subdir: str, fname: str) -> None:
    list_file_names = [0] * len(attributions)

    if not os.path.isdir(subdir):
        os.mkdir(subdir)

    # We normalize all pictures within one classification method and one attribution method
    min_value = np.min(attributions.map(np.min))
    max_value = np.max(attributions.map(np.max))

    abs_max = max(min_value, max_value, key=abs)
    for i in range(len(attributions)):
        img = attributions[i]
        # This scaling ensures that negative attributions stay negative and positive stay positive
        normalized_img = img / abs_max

        full_f_name = f"{subdir}/{fname}_{i+1}"
        list_file_names[i] = full_f_name
        # imsave(full_f_name, normalized_img, vmin=-1, vmax=1, cmap="seismic")
        add_hatch_fill(normalized_img, full_f_name)


    df_export = pd.DataFrame(
        {
            "Attributions": attributions.values,
            "Filenames": list_file_names,
            "Scores": scores,
            "Original": data["Filename"].values,
            "Original img": data["img_gray"].values,
            "Original Label": data["Label"].values
        }
    )

    fname = fname.replace("user_icon", "")
    df_export.to_pickle(f"pickled_data/{fname}results.pickle")


def run_gradients(data: pd.DataFrame, indices: Tuple[np.array, np.array]) -> None:
    print("Working on the gradient based explanations.")
    BASE_IMAGE = np.zeros((256, 256))

    # Quadratic differences method (The labels are chosen in such a way that this will create a perfect fit) #
    differences = quadratic_differences(data, indices)

    # Attributions of quadratic differences method
    grads_qd = bulk_attribution_method(data, grad_qd, indices=indices)
    smoothgrads_qd = bulk_attribution_method(data, smoothgrad_qd, indices=indices)
    integrated_grads_qd = bulk_attribution_method(data, ig_qd, x_0=BASE_IMAGE, indices=indices)

    save_results(data, differences, grads_qd, "vanilla_gradients", "vanilla_gradients_qd_user_icon")
    save_results(data, differences, smoothgrads_qd, "smoothgrad", "smoothgrad_qd_user_icon")
    save_results(data, differences, integrated_grads_qd, "integrated_gradients", "integrated_gradients_qd_user_icon")


if __name__ == "__main__":
    data = load_data()
    indices = get_person_indices(data)
    # Create two images (one with lighter background, one with darker background) with no attribution
    # (ie, the gradient is zero)
    data = create_zero_attribution(data, indices)

    run_gradients(data, indices)
