from typing import Tuple

import pandas as pd
import numpy as np

from skimage.io import imread
from skimage.color import rgb2gray, rgba2rgb

import matplotlib.pyplot as plt


def load_data() -> pd.DataFrame:
    data = pd.read_csv("Labeling.csv", delimiter=",")
    data["Filename"] = "pictures/" + data["Filename"]
    data["img"] = data["Filename"].apply(imread)
    data["img_gray"] = data["img"].map(lambda x: x[:, :, 0])
    return data


def show_example_img(data: pd.DataFrame) -> None:
    example_img = data.iloc[14]

    original, grayscale = example_img["img"], example_img["img_gray"]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    ax = axes.ravel()

    ax[0].imshow(original)
    ax[0].set_title("Original")
    ax[1].imshow(grayscale, cmap="gray", vmin=0, vmax=255)
    ax[1].set_title("Grayscale")

    fig.tight_layout()
    plt.show()


# Some utility functions
def get_person_indices(data: pd.DataFrame) -> Tuple[np.array, np.array]:
    example_img = data.iloc[0]["img_gray"]
    # person_indices = np.asarray((example_img < 200).nonzero())
    # background_indices = np.asarray((example_img >= 200).nonzero())
    person_indices = example_img < 200
    background_indices = example_img >= 200
    return person_indices, background_indices


def load_pickled_data(fname):
    return pd.read_pickle(f"pickled_data/{fname}")


if __name__ == "__main__":
    data = load_data()
    show_example_img(data)
