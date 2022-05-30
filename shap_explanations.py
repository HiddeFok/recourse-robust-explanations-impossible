import os
import json
from typing import List

import numpy as np
import matplotlib.pyplot as plt

import shap

from skimage.color import gray2rgb

from data_loader import *
from classification_models import *
from gradient_explanations import *


def get_max_abs_value(list_shap_values: List) -> float:
    n = len(list_shap_values)
    max_values = [0] * n
    for i in range(n):
        shap_values = list_shap_values[i]
        values = [shap_values.values[..., j] for j in range(shap_values.values.shape[-1])]

        abs_vals = np.stack([np.abs(values[j].sum(-1)) for j in range(len(values))], 0).flatten()
        max_values[i] = np.nanpercentile(abs_vals, 99.9)
    max_val = max(max_values)
    return max_val


def shap_img(data: pd.DataFrame, imgs: np.array, func: Callable, subdir: str, fname: str, **kwargs) -> None:
    class_names = {
        0: "Rejected",
        1: "Accepted"
    }

    if not os.path.isdir(subdir):
        os.mkdir(subdir)

    masker = shap.maskers.Image("inpaint_telea", imgs[0].shape)

    def f(x):
        return func(x, **kwargs)

    explainer = shap.Explainer(f, masker, output_names=["Rejected", "Accepted"])
    scores = func(imgs)
    idx_labels = np.flip(scores.argsort(axis=1), axis=1)
    labels = np.vectorize(class_names.get)(idx_labels)

    n = imgs.shape[0]
    list_shap_values = [0] * n
    for i in range(n):
        print(i)
        shap_values = explainer(
            imgs[i:i+1, :, :, :],
            max_evals=500,
            batch_size=50,
            outputs=shap.Explanation.argsort.flip[:2]
        )
        list_shap_values[i] = shap_values
        shap.image_plot(shap_values, labels=labels[i:i+1, :], show=False)
        plt.savefig(f"{subdir}/{fname}_{i+1}.png")

    abs_val = get_max_abs_value(list_shap_values)

    for i in range(n):
        # Single images
        shap_values = list_shap_values[i]
        values = [shap_values.values[..., j] for j in range(shap_values.values.shape[-1])]

        sv_values_acc = values[0][0].sum(-1)
        sv_values_rej = values[1][0].sum(-1)

        fig, ax = plt.subplots(figsize=(2.56, 2.56))
        ax.axis("off")
        ax.imshow(sv_values_acc / abs_val,
                  cmap="seismic",
                  extent=(0, sv_values_acc.shape[1], sv_values_acc.shape[0], 0),
                  vmin=-1,
                  vmax=1)
        ax.set_aspect("equal")
        fig.tight_layout(pad=0)
        fig.savefig(f"{subdir}/{fname}_accepted_{i+1}.png")

        fig, ax = plt.subplots(figsize=(2.56, 2.56))
        ax.axis("off")
        ax.imshow(sv_values_rej / abs_val,
                  cmap="seismic",
                  extent=(0, sv_values_rej.shape[1], sv_values_rej.shape[0], 0),
                  vmin=-1,
                  vmax=1)
        ax.set_aspect("equal")
        fig.tight_layout(pad=0)
        fig.savefig(f"{subdir}/{fname}_rejected_{i+1}.png")

    df_export = pd.DataFrame(
        {
            "Attribution ": list_shap_values,
            "Scores": list(zip(scores[:, 0], scores[:, 1])),
            "Original": data["Filename"].values,
            "Original img": data["img_gray"].values,
            "Original Label": data["Label"].values
        }
    )

    fname = fname.replace("user_icon", "")
    df_export.to_pickle(f"pickled_data/{fname}results.pickle")


def run_shap(data, indices, threshold):
    print("Working on the SHAP explanations")

    def predict_fn_qd(X, indices=indices, threshold=threshold):
        return quadratic_diff_proba(X, indices=indices, threshold=threshold)

    imgs = data["img_gray"].map(gray2rgb).to_numpy()
    imgs = np.stack(imgs)
    imgs = imgs.astype(float)

    shap_img(data, imgs, predict_fn_qd, "shap_pictures", "shap_qd_user_icon", indices=indices)


if __name__ == "__main__":
    data = load_data()
    indices = get_person_indices(data)
    data = create_zero_attribution(data, indices)
    THRESHOLD = 5961.335512888862

    run_shap(data, indices, THRESHOLD)


