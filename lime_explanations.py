import os
import time
import json
from typing import Callable, Optional, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imsave

from lime import lime_image
from skimage.segmentation import mark_boundaries, quickshift
from skimage.color import gray2rgb

from data_loader import *
from classification_models import *


def get_lime_explanations(data: pd.DataFrame, predict_fn: Callable, segmentation_fn: Callable) -> Tuple[np.array, List]:
    """Generates the lime explanations for all pictures"""
    explanations_self_segmented = [0] * len(data)
    explanations_auto_segmented = [0] * len(data)
    scores = [0] * len(data)
    explainer = lime_image.LimeImageExplainer()

    for i, img in enumerate(data["img_gray"].values):
        scores[i] = predict_fn(img)
        lime_explanation_self = explainer.explain_instance(
            img,
            predict_fn,
            hide_color=0,
            top_labels=2,
            num_samples=1000,
            segmentation_fn=segmentation_fn
        )
        lime_explanation_auto = explainer.explain_instance(
            img,
            predict_fn,
            hide_color=0,
            top_labels=2,
            num_samples=1000
        )

        explanations_self_segmented[i] = lime_explanation_self
        explanations_auto_segmented[i] = lime_explanation_auto

    explanations = np.array([explanations_self_segmented, explanations_auto_segmented]).T

    return explanations, scores


def create_subplots_figure(
        exp_self: lime_image.ImageExplanation,
        exp_auto: lime_image.ImageExplanation,
        fname: str) -> None:
    """Creates a figure with the LIME explanations for the auto segmented and self segmented methods. Shows
    the attributions for the accepted and rejected class (these are the negatives of each other)"""
    temp_self_rej, mask_self_rej = exp_self.get_image_and_mask(
        0, positive_only=True, negative_only=False, num_features=2, hide_rest=True
    )
    temp_auto_rej, mask_auto_rej = exp_auto.get_image_and_mask(
        0, positive_only=True, negative_only=False, num_features=1000, hide_rest=True
    )

    temp_self_acc, mask_self_acc = exp_self.get_image_and_mask(
        1, positive_only=True, negative_only=False, num_features=2, hide_rest=True
    )
    temp_auto_acc, mask_auto_acc = exp_auto.get_image_and_mask(
        1, positive_only=True, negative_only=False, num_features=1000, hide_rest=True
    )
    fig, axs = plt.subplots(2, 2)

    axs[0, 0].imshow(mark_boundaries(temp_self_rej / 2 + 0.5, mask_self_rej))
    axs[0, 0].set_title(f"Rejected self segmented")
    axs[0, 0].xaxis.set_visible(False)
    axs[0, 0].yaxis.set_visible(False)
    axs[0, 1].imshow(mark_boundaries(temp_auto_rej / 2 + 0.5, mask_auto_rej))
    axs[0, 1].set_title("Rejected auto segmented")
    axs[0, 1].xaxis.set_visible(False)
    axs[0, 1].yaxis.set_visible(False)
    axs[1, 0].imshow(mark_boundaries(temp_self_acc / 2 + 0.5, mask_self_acc))
    axs[1, 0].set_title("Accepted self segmented")
    axs[1, 0].xaxis.set_visible(False)
    axs[1, 0].yaxis.set_visible(False)
    axs[1, 1].imshow(mark_boundaries(temp_auto_acc / 2 + 0.5, mask_auto_acc))
    axs[1, 1].set_title("Accepted auto segmented")
    axs[1, 1].xaxis.set_visible(False)
    axs[1, 1].yaxis.set_visible(False)

    fig.suptitle(f"Coef = {exp_self.local_exp[0]}")
    fig.savefig(fname)


def create_single_image(
        exp_self: lime_image.ImageExplanation,
        exp_auto: lime_image.ImageExplanation,
        subdir: str,
        fname: str) -> None:
    """Creates a single figure for both the self and auto segmented LIME explanations"""
    dict_heatmap_self_acc = dict(exp_self.local_exp[1])
    dict_heatmap_auto_acc = dict(exp_auto.local_exp[1])
    heatmap_self = np.vectorize(dict_heatmap_self_acc.get)(exp_self.segments)
    heatmap_auto = np.vectorize(dict_heatmap_auto_acc.get)(exp_auto.segments)

    plt.imsave(f"{subdir}/self_{fname}", heatmap_self, cmap="seismic", vmin=-1, vmax=1)
    plt.imsave(f"{subdir}/auto_{fname}", heatmap_auto, cmap="seismic", vmin=-1, vmax=1)


def save_results(
        data: pd.DataFrame,
        scores: np.array,
        attributions: pd.Series,
        subdir: str,
        fname: str,
        qd_diff: Optional = None) -> None:
    list_file_names = [0] * len(scores)

    if not os.path.isdir(subdir):
        os.mkdir(subdir)

    for i in range(len(scores)):
        full_f_name = f"{subdir}/{fname}_{i+1}.png"
        list_file_names[i] = full_f_name

        exp_self, exp_auto = attributions[i, :]
        create_subplots_figure(exp_self, exp_auto, full_f_name)
        create_single_image(exp_self, exp_auto, subdir, f"{fname}_{i+1}.png")

    if qd_diff is not None:
        df_export = pd.DataFrame(
            {
                "Attribution self segmented": attributions[:, 0],
                "Attribution auto segmented": attributions[:, 1],
                "Scores": scores,
                "Quadratic differences": qd_diff,
                "Original": data["Filename"].values,
                "Original img": data["img_gray"].values,
                "Original Label": data["Label"].values
            }
        )
    else:
        df_export = pd.DataFrame(
            {
                "Attribution self segmented": attributions[:, 0],
                "Attribution auto segmented": attributions[:, 1],
                "Scores": scores,
                "Original": data["Filename"].values,
                "Original img": data["img_gray"].values,
                "Original Label": data["Label"].values
            }
        )

    fname = fname.replace("user_icon", "")
    df_export.to_pickle(f"pickled_data/{fname}results.pickle")


def run_lime(data, indices, threshold):
    print("Working on the LIME explanations")

    # The 2 superpixels are the background and the person
    # The row indices were found through inspection of the row indices of the person indices. Where there was a jump
    # indicated where the head ended.
    segments = indices[0].astype(int)

    def predict_fn_qd(X, indices=indices, threshold=threshold):
        return quadratic_diff_proba(X, indices=indices, threshold=threshold)

    def segmentation_fn(img, segments=segments):
        return segments

    qd_lime_explanations, qd_scores = get_lime_explanations(data, predict_fn_qd, segmentation_fn)
    qd_diff = quadratic_differences(data, indices)
    save_results(data, qd_scores, qd_lime_explanations, "lime_pictures", "lime_qd_user_icon", qd_diff)


if __name__ == "__main__":
    data = load_data()
    indices = get_person_indices(data)
    data = create_zero_attribution(data, indices)
    THRESHOLD = 5961.335512888862

    run_lime(data, indices, THRESHOLD)


