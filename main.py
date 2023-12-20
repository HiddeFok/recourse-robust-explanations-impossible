"""
Main file that runs all experiments. Depending on which parameters are called it will create all images that
can be found in the paper 'Attribution based explanations cannot be robust and recourse sensitive'.
"""
import os
import argparse

from typing import Literal

from data_loader import *
from classification_models import *
from colorbar import *

from shap_explanations import run_shap
from lime_explanations import run_lime
from gradient_explanations import run_gradients

YesNo = Literal["y", "n"]
THRESHOLD = 5961.335512888862


def run_experiments(gradient: YesNo, lime: YesNo, shap: YesNo) -> None:

    data = load_data()
    indices = get_person_indices(data)
    data = create_zero_attribution(data, indices)
    
    # Only for testing now
    # data = data.iloc[:5]

    # Quadratic differences method (The labels are chosen in such a way that this will create a perfect fit) #
    differences = quadratic_differences(data, indices)
    _ = evaluation_threshold(data, differences, THRESHOLD)

    if not os.path.isdir("pickled_data"):
        os.mkdir("pickled_data")

    if gradient == "y":
        run_gradients(data, indices)
    if lime == "y":
        run_lime(data, indices, THRESHOLD)
    if shap == "y":
        run_shap(data, indices, THRESHOLD)

    create_colorbar()


def main():
    parser = argparse.ArgumentParser(
        description="Indicate which parts of the experiment you want to run, default is all"
    )

    parser.add_argument('--gradient', '-g', choices=["y", "n"], default="y",
                        help="Create the gradient based explanations or not.")
    parser.add_argument('--lime', '-l', choices=["y", "n"], default="y",
                        help="Create the LIME explanations or not.")
    parser.add_argument('--shap', '-s', choices=["y", "n"], default="y",
                        help="Create the SHAP explanations or not.")

    args = parser.parse_args()
    kwargs = vars(args)

    run_experiments(**kwargs)


if __name__ == '__main__':
    main()
