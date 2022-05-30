import matplotlib.pyplot as plt
import matplotlib
import numpy as np


def create_colorbar():
    matplotlib.rcParams.update({'font.size': 14})

    a = np.array([[-1, 1]])

    plt.figure(figsize=(1.5, 9))
    img = plt.imshow(a, cmap="seismic")
    plt.gca().set_visible(False)
    cax = plt.axes([0.1, 0.05, 0.4, 0.9])
    plt.colorbar(cax=cax)

    plt.savefig("pictures/colorbar.png")
    plt.savefig("pictures/colorbar.pdf")


if __name__ == "__main__":
    create_colorbar()