import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

filepath = "F:/College_UCC/AM6021- Dissertation/Depth Map Numpy Files/Simulated data/2_15000_0.2_100_2.5/"
filename = "12_10_1_1_2_15000_0.2_100.npz"

with np.load(filepath + filename) as a:
    timeline = a["arr_0"]  # this is a compressed form of possibly multiple arrays, although only one is used
# splitfilename = filename.split("_")

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True
fig, ax = plt.subplots()

# fig.colorbar(cm.ScalarMappable(norm=None, cmap="Greys"), ax=ax)
image = timeline[:, :, 0]
img = ax.imshow(image, cmap="Greys", vmin=0, vmax=1000)
cbar = fig.colorbar(img, ax=ax, extend='both')

axcolor = 'yellow'
ax_slider = plt.axes([0.20, 0.01, 0.65, 0.03], facecolor=axcolor)
slider = Slider(ax_slider, 'Slide->', 0, timeline.shape[2], valinit=0, valstep=1)


def update(val):
    ax.imshow(timeline[:, :, val], cmap="Greys", vmin=0, vmax=1000)
    fig.canvas.draw_idle()


slider.on_changed(update)
plt.show()


def main():
    pass


if __name__ == "__main__":
    main()
