import rasterio
import numpy as np
import os
import matplotlib.pyplot as plt
from rasterio.windows import Window

mars_file = 'C:/Users/User/Downloads/Mars_MGS_MOLA_DEM_mosaic_global_463m.tif'
with rasterio.open(mars_file) as f:
    w = f.read(1, window=Window(41000, 11000, 2500, 1500))

plt.imshow(w, cmap="Greys_r")
plt.colorbar()
plt.show()
