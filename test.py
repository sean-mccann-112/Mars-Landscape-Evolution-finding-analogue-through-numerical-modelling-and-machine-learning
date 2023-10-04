import numpy
import matplotlib.pyplot as plt

path = "C:/Users/User/Documents/Leaf Blower images/leaf blower teaser 24.png"
path2 = "C:/Users/User/Documents/Leaf Blower images/leaf blower teaser 24_record.png"
img = plt.imread(path)
plt.imsave(img, path2)

