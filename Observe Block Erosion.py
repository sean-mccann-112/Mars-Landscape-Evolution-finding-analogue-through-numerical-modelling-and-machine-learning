import numpy as np
import matplotlib.pyplot as plt
from create_dataset import load_block

# 3d voxel rendering (this is not really working, takes too much )
from simple_3dviz import Mesh
from simple_3dviz.window import show, simple_window
from scipy.ndimage.morphology import binary_erosion

# load the timeline(every 100 slices)
# load the corresponding blocks associated with the specific timeline
# create a color array, of the blocks, the original facies blocks can be discarded.
filepath = "F:/College_UCC/AM6021- Dissertation/Depth Map Numpy Files/test file/"
filename = "3_20_1_1_2_15000_0.4_100.npz"
characteristics = filename.split(sep="_")[0:5]
characteristics = list(map(int, characteristics))


# creates a true/false array based on the inputted depth map
def create_3d_array(depth_map, max_depth):
    shape = (*depth_map.shape, max_depth)
    result = np.indices(shape)[2] > np.reshape(depth_map, (*depth_map.shape, 1))
    return result


# create a colour array of the value array, using the colour dictionary provided
def create_colour_array(value_array, colour_dict):
    colour_array = np.zeros((*value_array.shape, 3))

    for value in colour_dict.keys():
        colour = np.array(colour_dict[value])
        colour_array[value_array == value, :] = colour

    return colour_array


def meshify_3d_model(
    timeline,
    timestep,
    colour_block,
):
    print(timeline[..., timestep].max())
    terrain_block = create_3d_array(timeline[..., timestep], colour_block.shape[2])
    # print(terrain_block.shape)
    m = Mesh.from_voxel_grid(
        terrain_block != binary_erosion(terrain_block),
        colors=np.flip(colour_block, axis=2),
        bbox=[[-0.5, -0.5, -0.2], [0.5, 0.5, 0.2]])

    return m


"""reduced size"""
size = [50, 50, 1000]
# when referencing the durability_block, these are the colours that are produced
corresponding_colours = {
    0: (1, 0, 1),     # magenta
    10: (1, 1, 0),   # yellow
    8: (1, 0.5, 0),  # orange
    3: (0, 1, 0),    # green
    1: (0, 1, 0.5),  # turqoise green
}
colour_array = create_colour_array(
    load_block(
        channel_depth=characteristics[0],
        net_gross=characteristics[1],
        stack_size=characteristics[4]
    )[0:size[0], 0:size[1], 0:size[2]],
    corresponding_colours,
)
print(colour_array.shape)


with np.load(filepath + filename) as a:
    timeline = a["arr_0"][0:size[0], 0:size[1]]  # every 100 slices are loaded into memory
print(timeline.shape)

mesh = meshify_3d_model(timeline=timeline, timestep=80, colour_block=colour_array)
# binary erosion takes 1 "layer" off of all sides,
show(
    mesh,
    size=(512, 512),
    camera_position=(-1, -1, -1),
    light=(-1, -1, -1),
)


