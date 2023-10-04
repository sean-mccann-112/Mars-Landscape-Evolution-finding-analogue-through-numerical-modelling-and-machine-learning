import pandas as pd
import numpy as np


def read_block_file(file_path: str, column: int = 0):
    # return a numpy array of the full data from a single "block"
    # skip 5 lines which are extra data saved in the file format
    read_file = pd.read_csv(file_path, sep=" ", header=None, skiprows=5).to_numpy()
    return read_file.reshape([251, 201, 500, 3], order='F')


def filter_matrix(matrix, list_of_mappings, remap_values=None):
    # remaps the facies values from the original set, to new values
    if remap_values is None:
        remap_values = range(len(list_of_mappings))

    output = np.zeros_like(matrix)
    for i in range(len(list_of_mappings)):
        for value in list_of_mappings[i]:
            output_entries = np.where(matrix == value, remap_values[i], 0)
            output += output_entries
    return output


def load_block(
    channel_depth=3,  # max channel depth of Flumy simulation, range(3, 18, 3)
    net_gross=10,  # net to gross ratio of FlUMY simulation, range(10, 60, 10)
    seed=1,  # seed of the FLUMY simulation, range(1, 2, 1) (ie just 1)
    select=0,  # initial block location range(0, 5, 1)
    stack_size=1,  # total number of blocks to stack together
    file_location="F:/College_UCC/AM6021- Dissertation/FLUMY Stuff/FLUMY export blocks/",  # file location
    remap_list=([10, 11, 12, 13, 14, 15], [1, 2], [3, 9], [4, 5, 6, 7], [8]),  # grouping facies
    remap_value=(0, 10, 8, 3, 1),  # what those groups are mapped to
):
    # filter and stack multiple blocks according to stack_size
    # temp sanity checker:
    if stack_size + select == 6:
        print("the stack size or start selection is too big, would try and select block", stack_size + select - 1)
        return None

    block_list = []
    for i in range(stack_size):
        file = read_block_file(file_location+f'{channel_depth}_{net_gross}_{seed}_{select+i}')
        block = filter_matrix(matrix=file[..., 0], list_of_mappings=remap_list, remap_values=remap_value)
        block_list.append(block)

    full_block = np.concatenate(block_list, axis=2)
    return full_block[:200, :200, :]  # crop to 200 by 200 by x blocks, where x is the amount of stacks x 500


def erosion(initial_condition, durability_sample, erosion_constant=1.):
    # erode a 2d value map, by 1 step
    """
    given an initial condition, and the durability of the surface of this initial condition,
    subtract the amount eroded in a timestep, then return the output condition
    both the initial and output condition is a 2d depth map
    since it's a depth map subtraction is actually adding to the values
    """
    output_condition = initial_condition + erosion_constant/durability_sample

    return output_condition


def diffusion(initial_condition, durability_sample, diffusion_constant=1.):
    # diffuse a 2d value map, by 1 step
    """
    Assume a point, surrounded by 4 other points, they all have values. the effect of diffusion is to
    move the point towards the average of the collective points, at a rate dependent on how far away
    from the average the point is. 
    """
    u = np.pad(initial_condition, pad_width=1, mode="edge")

    # from above, from below, from left, from right.
    A = u[2:, 1:-1]
    B = u[:-2, 1:-1]
    C = u[1:-1, 2:]
    D = u[1:-1, :-2]
    change = (A + B + C + D - 4 * initial_condition)

    result = initial_condition + np.multiply(change, diffusion_constant / durability_sample)

    return result


def erosion_model(
        durability_block,
        erosion_steps=1000,
        timestep=0.01,
        erosion_diffusion_ratio=1.,
        printout=False
):
    # create the stacked blocks
    depth_map_shape = durability_block.shape[0:2]
    depth_map_timeline = np.zeros([*depth_map_shape, erosion_steps + 1])
    x_coords, y_coords = np.meshgrid(np.arange(depth_map_shape[1]), np.arange(depth_map_shape[0]))

    for i in range(erosion_steps):
        # initial_condition:
        depth_map = depth_map_timeline[..., i]
        # sampled_values:
        sampled_values = durability_block[y_coords, x_coords, depth_map.astype(int)]

        # erode and diffuse per timestep, the erosion and diffusion constant is altered;
        # first to be in a ratio,
        # second to be small enough that the equivalent timestep is relatively small,
        # reducing inaccuracy through the finite difference method
        eroded_map = erosion(initial_condition=depth_map, durability_sample=sampled_values, erosion_constant=timestep)
        diffused_map = diffusion(initial_condition=eroded_map, durability_sample=sampled_values + 2.5, diffusion_constant=timestep/erosion_diffusion_ratio)
        next_layer = np.clip(diffused_map, 0, durability_block.shape[2]-1)

        # writes next layer to timeline
        depth_map_timeline[..., i + 1] = next_layer

        # print out of progress, every 100 steps
        if i % 100 == 99 and printout:
            print(f"{i + 1}/{erosion_steps}", end='\r')

    return depth_map_timeline


def save_depth_map_timeline(depth_map_timeline, file_path, array_name):
    # save the integer values of the array as a numpy file format,
    # there are possibly other file formats that are easier to use/take less space
    np.savez_compressed(file_path, depth_map_timeline.astype(int))


def main(
    file_path,
    erosion_steps=15000,
    timestep=0.1,
    edr=20,
    timestep_skip=100,
):
    """
    :param file_path: str: file path to save
    :param erosion_steps: number of timesteps to compute
    :param timestep: size of a single timestep
    :param edr: erosion/diffusion ratio
    :param timestep_skip: no. of initial timesteps to skip over before saving
    :return:
    """
    block_stacking = 2

    # Training Data ranges
    cd_range = range(3, 18, 3)
    ntg_range = range(10, 60, 10)
    seed_range = range(1, 2, 1)
    selection_range = range(0, 6 - block_stacking, 1)

    # Model tuning ranges (to match earth data)
    # cd_range = range(3, 4, 1)
    # ntg_range = range(20, 21, 1)
    # seed_range = range(1, 2, 1)
    # selection_range = range(0, 6 - block_stacking, 1)

    for cd in cd_range:
        for ntg in ntg_range:
            for seed in seed_range:
                for select in selection_range:

                    if cd != 15 or ntg <=10:
                        continue

                    durability_block = load_block(
                        channel_depth=cd,
                        net_gross=ntg,
                        seed=seed,
                        select=select,
                        stack_size=block_stacking,
                    )
                    """erosion_steps, timestep and erosion_diffusion_ratio
                    need to be tweaked to match real world datapoint"""

                    depth_map_timeline = erosion_model(
                        durability_block=durability_block,
                        erosion_steps=erosion_steps,
                        timestep=timestep,
                        erosion_diffusion_ratio=edr,
                        printout=True,
                    )
                    file_name = f"{cd}_{ntg}_{seed}_{select}_{block_stacking}_{erosion_steps}_{timestep}_{edr}"
                    save_depth_map_timeline(
                        depth_map_timeline=depth_map_timeline[:, :, ::timestep_skip],
                        file_path=file_path + file_name,
                        array_name=file_name,
                    )
                    print("saved file ", file_name)


if __name__ == "__main__":
    edr = 100
    main(
        file_path="F:/College_UCC/AM6021- Dissertation/Depth Map Numpy Files/Simulated data/2_15000_0.2_100_2.5/",
        erosion_steps=15000,
        timestep=0.2,
        edr=edr,
        timestep_skip=1,
    )
