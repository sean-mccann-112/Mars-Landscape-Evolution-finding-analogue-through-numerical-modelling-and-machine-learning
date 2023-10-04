import numpy as np
import pandas as pd
import os


def import_data(
        filenamelist,
        filepath="F:/College_UCC/AM6021- Dissertation/Depth Map Numpy Files/",
        layerresolution: int = 1
):
    """
    There is a LOT of data that's comparitavely useless,
    this function will load the data, which is a compressed 3d array, uncompressing it,
    then remove the unnecessary data(ie, when it's too early, when it's too late).
    It will then have a dataframe, associated with the individual images, with columns for:
    channel depth, net to gross %, seed, origin block, and the timestep, the values for which will be in the title,
    or for the timestep, which slice of the original timeline.
    only the cd and ntg are going to be the variables the AI will try and predict,
    although timestep might be important too, we don't currently know.
    """
    # pass
    labels = pd.DataFrame(columns=["cd", "ntg", "seed", "origin", "block_size", "timestep"])
    datalist = []  # list of arrays to be later stacked and form a larger array, whith a t axis = no. of rows in the dataframe

    for filename in filenamelist:
        print("currently processing", filename)
        with np.load(filepath + filename) as a:
            timeline = a["arr_0"]  # this is a compressed form of possibly multiple arrays, although only one is used
            splitfilename = filename.split("_")

            # find the first element where the depth reached exceeds the original channel depth, this will then be removed
            # find the first element where the max depth is reached, anything above will also be removed
            cd = int(splitfilename[0]) * 10  # this is the channel depth in blocks
            # initialize
            firstlevel = 0
            lastlevel = timeline.shape[2]
            for t in range(0, timeline.shape[2]):
                # this sums all the boolian values in a layer, if greater than 0, then this is the first layer that reached
                # a point deeper than the original channel depth, this value was arbitrary, and could be tweaked.
                if np.sum(np.sum(np.where(timeline[:, :, t] > cd))) > 0:
                    firstlevel = t
                    break
            for t in range(timeline.shape[2]-1, firstlevel-1, -1):
                # this finds when the first layer (counting in reverse), that doesn't have a value that's equal to the max timestep
                maxtimestep = int(splitfilename[4])*500 - 1
                if not np.sum(np.sum(np.where(timeline[:, :, t] >= maxtimestep))):
                    lastlevel = t
                    break

            # create the block of data thats consistant for the currently viewed file
            timestep = list(range(firstlevel, lastlevel + 1, layerresolution))

            constantrow = np.array([int(x) for x in splitfilename[0:5]])
            layercount = len(timestep)
            constblock = np.tile(constantrow, (layercount, 1))
            filenamelabeldataframe = pd.DataFrame(constblock, columns=["cd", "ntg", "seed", "origin", "block_size"])  # .iloc[:, ::layerresolution]

            # print(len(timestep), len(filenamelabeldataframe), lastlevel-firstlevel)
            filenamelabeldataframe['timestep'] = timestep  # add the timestep column

            # add to the data list to be stacked later
            datalist.append(timeline[:, :, firstlevel:lastlevel + 1:layerresolution].astype("uint16"))
            labels = pd.concat((labels, filenamelabeldataframe), ignore_index=True)  #

    # stack all the data into a single array
    data = np.concatenate(datalist, axis=2)

    return data, labels


def file_name_generator(
        block_stacking: int = 1,
        erosion_steps: int = 1000,
        timestep: float = 1.,
        edr: float = 1.,
):
    """
    :param block_stacking: number of FLumy blocks that were stacked
    :param erosion_steps: amount of timesteps
    :param timestep: timestep size
    :param edr: erosion/diffusion ratio
    :return:
    """
    filenamelist = []
    # various ranges of generated data (this will be consistant, since they originate from the FLUMY simulations)
    cdrange = range(3, 18, 3)
    ntgrange = range(10, 60, 10)
    seedrange = range(1, 2, 1)
    selectrange = range(0, 6-block_stacking, 1)
    for cd in cdrange:
        for ntg in ntgrange:
            for seed in seedrange:
                for select in selectrange:
                    filename = f"{cd}_{ntg}_{seed}_{select}_{block_stacking}_{erosion_steps}_{timestep}_{edr}.npz"
                    filenamelist.append(filename)
    return filenamelist


def main():
    filenamelist = os.listdir("F:/College_UCC/AM6021- Dissertation/Depth Map Numpy Files/test file/")
    print(filenamelist)
    # filenamelist = file_name_generator(2, 1500, 1, 150)
    # fulldata, full_labels = import_data(filenamelist, layerresolution=4)
    # full_labels.to_csv("F:/College_UCC/AM6021- Dissertation/Depth Map Numpy Files/label dataframe.csv", index=False)
    # np.savez_compressed("F:/College_UCC/AM6021- Dissertation/Depth Map Numpy Files/2_1500_1_150_fulldata.npz", fulldata)


if __name__ == "__main__":
    main()
