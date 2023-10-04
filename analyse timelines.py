import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from DEM_Parser import ridge_decider, detect_ridge, fast_line_detection, find_perpendicular_endpoints, plot_ridge_variable_calc, plot_ridges
from scipy import signal


def generate_image_data(root, save_path, layerresolution, cross_section_size, threshold, image_size, plot_image=False):
    filenamelist = [f for f in os.listdir(root) if f.endswith('.npz')]
    df = pd.DataFrame(columns=["File_Name", "Ridge_Height", "Ridge_Width", "True_cd", "True_ntg"])
    file_name_list = []

    for filename in filenamelist:
        # if filename != "3_20_1_1_2_15000_0.2_100.npz":
        #     continue

        with np.load(root + filename) as a:
            timeline = a["arr_0"]
        # sample timeline by a resolution, higher resolution, more skipped.
        timeline_block = timeline[:, :, ::layerresolution]

        if plot_image:
            fig, axes = plt.subplots(nrows=1, ncols=2)
            axes[0].title.set_text("ridge height")
            axes[1].title.set_text("ridge width")
            ridge_h_list = []
            ridge_w_list = []

        for i in range(timeline_block.shape[2]):
            array = timeline_block[..., i] / 1000  # divide by max unit count to get values between 0 and 1
            # if the erosion model reached the bottom of the block: don't process/save it
            if array.max() >= 999:
                continue

            array -= array.min()
            image = detect_ridge(array, sigma=2.0, threshold=0.01)

            potential = fast_line_detection(image)
            if potential is None:
                continue

            possible_ridge_lines = np.transpose(potential[:, 0])
            lines = np.clip(find_perpendicular_endpoints(*possible_ridge_lines, cross_section_size), 0, image_size - 1)
            ridge_bool, highest_ridge, peak_point = ridge_decider(ar=array, cs_lines=lines, t=threshold,
                                                                  cs_size=cross_section_size, plot_decision=False)

            if ridge_bool:
                # determine ridge height/width, and if possible, edge curve
                peak_w = signal.peak_widths(x=-highest_ridge, peaks=[peak_point], rel_height=0.9)
                # peak_w: (width, height the width was measured at, left intesection point, right intersetion point)

                # if i >= 100:
                #     plt.title(filename+f" at timestep: {i}")
                #     plot_ridges(array, possible_ridge_lines, cross_section_size, mask=False)
                #     plot_ridge_variable_calc(highest_ridge, peak_point)

                # define the variables
                ridge_width = peak_w[0][0]
                ridge_height = -peak_w[1][0]

                if plot_image:
                    ridge_w_list.append(ridge_width)
                    ridge_h_list.append(ridge_height)

                name_list = filename.split("_")
                given_cd = int(name_list[0])
                given_ntg = int(name_list[1])

                # generate unique file name
                file_name = filename[:-4] + f"_{i}.npy"
                file_name_list.append(file_name)

                # dd new row to dataframe
                new_row = {"File_Name": file_name, "Ridge_Height": ridge_height, "Ridge_Width": ridge_width,
                           "True_cd": given_cd, "True_ntg": given_ntg}
                df.loc[len(df)] = new_row

                # save image
                np.save(save_path + file_name, array)
        if plot_image:
            axes[0].plot(ridge_h_list)
            axes[1].plot(ridge_w_list)

            plt.suptitle(f"Ridge evolution {filename}")
            plt.show()

    # save name list
    label_txt = "\n".join(file_name_list)
    with open(save_path + "name_list.txt", "w") as file:
        file.write(label_txt)

    # save dataframe
    df.to_csv(save_path + "data_frame.csv")
    pass


def main(
    root="F:/College_UCC/AM6021- Dissertation/Depth Map Numpy Files/test file/",
    save_path="F:/College_UCC/AM6021- Dissertation/Depth Map Numpy Files/test file/temp/",
    layerresolution=100,
    cross_section_size=50,
    threshold=0.1,
    image_size=200,
):
    generate_image_data(
        root=root,
        save_path=save_path,
        layerresolution=layerresolution,
        cross_section_size=cross_section_size,
        threshold=threshold,
        image_size=image_size,
        plot_image=False
    )


if __name__ == "__main__":
    main(
        root="F:/College_UCC/AM6021- Dissertation/Depth Map Numpy Files/Simulated data/2_15000_0.2_100_2.5/",
        save_path="F:/College_UCC/AM6021- Dissertation/Depth Map Numpy Files/Simulated data/2_15000_0.2_100_2.5/temp/",
        layerresolution=100,
        cross_section_size=50,
        threshold=0.1,
        image_size=200,
    )



