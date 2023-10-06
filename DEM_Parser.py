import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from geotiff import GeoTiff
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from cv2 import ximgproc
from scipy import signal

from PIL import Image


def fast_line_detection(img):
    image = img.astype('uint8') * 255
    # Create default Fast Line Detector class
    fld = ximgproc.createFastLineDetector(distance_threshold=15)
    # Get line vectors from the image format:(x1, y1, x2, y2)
    lines = fld.detect(image)
    if lines is None:
        return None
    # sqrt((x2-x1)^2 + (y2-y1)^2)
    distances = np.sqrt(
        (lines[:, 0, 2] - lines[:, 0, 0]) ** 2 + (lines[:, 0, 3] - lines[:, 0, 1]) ** 2
    )
    # top 5 longest lines
    count = 5
    if len(lines) >= count:
        selected_lines = lines[np.argpartition(distances, -count)[-count:]]
    else:
        selected_lines = np.copy(lines)
    return selected_lines


def find_perpendicular_endpoints(x1, y1, x2, y2, ln):
    # Calculate the midpoint of the given line segment
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2

    # Calculate the direction vector of the given line segment
    direction_x = x2 - x1
    direction_y = y2 - y1

    # Rotate the direction vector by 90 degrees counterclockwise
    perpendicular_x = -direction_y
    perpendicular_y = direction_x

    # Normalize the perpendicular vector
    length = np.sqrt(perpendicular_x ** 2 + perpendicular_y ** 2)
    normalized_perpendicular_x = perpendicular_x / length
    normalized_perpendicular_y = perpendicular_y / length

    # Calculate the endpoints of the perpendicular line segment
    x3 = mid_x + (ln / 2) * normalized_perpendicular_x
    y3 = mid_y + (ln / 2) * normalized_perpendicular_y
    x4 = mid_x - (ln / 2) * normalized_perpendicular_x
    y4 = mid_y - (ln / 2) * normalized_perpendicular_y

    return x3, y3, x4, y4


def extract_values_along_line(image, x1, y1, x2, y2, n):
    """
    :param image: numpy array to sample, across line
    :param x1: point 1
    :param y1: point 1
    :param x2: point 2
    :param y2: point 2
    :param n: number of samples
    :return: list: List of pixel values sampled along the line.
    """
    x_coords = np.linspace(x1, x2, n, dtype=int)
    y_coords = np.linspace(y1, y2, n, dtype=int)

    line_values = [image[y, x] for x, y in zip(x_coords, y_coords)]

    return line_values


def process_data(array):
    # since the max value in simulation was 100m (or 1000 units of 0.1m),
    # the DEM output is also defined between 0 and 1, where 1 = 100m
    output = array.max() - array
    return output / 100


def save_image(array, file_name):
    plt.imsave(file_name, array, cmap="gray_r")


def detect_ridge(gray_img, sigma=3.0, threshold=0.4):
    # uses hessian matrix(second order derivitive of image):
    # hessian matrix eigenvalues are the ridges
    h_elems = hessian_matrix(gray_img, sigma=sigma, order='rc', use_gaussian_derivatives=False)
    maxima_ridges, _ = hessian_matrix_eigvals(h_elems)
    arr = np.where(maxima_ridges > threshold, 1, 0)
    return arr


def load_tif(tiffilename):
    img = Image.open(tiffilename)
    # array = np.array(img)[400:10400:10, 2000:5000:10]
    # array = np.array(img)[8000:18000:10, 3000:5500:10]
    plt.imshow(array, cmap="Greys_r")
    plt.colorbar()
    plt.show()
    # array = np.array(img.read())[9:-9, 3:-4]
    return array


def plot_overview(array, mask=False):
    """generate overview, with possibility of mask for of possible ridges"""
    ar = np.copy(array)
    if mask:
        ar = detect_ridge(process_data(ar), sigma=2.0, threshold=0.004)
    img = plt.imshow(ar, cmap="Greys_r")  # sub ridges with array to get overview dem map image
    plt.colorbar(img)
    plt.show()


def plot_ridges(img, ridge_lines, cs_size, mask=False):
    """plot image"""
    cmap = "Greys"
    if mask:
        img = detect_ridge(img, sigma=2.0, threshold=0.004)
        cmap = "Greys_r"
    cross_lines = np.clip(find_perpendicular_endpoints(*ridge_lines, cs_size), 0, img.shape[0] - 1)
    plt.imshow(img, cmap=cmap, vmin=0, vmax=1)
    plt.plot([cross_lines[0], cross_lines[2]], [cross_lines[1], cross_lines[3]], c="r")
    plt.plot([ridge_lines[0], ridge_lines[2]], [ridge_lines[1], ridge_lines[3]], alpha=1, c="b")
    plt.plot([0, 0], [0, 0], c="r", alpha=1, label="cross section")
    plt.plot([0, 0], [0, 0], c="b", alpha=1, label="possible ridge")
    plt.colorbar()
    plt.legend()
    plt.show()


def plot_ridge_variable_calc(ridge, point):
    """Image that shows the measurement for the ridge width/height, using cross section"""
    peak_width_data = signal.peak_widths(x=-ridge, peaks=[point], rel_height=0.9)
    plt.plot(ridge, label="ridge contour")
    plt.plot([peak_width_data[2][0], peak_width_data[3][0]], [-peak_width_data[1][0], -peak_width_data[1][0]],
             label="width measurement")  # x values of point, y value of point
    plt.plot([point, point], [-peak_width_data[1][0], 0], label="peak height")
    plt.legend()
    plt.show()


def ridge_decider(ar, cs_lines, t, cs_size, plot_decision=False):
    """initial conditions for loop through cs_lines"""
    ridge_bool = False
    ridge_h = 0
    highest_ridge = None
    peak_point = 0

    if plot_decision:
        plt.title("Decision Gaph")

    for k in range(len(np.transpose(cs_lines))):
        line = np.transpose(cs_lines)[k]
        line_values = np.array(extract_values_along_line(ar, *line, cs_size))

        # 0 the line to lowest value (ie since this describes a depth map, it'd be the tallest point in the cross section)
        line_values -= line_values.min()
        min_index = line_values.argmin()

        # shift line_values so the min index = 25 (with padding equal to the first/last value, depending on how we shift it)
        if min_index > cs_size // 2:
            """shift over to left, cut off shift amount of the left side, then add shift amount of padding to right"""
            shift = min_index - cs_size // 2
            new_line_values = list(line_values[shift:])
            block = [line_values[-1]] * shift
            new_line_values += block
            new_line_values = np.array(new_line_values)
        elif min_index < cs_size // 2:
            """shift over to right, cut off shift amount of the right side, then add shift amount of padding to left"""
            shift = cs_size // 2 - min_index
            new_line_values = list(line_values[:-shift])
            block = [line_values[0]] * shift
            block += new_line_values
            new_line_values = np.array(block)
        else:
            """min point is at mid point"""
            new_line_values = np.copy(line_values)

        peak_width_data = signal.peak_widths(x=-new_line_values, peaks=[min_index], rel_height=0.9)
        peak_height = peak_width_data[1][0]
        print(peak_height, new_line_values.max())
        if line_values[np.clip(min_index - 10, 0, len(line_values) - 1)] > t and line_values[np.clip(min_index + 10, 0, len(line_values) - 1)] > t:
            ridge_bool = True
            """if ridge is true, set initial condtion to new ridge values"""

            if ridge_h < new_line_values.max():
                ridge_h = new_line_values.max()
                highest_ridge = np.copy(line_values)
                peak_point = min_index
            if plot_decision:  # plot section
                plt.plot(new_line_values, label="True Ridge", alpha=0.4)

        else:
            if plot_decision:  # plot section
                plt.plot(new_line_values, label="False Ridge", alpha=0.4, ls="--")
            pass
    if plot_decision:  # plot section
        plt.legend()
        plt.show()

    return ridge_bool, highest_ridge, peak_point


def main(tif_file_name, save_path, stride, cross_section_size, threshold, given_cd, given_ntg, image_size,
         plot_all_image=False):
    """
    :param tif_file_name: path to TIF file for GeoTiff to work with
    :param save_path: path to save the images to
    :param stride: step size for subsectioning image
    :param cross_section_size: length of cross section sampling, in units (so x10 in meters)
    :param threshold: determines whether a ridge occurs, by checking this length on each side of the ridge point,
        and if both sides height compared to the max point exceeds this value, the ridge is considered "True"
    :param given_cd: given values of the channel depth of paleo-river, from Literature
    :param given_ntg: given values of the net to gross % of paleo-river, from Literature
    :param image_size: size of the images to process/save
    :param plot_all_image: a bool that decides whether to use all the plot image functions
    :return: None
    """
    array = load_tif(tif_file_name)
    if plot_all_image:  # plot overview images
        plot_overview(array, mask=False)

    df = pd.DataFrame(columns=["File_Name", "Ridge_Height", "Ridge_Width", "True_cd", "True_ntg"])
    file_name_list = []

    for i in range(0, (array.shape[0] - image_size) // stride - 1):
        for j in range(0, (array.shape[1] - image_size) // stride - 1):

            # 200, 200 image
            strided_image = array[i * stride:i * stride + image_size, j * stride:j * stride + image_size]
            fixed_image = process_data(strided_image)
            image = detect_ridge(fixed_image, sigma=2.0, threshold=0.004)

            # (x1, y1, x2, y2)
            possible_ridge_lines = np.transpose(fast_line_detection(image)[:, 0])  # [:, 0]

            if plot_all_image:
                plot_ridges(fixed_image, possible_ridge_lines, cross_section_size, mask=False)

            lines = np.clip(find_perpendicular_endpoints(*possible_ridge_lines, cross_section_size), 0, image_size - 1)
            ridge_bool, highest_ridge, peak_point = ridge_decider(ar=fixed_image, cs_lines=lines, t=threshold,
                                                                  cs_size=cross_section_size, plot_decision=False)

            if ridge_bool:
                # determine ridge height/width, and if possible, edge curve
                peak_w = signal.peak_widths(x=-highest_ridge, peaks=[peak_point], rel_height=0.9)
                # peak_w: (width, height the width was measured at, left intesection point, right intersetion point)

                if plot_all_image:
                    plot_ridge_variable_calc(highest_ridge, peak_point)

                # define the variables
                ridge_width = peak_w[0][0]
                ridge_height = -peak_w[1][0]

                # generate unique file name
                file_name = f"{given_cd}_{given_ntg}_{i}_{j}.npy"
                file_name_list.append(file_name)

                # dd new row to dataframe
                new_row = {"File_Name": file_name, "Ridge_Height": ridge_height, "Ridge_Width": ridge_width,
                           "True_cd": given_cd, "True_ntg": given_ntg}
                df.loc[len(df)] = new_row

                # save image
                np.save(save_path + file_name, fixed_image)


if __name__ == "__main__":
    main(
        tif_file_name='F:/College_UCC/AM6021- Dissertation/Tif Files/Mars DEM Geotiff.tif',
        save_path="F:/College_UCC/AM6021- Dissertation/Depth Map Numpy Files/Mars data/mars1/",
        stride=25,
        cross_section_size=50,
        threshold=0.1,
        given_cd=1,
        given_ntg=1,
        image_size=200,
        plot_all_image=False,
    )
