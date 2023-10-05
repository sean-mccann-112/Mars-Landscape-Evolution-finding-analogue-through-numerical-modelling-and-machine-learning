import numpy as np
import matplotlib.pyplot as plt
from DEM_Parser import detect_ridge, fast_line_detection, plot_ridges, find_perpendicular_endpoints, ridge_decider, plot_ridge_variable_calc, extract_values_along_line
from scipy import signal


def load_data(filepath, filename):
    array = np.load(file=filepath+filename)
    return array


def ridge_plotting(ar, cs_lines, t, cs_size):
    for k in range(len(np.transpose(cs_lines))):
        line = np.transpose(cs_lines)[k]
        line_values = np.array(extract_values_along_line(ar, *line, cs_size))

        # 0 the line to lowest value (ie since this describes a depth map, it'd be the tallest point in the cross section)
        line_values *= -1
        line_values -= line_values.min()
        max_index = line_values.argmax()

        # shift line_values so the min index = 25 (with padding equal to the first/last value, depending on how we shift it)
        if max_index > cs_size // 2:
            """shift over to left, cut off shift amount of the left side, then add shift amount of padding to right"""
            shift = max_index - cs_size // 2
            new_line_values = list(line_values[shift:])
            block = [line_values[-1]] * shift
            new_line_values += block
            new_line_values = np.array(new_line_values)
        elif max_index < cs_size // 2:
            """shift over to right, cut off shift amount of the right side, then add shift amount of padding to left"""
            shift = cs_size // 2 - max_index
            new_line_values = list(line_values[:-shift])
            block = [line_values[0]] * shift
            block += new_line_values
            new_line_values = np.array(block)
        else:
            """min point is at mid point"""
            new_line_values = np.copy(line_values)

        plt.plot(new_line_values*100, alpha=0.4)
        # plt.plot(line_values)

    plt.show()


def main(path, name, threshold, cross_section_length):
    arr = load_data(filepath=path, filename=name)
    potential_ridge_mask = detect_ridge(arr, sigma=2.0, threshold=threshold)

    """show DEM"""
    plt.imshow(-100*arr, cmap="Greys_r", vmin=-100, vmax=0)
    plt.colorbar()
    plt.show()

    """plot mask"""
    plt.imshow(potential_ridge_mask, cmap="Greys_r", vmin=0, vmax=1)
    plt.show()

    # (x1, y1, x2, y2)
    possible_ridge_lines = np.transpose(fast_line_detection(potential_ridge_mask)[:, 0])

    """DEM possible ridge lines"""
    plt.imshow(-100 * arr, cmap="Greys_r", vmin=-100, vmax=0)
    plt.plot([possible_ridge_lines[0], possible_ridge_lines[2]], [possible_ridge_lines[1], possible_ridge_lines[3]], alpha=1, c="b", label="possible ridge line")
    plt.colorbar()
    plt.show()

    """DEM mask possible ridge lines"""
    plt.imshow(potential_ridge_mask, cmap="Greys_r", vmin=0, vmax=1)
    plt.plot([possible_ridge_lines[0], possible_ridge_lines[2]], [possible_ridge_lines[1], possible_ridge_lines[3]], alpha=1, c="b", label="possible ridge line")
    plt.show()

    """plot ridges with ridge line and cross section"""
    plot_ridges(arr, possible_ridge_lines[:, 4], cross_section_length, mask=False)
    plot_ridges(arr, possible_ridge_lines, cross_section_length, mask=True)

    lines = np.clip(find_perpendicular_endpoints(*possible_ridge_lines, cross_section_length), 0, arr.shape[0] - 1)

    """"""
    ridge_plotting(ar=arr, cs_lines=lines, t=threshold, cs_size=cross_section_length)

    _, highest_ridge, peak_point = ridge_decider(ar=arr, cs_lines=lines, t=threshold, cs_size=cross_section_length, plot_decision=False)
    peak_width_data = signal.peak_widths(x=-highest_ridge, peaks=[int(peak_point)], rel_height=0.9)

    """Plot Ridge Cross section and width/height measurements"""
    plt.plot(-highest_ridge*100, label="ridge contour")
    plt.plot([peak_width_data[2][0], peak_width_data[3][0]], [100*peak_width_data[1][0], 100*peak_width_data[1][0]], label="width measurement")
    plt.plot([peak_point, peak_point], [100*peak_width_data[1][0], 0], label="peak height")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    path = "F:/College_UCC/AM6021- Dissertation/Depth Map Numpy Files/Mars Data/mars0/"
    name = "1_1_20_0.npy"
    main(path=path, name=name, threshold=0.005, cross_section_length=50)
