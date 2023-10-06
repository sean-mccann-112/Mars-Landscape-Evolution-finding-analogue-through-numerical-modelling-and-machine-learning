import numpy
import matplotlib.pyplot as plt

import random

# Sample randomized list with distinct elements
randomized_list = [4, 2, 1, 4, 3, 2, 5, 1, 5, 3]

element_indices = {}
for index, element in enumerate(randomized_list):
    element_ = element[:, 0]
    if element_ in element_indices:
        element_indices[element_].append(index)
    else:
        element_indices[element_] = [index]

# Separate the elements into 5 separate lists
separated_lists = []
for element, indices in element_indices.items():
    separated_lists.append((element, indices))

# Sort the separated lists by element for clarity
separated_lists.sort(key=lambda x: x[0])

# Print the separated lists
for element, indices in separated_lists:
    print(f"Element {element} appeared at indices: {indices}")

