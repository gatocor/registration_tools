import os
import re
from collections import Counter
import numpy as np
from skimage.io import imread

# def analyze_folder_pattern(folder_path):
#     """
#     Analyzes the pattern of file names in a given folder and identifies the common pattern,
#     changing values, and non-matching files.

#     Args:
#         folder_path (str): The path to the folder containing the files to be analyzed.

#     Returns:
#         list: A list of patterns with their counts and changing values. Each element in the list
#               is a tuple containing:
#               - pattern (str): The common pattern of the file names.
#               - count (int): The number of files matching this pattern.
#               - changing_values (list): A list of lists containing the changing numeric values in the file names.
#     """
#     files = os.listdir(folder_path)
#     pattern = None
#     changing_values = []
#     non_matching_files = []

#     if not files:
#         return pattern, changing_values, non_matching_files

#     # Extract the common pattern
#     def extract_pattern(file_name):
#         num_counter = 0
#         def replace_with_num(match):
#             nonlocal num_counter
#             replacement = f'{{num{num_counter}:{len(match.group(0))}}}'
#             num_counter += 1
#             return replacement

#         return re.sub(r'\d+', replace_with_num, file_name)

#     # Find the most common pattern
#     patterns = [extract_pattern(file) for file in files]
#     patterns = Counter(patterns).most_common()
#     patterns_list = [[pattern, count, []] for pattern, count in patterns]

#     # Keep only the most frequent patterns
#     max_count = patterns_list[0][1]
#     patterns_list = [pattern_data for pattern_data in patterns_list if pattern_data[1] == max_count]

#     # Identify changing values and non-matching files
#     for file in files:
#         current_pattern = extract_pattern(file)
#         for pattern, count, changing_values in patterns_list:
#             if current_pattern == pattern:
#                 changing_values.append(re.findall(r'\d+', file))
#                 break

#     # Filter out numbers that do not change
#     for pattern_data in patterns_list:
#         pattern, _, values = pattern_data
#         new_values = []
#         if values:
#             values_ = list(zip(*values))
#             for pos, value in enumerate(values_):
#                 if all(v == value[0] for v in value):
#                     pattern = re.sub(r'\{num'+str(pos)+':\d+\}', value[0], pattern, 1)
#                 else:
#                     new_values.append(value)

#         pattern_data[0] = pattern
#         pattern_data[2] = new_values

#     # Convert changing values to leading zeros format
#     for pattern_data in patterns_list:
#         pattern, _, values = pattern_data
#         if values:
#             values_ = list(zip(*values_))
#             conserved_values = [list(set(v)) for v in values]
#             changing_values = [v for v in conserved_values if len(v) > 1]
#             conserved_values = [v[0] for v in conserved_values if len(v) == 1]

#             # Convert changing values to leading zeros format
#             for i in range(len(changing_values)):
#                 max_length = max(len(value) for value in changing_values[i])
#                 changing_values[i] = [value.zfill(max_length) for value in changing_values[i]]

#             # Change {num:\d} to {num:0\d} format
#             pattern_data[0] = re.sub(r'\{num\d+:(\d+)\}', r'{num\1:0\2d}', pattern_data[0])

#         pattern_data[2] = changing_values
#         # Convert changing values to integers, removing leading zeros
#         pattern_data[2] = [sorted([int(value.lstrip('0') or '0') for value in values]) for values in pattern_data[2]]

#     return patterns_list

class Dataset:
    def __init__(self, paths_with_regex, numbers, format):

        """
        Initializes the Dataset object with a list of paths with regex patterns, a list of numbers,
        and a format for the numbers.

        Args:
            paths_with_regex (list): A list of paths with regex patterns.
            numbers (list): A list of numbers to be substituted in the regex paths.
            format (str): The format to be used for the numbers. It can only contain 'C', 'T', 'X', 'Y', 'Z',
                      each representing a dimension in a multidimensional image. Each character must appear only once.
        """
        self.paths_with_regex = paths_with_regex
        self.numbers = [format.format(number) for number in numbers]
        if not all(f in {"X", "T", "Y", "Z", "C"} for f in format):
            raise ValueError("Format can only contain 'X', 'Y', 'Z', 'T', and 'C'")
        if len(set(format)) != len(format):
            raise ValueError("Format can only contain each of 'X', 'Y', 'Z', 'T', and 'C' once")
        self.format = format
        self.check_files_exist()
        if "C" in format and len(paths_with_regex) > 1:
            raise ValueError("If 'C' is in the format, only one path can be provided")
        self.shape, self.ndim = self.check_consistent_shapes()

    def check_files_exist(self):
        """
        Checks if all files exist based on the paths with regex patterns and numbers.

        Returns:
            dict: A dictionary with the paths as keys and a boolean indicating if the file exists as values.
        """
        for path_with_regex in self.paths_with_regex:
            for number in self.numbers:
                path = path_with_regex.format(number)
                if os.path.exists(path):
                    raise FileNotFoundError(f"File not found: {path}")
        return
    
    def check_consistent_shapes(self):
        """
        Checks if all files have consistent shapes and number of dimensions.

        Returns:
            tuple: A tuple containing the shape and number of dimensions if consistent, otherwise raises an error.
        """

        expected_shape = None
        expected_ndim = None

        for path_with_regex in self.paths_with_regex:
            number = self.numbers[0]
            path = path_with_regex.format(number)
            if os.path.exists(path):
                image = imread(path)
                if expected_shape is None:
                    expected_shape = image.shape
                    expected_ndim = image.ndim
                else:
                    if image.shape != expected_shape or image.ndim != expected_ndim:
                        raise ValueError(f"Inconsistent shape or number of dimensions found in file: {path}")

                if len(image.shape) != len(self.format):
                    raise ValueError(f"Number of dimensions in file {path} does not match the length of the format")

        return expected_shape, expected_ndim

    def add_path(self, new_path_with_regex):
        """
        Adds a new path with regex pattern to the list of paths and checks if the files exist.

        Args:
            new_path_with_regex (str): The new path with regex pattern to be added.
        """
        self.paths_with_regex.append(new_path_with_regex)
        self.check_files_exist()
        self.check_consistent_shapes()
        if "C" in format and len(paths_with_regex) > 1:
            raise ValueError("If 'C' is in the format, only one path can be provided")

    def remove_path_by_position(self, position):
        """
        Removes a path with regex pattern from the list of paths by its position.

        Args:
            position (int): The position of the path with regex pattern to be removed.
        """
        if 0 <= position < len(self.paths_with_regex):
            self.paths_with_regex.pop(position)
        else:
            raise IndexError("Position out of range")
        
    def set_numbers(self, new_numbers):
        """
        Sets a new list of numbers to be substituted in the regex paths and checks if the files exist.

        Args:
            new_numbers (list): The new list of numbers to be set.
        """
        self.numbers = new_numbers
        self.check_files_exist()

