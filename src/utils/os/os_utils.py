"""OS utilities for file path handling and directory operations."""

import glob
import json
import os
import re
import shutil
from pathlib import Path

from src.utils.logging.logging_manager import get_logger

logger = get_logger("os_utils")


def get_file_paths(directory, ext=".json"):
    """
    Processes the given directory and returns a list of MIDI files.

    :param  str midi_path: the path to the MIDI file to process
            str ext: file extension to look for
    :return: the list containing the path to the MIDI files
    """
    file_paths = []

    # Walk through all files in the directory that includes the sub-directories as well
    for dirpath, _dirnames, filenames in os.walk(directory):
        for filename in [f for f in filenames if f.endswith(ext) or f.endswith(ext)]:
            # Add the file path in the list
            file_paths.append(os.path.join(dirpath, filename))

    return file_paths


def get_parent_filename_without_ext(filepath):
    """
    Takes in a file path, and returns the base filename without the extension,
    converted to lowercase, and with spaces replaced by underscores.

    :params:str: The full path of the file.

    :return:str: The parent directory and filename without extension, in lowercase,
                  and with spaces replaced by underscores.
    """

    # Use os.path to get the filename and split the extension
    base_name = os.path.basename(filepath)
    # Get the parent directory
    parent_dir = os.path.dirname(filepath).split("/")[-1]

    filename_without_ext = os.path.splitext(base_name)[0]

    # Convert to lowercase and replace spaces with underscores
    filename_without_ext = filename_without_ext.lower().replace(" ", "_")

    return parent_dir, filename_without_ext


def create_directory_if_not_exists(dir_path):
    try:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            logger.info("Directory {} created.".format(dir_path))
        else:
            logger.info("Directory {} already exists.".format(dir_path))
    except PermissionError:
        logger.error(
            "Permission denied: Could not create directory at {}".format(dir_path)
        )


def check_error_code(error, code):
    """
    This function checks if the received error message contains a specific error code.

    Parameters:
    error (Exception): The exception object
    code (int): The error code to look for

    Returns:
    bool: True if the error message contains the specific error code, False otherwise
    """
    # If the error is an OSError, directly check the errno attribute
    if isinstance(error, OSError):
        return error.errno == code

    # For other error types, check if the error message contains the code
    return str(code) in str(error)


def write_to_json_file(file_path, data):
    """
    Write new data to the existing JSON file.

    This function reads the existing JSON data from the specified file, appends or
    modifies it with the provided new data, and writes the updated data back to the
    same JSON file.

    :params:
        file_path (str): The path to the JSON file to be updated.
        new_data (dict): The new data to be added to the JSON file. It should be a
                         dictionary representing the data to be appended to the file.
    """
    try:
        # Check if the file exists
        if not os.path.exists(file_path):
            logger.info("Creating a new Json file to write the file.")
            # If the file doesn't exist, create a new one with the provided data
            with open(file_path, "w") as json_file:
                json.dump(data, json_file, indent=4)

        else:
            logger.info("Adding data to the current Json file.")
            # If the file exists, read the existing JSON data
            with open(file_path, "r") as json_file:
                if json_file.read().strip() == "":
                    _data = []  # Initialize an empty list for data if the file is empty
                else:
                    json_file.seek(0)  # Reset the file pointer to the beginning
                    _data = json.load(json_file)

            # Append or modify the data as needed
            # For example, let's say new_data is a dictionary, and we want to add it to the existing data.
            _data.append(data)

            # Step 3: Write the updated data back to the same JSON file
            with open(file_path, "w") as json_file:
                json.dump(_data, json_file, indent=4)

        logger.info("Data saved successfully in {}".format(file_path))

    except FileNotFoundError:
        logger.error("File '{}' not found.".format(file_path))
    except json.JSONDecodeError as e:
        logger.error("Failed to decode JSON data from '{}'. {}".format(file_path, e))
    except TypeError:
        logger.error("new_data must be a dictionary.")
    except Exception as e:
        logger.error("Error: {}".format(e))


def add_suffix_to_basename(file_path, suffix):
    """
    Add a suffix to the basename of the given file path.

    Parameters:
    - file_path (str): The input file path.
    - suffix (str): The suffix to be added to the basename. Default is '_quantized'.

    Returns:
    - str: The modified file path with the added suffix.
    """
    # Split the file path into directory and filename
    directory, filename = os.path.split(file_path)

    # Split the filename into basename and extension
    basename, extension = os.path.splitext(filename)

    # Add the suffix to the basename
    modified_basename = basename + suffix

    # Combine the modified basename and extension
    modified_filename = modified_basename + extension

    # Combine the directory and modified filename to get the new file path
    modified_file_path = os.path.join(directory, modified_filename)

    return modified_file_path


def read_files_from_directory(input_directory, extensions="wav", output_directory=None):
    """
    Read files with specified extensions from a directory.

    Parameters:
    - input_directory (str): The path to the input directory.
    - extensions (str or list): File extensions to look for (e.g., 'wav' or ['wav', 'mp3']).
    - output_directory (str, optional): The path to the output directory where files will be copied. If not provided, returns a list of file paths.

    Returns:
    - List: If output_directory is provided, a list of input-output file path pairs. Otherwise, a list of input file paths.

    Raises:
    - Exception: If there is an error during the process.

    Note:
    - If output_directory is provided, the method looks for files with specified extensions in the input_directory, copies them to the output_directory, and returns a list of input-output file path pairs.
    - If output_directory is not provided, the method returns a list of input file paths found in the input_directory.

    Example:
    ```
    input_files = read_files_from_directory('/path/to/input', extensions=['wav', 'mp3'], output_directory='/path/to/output')
    ```
    """
    try:
        if not isinstance(extensions, list):
            logger.debug("Converting")
            extensions = [extensions]
        extensions = ["*." + ext for ext in extensions]
        logger.info("Looking for the following file extensions: {}".format(extensions))

        file_paths = []

        # Create a glob pattern to match multiple extensions
        for ext in extensions:
            file_paths.extend(
                glob.glob(os.path.join(input_directory, "**", ext), recursive=True)
            )
        logger.info("Found {}".format(len(file_paths)))

        if output_directory:
            # Create a list by pairing up the input and output file paths
            pair_paths = []
            logger.info(
                "Creating a new directory for file paths in {}".format(output_directory)
            )

            # Iterate through each file and copy it to the destination directory
            for file_path in file_paths:
                # Create the new path by replacing the input_directory with output_directory
                out_path = file_path.replace(input_directory, output_directory)
                out_path = os.path.split(out_path)[0]

                # Make sure the directory structure exists in the destination directory
                os.makedirs(os.path.dirname(out_path), exist_ok=True)

                pair_paths.append([file_path, out_path])

            return pair_paths

        return file_paths

    except Exception as e:
        logger.error(f"An error occurred: {e}")


def get_clean_file_name_from_path(file_path):
    """
    Extract a clean and sanitized file name from a file path.

    Parameters:
    - file_path (str): The path to the file.

    Returns:
    - str: The cleaned file name.

    Note:
    - The method extracts the file name from the file path, removes symbols incompatible with Unix and Windows, replaces spaces with underscores, and converts the name to lowercase.
    """
    try:
        file_name = os.path.splitext(os.path.basename(file_path))[0]

        # Define a regular expression pattern to match symbols not compatible with Unix and Windows
        pattern = r'[\/:*?"<>|-]'

        # Replace matched symbols with underscores
        sanitized_name = re.sub(pattern, "_", file_name)

        # Replace spaces with underscores and remove single quotes
        cleaned_file_name = sanitized_name.replace(" ", "_").replace("'", "")

        # Convert the cleaned name to lowercase
        lowered_file_name = cleaned_file_name.lower()

        return lowered_file_name

    except Exception as e:
        logger.error(f"An error occurred: {e}")


def copy_files_to_new_directory(file_paths, new_directory):
    """
    Copies a list of files to a new directory and returns the new file paths.

    :param file_paths: A list of file paths to copy.
    :param new_directory: The path to the new directory where files will be copied.
    :return: A list of new file paths in the new directory.
    """
    # Create the new directory if it does not exist
    Path(new_directory).mkdir(parents=True, exist_ok=True)

    # List to hold the new file paths
    new_file_paths = []

    # Copy each file to the new directory
    for file_path in file_paths:
        if os.path.isfile(file_path):
            # Define the destination path for the file
            dest_path = os.path.join(new_directory, os.path.basename(file_path))
            # Copy the file to the new directory
            shutil.copy(file_path, dest_path)
            # Add the destination path to the list
            new_file_paths.append(dest_path)
        else:
            logger.warning("File not found: {}".format(file_path))

    # Return the list of new file paths
    return new_file_paths
