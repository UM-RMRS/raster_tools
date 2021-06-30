import os


def validate_file(path):
    if os.path.isfile(path):
        return path
    raise FileNotFoundError(f"Could not find file: '{path}'")
