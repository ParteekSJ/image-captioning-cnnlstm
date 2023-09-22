import shutil
from pathlib import Path
import json
from pathlib import Path
from collections import OrderedDict


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def delete_folder(folder_name):
    try:
        shutil.rmtree(folder_name)
        print(f"Successfully deleted the '{folder_name}' folder and its contents.")
    except OSError as e:
        print(f"Error: {e} - '{folder_name}' folder was not deleted.")


# def download_colab():
#     # Provide the path to the folder you want to download
#     folder_path = (
#         "/content/proj/train_log/2023-0922_1121"  # Replace with the actual folder path
#     )

#     # Provide a name for the zip file
#     zip_file_name = "ttt.zip"  # Replace with your desired zip file name

#     # Change the current working directory to the parent directory of the folder
#     import os

#     os.chdir(os.path.dirname(folder_path))

#     # Create a zip archive of the folder
#     shutil.make_archive(zip_file_name[:-4], "zip", os.path.basename(folder_path))

#     # Trigger the download

#     files.download(zip_file_name)
