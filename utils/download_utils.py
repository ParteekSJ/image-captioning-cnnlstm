import gdown
import os
import requests

"""
https://developers.google.com/oauthplayground/
1. Drive API v3
2. https://www.googleapis.com/auth/drive
3. Authorize APIs
4. Exchange Authorization code for tokens
5. Grab the 'access token' and add 'Bearer' as a prefix to it.
"""


def download_file_from_drive(link):
    # Original Google Drive link
    # link = "https://drive.google.com/file/d/1MoL4Oi6Ve32mwFZR9yJNGWV5GHl0J2DR/view?usp=sharing"

    # Extract the file ID from the original link
    file_id = link.split("/")[-2]

    # Construct thea modified download link using the file ID
    download_link = f"https://drive.google.com/uc?id={file_id}"

    # Download the file using gdown.download
    output_file = "./pytorch_model.bin"  # Replace 'output_file_name.extension' with the desired output filename and extension
    gdown.download(download_link, output_file)


# header = {
#     "authorization": "Bearer authcode",
# }


def create_folder_on_drive(name, parent_id, header):
    data = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_id],
    }

    url = "https://www.googleapis.com/drive/v3/files"

    response = requests.post(url, headers=header, json=data)

    if response.status_code == 200:
        print(f"Folder {name} created successfully.")
        return response.json()["id"]  # return the ID of the created folder
    else:
        print(f"Failed to create the folder {name}: {response.status_code}")
        return None


def upload_file_to_drive(file_path, parent_id, header):
    param = {
        "name": os.path.basename(file_path),  # Name will be the original filename
        "parents": [parent_id],  # ID of the destination folder
    }

    upload_url = "https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable"

    response = requests.post(upload_url, headers=header, json=param)
    if response.status_code != 200:
        print(f"Failed to initiate the upload for {file_path}: {response.status_code}")
        return

    location = response.headers["Location"]

    with open(file_path, "rb") as file:
        file_contents = file.read()
        response = requests.put(location, data=file_contents)
        if response.status_code == 200:
            print(f"File {file_path} uploaded successfully.")
        else:
            print(f"Failed to upload the file {file_path}: {response.status_code}")


def upload_folder_to_drive(folder_path, parent_id, header):
    for root, dirs, files in os.walk(folder_path):
        for dir_name in dirs:
            dir_id = create_folder_on_drive(dir_name, parent_id, header)
            if dir_id:
                for file in files:
                    file_path = os.path.join(root, file)
                    upload_file_to_drive(
                        file_path, dir_id, header
                    )  # Upload files into the created folder


# folder_path = "../FINALPROJECT/psj201/summaries" # local folder path
# parent_id = "1IN5N4rTV70twuxZeUhtZv_tZFMAOe4CN" # id of folder on gdrive
# upload_folder_to_drive(folder_path, parent_id, header)
