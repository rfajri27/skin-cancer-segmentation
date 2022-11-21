import os
import urllib.request
import zipfile
import io
import path

DATASET_URL = "https://drive.google.com/file/d/1E5P7Wt-2I40eb4qhKL3zjwB_LfnYtWkW/view?usp=share_link"

def get_dataset():
    # os.mkdir("data")
    with urllib.request.urlopen(DATASET_URL) as dl_file:
        with open(path.ROOT_PATH, 'wb') as out_file:
            out_file.write(dl_file.read())
    z = zipfile.ZipFile(os.path.join(path.ROOT_PATH, 'data.zip'))
    z.extractall(path="data")
    z.close()

if __name__ == '__main__':
    get_dataset()