import os
import urllib.request
import zipfile
import io
import path

DATASET_URL = "https://drive.google.com/file/d/1Bn9slyX_5qYLUgwxDPxJ7iwwzaKE6Tap/view?usp=share_link"

def get_dataset():
    with urllib.request.urlopen(DATASET_URL) as dl_file:
        with open(path.ROOT_PATH, 'wb') as out_file:
            out_file.write(dl_file.read())
    z = zipfile.ZipFile(os.path.join(path.ROOT_PATH, 'data.zip'))
    z.extractall()
    z.close()

if __name__ == '__main__':
    get_dataset()