import numpy as np
import cv2

def read_image(df, index, image_size):
    img = cv2.imread(df['image_path'].iloc[index])
    img = cv2.resize(img ,image_size)
    img = img / 255
    img = img[np.newaxis, :, :, :]
    return img

def read_mask(df, index, image_size):
    mask_img = cv2.imread(df['mask_path'].iloc[index])
    mask_img = cv2.resize(mask_img , image_size)
    mask_img = mask_img / 255
    mask_img = mask_img[np.newaxis, :, :, :]
    return mask_img