import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from modules import data_preparation, metrics, load_image
from modules.utility import parameters
from training import df_test

test_dataset = data_preparation.image_data_generator(
    data_frame=df_test,
    batch_size=parameters.BATCH_SIZE,
    aug_dict=dict(),
    target_size=parameters.IMAGE_SIZE
)

model = load_model(
    'models/unet-model.h5', 
    custom_objects={
        "bce_dice_loss":metrics.bce_dice_loss,
        "iou":metrics.iou,
        "dice_coef":metrics.dice_coef
    }
)

if __name__ == '__main__':
    for i in range(3):
        index = np.random.randint(1,len(df_test.index))
        img = load_image.read_image(df_test, index, parameters.IMAGE_SIZE)
        mask_img = load_image.read_mask(df_test, index, parameters.IMAGE_SIZE)
        pred = model.predict(img)
        
        plt.figure(figsize=(7,7))
        plt.subplot(1,3,1)
        plt.imshow(np.squeeze(img))
        plt.title('Original Image')
        plt.subplot(1,3,2)
        plt.imshow(np.squeeze(mask_img))
        plt.title('Original Mask')
        plt.subplot(1,3,3)
        plt.imshow(np.squeeze(pred) > .5)
        plt.title('Prediction')
        plt.savefig(
            "images/prediction_results/prediction_results_{}.png".format(i))