from tensorflow.keras.models import load_model
from training import df_test
from modules import data_preparation, metrics
from modules.utility import parameters

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
    results = model.evaluate(test_dataset, steps=len(df_test)//parameters.BATCH_SIZE)
    print("Test loss: ",results[0])
    print("Test IOU: ",results[1])
    print("Test Dice Coefficent: ",results[2])