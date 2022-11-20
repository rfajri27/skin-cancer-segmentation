import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from modules.utility import path, parameters
from modules.unet_model import unet
from modules import data_preparation, metrics


df = data_preparation.load_dataset(path=path.DATA_PATH)
df_train, df_val, df_test = data_preparation.split_dataset(df=df)

train_dataset = data_preparation.image_data_generator(
    data_frame=df_train,
    batch_size=parameters.BATCH_SIZE,
    aug_dict=parameters.train_generator_args,
    target_size=parameters.IMAGE_SIZE
)
val_dataset = data_preparation.image_data_generator(
    data_frame=df_val,
    batch_size=parameters.BATCH_SIZE,
    aug_dict=dict(),
    target_size=parameters.IMAGE_SIZE
)

optimizer = Adam(
    lr=parameters.LEARNING_RATE, 
    beta_1=0.9, 
    beta_2=0.999, 
    epsilon=None, 
    amsgrad=False
)

callbacks = [
    ModelCheckpoint('models/unet/UNET_skin_seg.hdf5', verbose=0, save_best_only=True),
    EarlyStopping(monitor='val_loss', restore_best_weights=True, patience=15)
]

model = unet()
tf.keras.utils.plot_model(
    model, to_file='images/unet_model.png', show_shapes=False, show_dtype=False,
    show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96,
    layer_range=None
)

if __name__ == '__main__':
    model.compile(
        optimizer=optimizer, 
        loss=metrics.bce_dice_loss, 
        metrics=[
            metrics.iou,
            metrics.dice_coef
        ]
    )
    
    model.fit(
        train_dataset,
        steps_per_epoch=len(df_train)//parameters.BATCH_SIZE, 
        epochs=parameters.EPOCHS, 
        callbacks=callbacks,
        validation_data=val_dataset,
        validation_steps=len(df_val)//parameters.BATCH_SIZE
    )
    
    model.save("models/unet-model.h5")
