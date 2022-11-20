IMAGE_SIZE = (256, 256)
EPOCHS = 7
BATCH_SIZE = 20
LEARNING_RATE = 0.5e-3

train_generator_args = dict(
    rotation_range=0.1,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

