import os
import pickle

import joblib

import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras

BATCH = 64
AT = tf.data.AUTOTUNE
BUFFER = 1000
STEPS_PER_EPOCH = 800 // BATCH
VALIDATION_STEPS = 200 // BATCH
NORM = mpl.colors.Normalize(vmin=0, vmax=58)


def resize_image(image):
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    image = tf.image.resize(image, (128, 128))
    return image


def resize_mask(mask):
    mask = tf.image.resize(mask, (128, 128))
    mask = tf.cast(mask, tf.uint8)
    return mask


def brightness(img, mask):
    img = tf.image.adjust_brightness(img, 0.1)
    return img, mask


def gamma(img, mask):
    img = tf.image.adjust_gamma(img, 0.1)
    return img, mask


def hue(img, mask):
    img = tf.image.adjust_hue(img, -0.1)
    return img, mask


def crop(img, mask):
    img = tf.image.central_crop(img, 0.7)
    img = tf.image.resize(img, (128, 128))
    mask = tf.image.central_crop(mask, 0.7)
    mask = tf.image.resize(mask, (128, 128))
    mask = tf.cast(mask, tf.uint8)
    return img, mask


def flip_horizontal(img, mask):
    img = tf.image.flip_left_right(img)
    mask = tf.image.flip_left_right(mask)
    return img, mask


def flip_vertical(img, mask):
    img = tf.image.flip_up_down(img)
    mask = tf.image.flip_up_down(mask)
    return img, mask


def rotate(img, mask):
    img = tf.image.rot90(img)
    mask = tf.image.rot90(mask)
    return img, mask


def preprocess():
    image_path = []
    for root, dirs, files in os.walk('png_images'):
        for file in files:
            path = os.path.join(root, file)
            image_path.append(path)

    mask_path = []
    for root, dirs, files in os.walk('png_masks'):
        for file in files:
            path = os.path.join(root, file)
            mask_path.append(path)

    image_path.sort()
    mask_path.sort()

    images = []
    for path in image_path:
        file = tf.io.read_file(path)
        image = tf.image.decode_png(file, channels=3, dtype=tf.uint8)
        images.append(image)

    masks = []
    for path in mask_path:
        file = tf.io.read_file(path)
        mask = tf.image.decode_png(file, channels=1, dtype=tf.uint8)
        masks.append(mask)

    X = [resize_image(i) for i in images]
    y = [resize_mask(m) for m in masks]

    train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2)
    val_X, test_X, val_y, test_y = train_test_split(X, y, test_size=0.5)

    train_X = tf.data.Dataset.from_tensor_slices(train_X)
    val_X = tf.data.Dataset.from_tensor_slices(val_X)
    test_X = tf.data.Dataset.from_tensor_slices(test_X)

    train_y = tf.data.Dataset.from_tensor_slices(train_y)
    val_y = tf.data.Dataset.from_tensor_slices(val_y)
    test_y = tf.data.Dataset.from_tensor_slices(test_y)

    train = tf.data.Dataset.zip((train_X, train_y))
    val = tf.data.Dataset.zip((val_X, val_y))
    test = tf.data.Dataset.zip((test_X, test_y))

    x_brightness = train.map(brightness)
    x_gamma = train.map(gamma)
    x_hue = train.map(hue)
    x_crop = train.map(crop)
    x_flip_h = train.map(flip_horizontal)
    x_flip_v = train.map(flip_vertical)
    x_rotated = train.map(rotate)

    train = train.concatenate(x_brightness)
    train = train.concatenate(x_gamma)
    train = train.concatenate(x_hue)
    train = train.concatenate(x_crop)
    train = train.concatenate(x_flip_h)
    train = train.concatenate(x_flip_v)
    train = train.concatenate(x_rotated)

    train = train.cache().shuffle(BUFFER).batch(BATCH).repeat()
    train = train.prefetch(buffer_size=AT)
    val = val.batch(BATCH)
    test = test.batch(BATCH)

    return train, val, test


def main():
    train, val, test = preprocess()

    base = keras.applications.DenseNet121(input_shape=[128, 128, 3], include_top=False, weights='imagenet')
    skip_names = ['conv1/relu', 'pool2_relu', 'pool3_relu', 'pool4_relu', 'relu']
    skip_outputs = [base.get_layer(name).output for name in skip_names]
    for i in skip_outputs:
        print(i)
    downstack = keras.Model(inputs=base.input, outputs=skip_outputs)

    from tensorflow_examples.models.pix2pix import pix2pix

    upstack = [pix2pix.upsample(512, 3), pix2pix.upsample(256, 3), pix2pix.upsample(128, 3), pix2pix.upsample(64, 3)]
    inputs = keras.layers.Input(shape=[128, 128, 3])
    down = downstack(inputs)
    out = down[-1]
    skips = reversed(down[:-1])

    for up, skip in zip(upstack, skips):
        out = up(out)
        out = keras.layers.Concatenate()([out, skip])

    out = keras.layers.Conv2DTranspose(59, 3, strides=2, padding='same', )(out)
    unet = keras.Model(inputs=inputs, outputs=out)

    unet.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 optimizer=keras.optimizers.RMSprop(learning_rate=0.001),
                 metrics=['accuracy'])

    unet.fit(train,
             validation_data=val,
             steps_per_epoch=STEPS_PER_EPOCH,
             validation_steps=VALIDATION_STEPS,
             epochs=20,
             verbose=2)

    img, mask = next(iter(test))
    pred = unet.predict(img)
    plt.figure(figsize=(20, 28))
    k = 0
    for i in pred:
        plt.subplot(4, 3, 1 + k * 3)
        i = tf.argmax(i, axis=-1)
        plt.imshow(i, cmap='jet', norm=NORM)
        plt.axis('off')
        plt.title('pred')
        plt.subplot(4, 3, 2 + k * 3)
        plt.imshow(mask[k], cmap='jet', norm=NORM)
        plt.axis('off')
        plt.title('gold')
        plt.subplot(4, 3, 3 + k * 3)
        plt.imshow(img[k])
        plt.axis('off')
        plt.title('img')
        k += 1
        if k == 4: break
    plt.show()

    joblib.dump(unet, 'unet.pkl')


def load_and_predict():
    unet = joblib.load('unet_BEST.pkl')
    train, val, test = preprocess()
    img, mask = next(iter(test))
    pred = unet.predict(img)
    plt.figure(figsize=(20, 28))
    k = 0
    for i in pred:
        plt.subplot(4, 3, 1 + k * 3)
        i = tf.argmax(i, axis=-1)
        plt.imshow(i, cmap='jet', norm=NORM)
        plt.axis('off')
        plt.title('pred')
        plt.subplot(4, 3, 2 + k * 3)
        plt.imshow(mask[k], cmap='jet', norm=NORM)
        plt.axis('off')
        plt.title('gold')
        plt.subplot(4, 3, 3 + k * 3)
        plt.imshow(img[k])
        plt.axis('off')
        plt.title('img')
        k += 1
        if k == 4: break
    plt.show()


if __name__ == "__main__":
    # main()
    load_and_predict()
