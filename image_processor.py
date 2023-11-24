import os
import pickle

import numpy as np
from keras.preprocessing.image import img_to_array, load_img
from sklearn.preprocessing import LabelEncoder


def load_images_from_folder(folder, class_names):
    images = []
    labels = []
    for class_name in class_names:
        train_path = os.path.join(folder, class_name, 'data', 'train')
        test_path = os.path.join(folder, class_name, 'data', 'test')
        val_path = os.path.join(folder, class_name, 'data', 'val')

        for dataset_path in [train_path, test_path, val_path]:
            for dir, _, files in os.walk(dataset_path):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(dir, file)
                        img = load_img(img_path, target_size=(128, 128))
                        img_array = img_to_array(img)
                        img_array = img_array.transpose(2, 0, 1)
                        images.append(img_array)
                        labels.append(class_name)

    images_np = np.array(images, dtype='float32')
    labels_np = np.array(labels)

    return images_np, labels_np


if __name__ == "__main__":
    image_folder = 'images'
    class_names = [folder for folder in os.listdir(image_folder) if os.path.isdir(os.path.join(image_folder, folder))]

    images, labels = load_images_from_folder(image_folder, class_names)

    images /= 255.0

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    with open('X_images_scaled.pkl', 'wb') as f:
        pickle.dump(images.reshape(len(images), -1), f)

    with open('Y_images_names.pkl', 'wb') as f:
        pickle.dump(labels, f)

    with open('Y_images_encoded.pkl', 'wb') as f:
        pickle.dump(encoded_labels, f)

    with open('X_images_scaled.pkl', 'rb') as f:
        X = pickle.load(f)

    with open('Y_images_names.pkl', 'rb') as f:
        y_raw = pickle.load(f)

    with open('Y_images_encoded.pkl', 'rb') as f:
        y = pickle.load(f)

