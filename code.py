import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
import glob

# List of folder paths
folders = [
    '1',
    '2',
    '3',
    '4',
    '5'
]

for folder in folders:
    jpg_files = glob.glob(os.path.join(os.path.join("Data",folder), '*.jpg'))
    num_jpg_files = len(jpg_files)
    print(f"Folder '{folder}' contains {num_jpg_files} JPG images.")



def applyLaplacian(laplacianMask, img):
    filteredImg = cv2.filter2D(img, -1, laplacianMask)
    finalImg = np.uint8(np.clip(img + filteredImg, 0, 255))
    return finalImg
  
def applySobel(sobelMask, img):
    filteredImg = cv2.filter2D(img, -1, sobelMask)
    finalImg = np.uint8(np.clip(img + filteredImg, 0, 255))  
    return finalImg


img = cv2.imread('Data//1//2778091.jpg',0)

laplacianMask = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])  
final_img = applyLaplacian(laplacianMask, img)

laplacianMask2 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
final_img2 = applyLaplacian(laplacianMask2, img)

sobelMask = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
final_img3 = applySobel(sobelMask, img)

fig, axs = plt.subplots(1, 3, figsize=(15, 6))
axs[0].imshow(final_img)
axs[0].set_title('Laplacian Mask 1')
axs[1].imshow(final_img2)
axs[1].set_title('Laplacian Mask 2')
axs[2].imshow(final_img3)
axs[2].set_title('Sobel Mask')
plt.show()



def hist_equalization(img):
    img = cv2.equalizeHist(img)
    return img

# 2778062

img = cv2.imread('Data//1//2778091.jpg', 0) 
# img = cv2.resize(img, (224, 224))
equ = hist_equalization(img)  

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.title('Original')
plt.subplot(122)
plt.imshow(equ, cmap='gray')
plt.title('Equalized')
plt.show()

def hist_equalization(img):
    img = cv2.equalizeHist(img)
    return img

# 2778062

img = cv2.imread('Data//1//2778091.jpg', 0) 
# img = cv2.resize(img, (224, 224))
equ = hist_equalization(img)  

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.title('Original')
plt.subplot(122)
plt.imshow(equ, cmap='gray')
plt.title('Equalized')
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(data_dir):
    images = []
    labels = []
    class_folders = os.listdir(data_dir)

    for class_folder in class_folders:
        class_path = os.path.join(data_dir, class_folder)
        count = 0
        for filename in os.listdir(class_path):
            if filename.endswith('.jpg'):
                image_path = os.path.join(class_path, filename)
                img = cv2.imread(image_path)
                img = cv2.resize(img, (75, 75))  
                img = np.array(img)
                img = img / 255.0 
                images.append(img)
                labels.append(class_folder)
                count+=1
            if count == 500:
                break
    images = np.array(images)
    labels = np.array(labels)
    return images, labels



data_dir = 'Data'
images, labels = load_and_preprocess_data(data_dir)

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

encoded_labels[0:10]
# classes = [4,3,2,1]

data_dir = 'Data'
images, labels = load_and_preprocess_data(data_dir)
print(images[0].shape)
print(len(labels))

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, stratify=labels)

model = ExtraTreesClassifier(n_estimators=100)
model.fit(X_train, y_train)
ypred = model.predict(X_test)
print("Accuracy:",accuracy_score(y_test, ypred))
print("Classification Report:", classification_report(y_test, ypred))
print("Confusion Matrix:\n", confusion_matrix(y_test, ypred))


model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
model.trainable = False

inp = layers.Input(shape=(32, 32, 3))
layer = model(inp, training=False)
layer = layers.GlobalAveragePooling2D()(layer)
layer = layers.Dense(128, activation='relu')(layer)
layer = layers.Dropout(0.5)(layer)
layer = layers.Dense(64, activation='relu')(layer)
layer = layers.Dropout(0.5)(layer)
layer = layers.Dense(32, activation='relu')(layer)
layer = layers.Dropout(0.5)(layer)
layer = layers.Dense(32, activation='relu')(layer)
layer = layers.Dropout(0.5)(layer)
layer = layers.Dense(5, activation='softmax')(layer)
model = tf.keras.Model(inp, layer)

lr = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


X_train, X_test, y_train, y_test = train_test_split(images, encoded_labels, test_size=0.3, stratify=encoded_labels)


model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), batch_size = 100)

ypred = np.argmax(model.predict(X_test),axis=1)
print("Accuracy:",accuracy_score(y_test, ypred))
print("Classification Report:\n", classification_report(y_test, ypred))
print("Confusion Matrix:\n", confusion_matrix(y_test, ypred))


import tensorflow as tf
from tensorflow.keras import datasets, layers, models

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5)) 
model.add(layers.Dense(5, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
model.summary()

X_train, X_test, y_train, y_test = train_test_split(images, encoded_labels, test_size=0.3, stratify=encoded_labels)

model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size = 100)




from keras.preprocessing.image import ImageDataGenerator

def load_and_preprocess_data(data_dir, laplacianMask, augment=True):
    images = []
    labels = []
    class_folders = os.listdir(data_dir)
    
    max_images_per_class = max([len(os.listdir(os.path.join(data_dir, class_folder))) for class_folder in class_folders])
#     print(max_images_per_class)

    # Create an ImageDataGenerator with desired augmentation settings
    datagen = ImageDataGenerator(
        rotation_range=45, 
        horizontal_flip=True, 
        width_shift_range=0.5, 
        height_shift_range=0.5, 
        dtype='float32'
    )

    for class_folder in class_folders:
        class_path = os.path.join(data_dir, class_folder)
        for filename in os.listdir(class_path):
            if filename.endswith('.jpg'):
                image_path = os.path.join(class_path, filename)
                img = cv2.imread(image_path)
                img = applyLaplacian(laplacianMask, img)
                img = cv2.resize(img, (224, 224)) 
                img = np.array(img)
                img = img / 255.0  
                images.append(img)
                labels.append(class_folder)
            

        if augment:
            class_images = np.array([img for img, label in zip(images, labels) if label == class_folder])
            num_images_to_generate = max_images_per_class - len(class_images)

            if num_images_to_generate > 0:
                augmented_images = []
                for i in range(num_images_to_generate):
                    augmented_img = datagen.flow(np.array([class_images[i % len(class_images)]]), batch_size=1)[0]
#                     print(augmented_img.shape)
                    augmented_img = augmented_img[0]
                    augmented_images.append(augmented_img)
                
                images.extend(augmented_images)
                labels.extend([class_folder] * num_images_to_generate)

    images = np.array(images)
    labels = np.array(labels)
    return images, labels
