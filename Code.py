import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import cv2
from sklearn.model_selection import train_test_split
import pickle
import os
import pandas as pd
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator

path = "myData"  
labelFile = 'labels.csv' 
batch_size_val = 50
steps_per_epoch_val = 2000
epochs_val = 10
imageDimensions = (32, 32, 3)
testRatio = 0.2 
validationRatio = 0.2
threshold = 0.75
print("Importing Classes...")
images, classNo = [], []
myList = os.listdir(path)
print("Total Classes Detected:", len(myList))
for count, folder in enumerate(myList):
    for img_name in os.listdir(f"{path}/{folder}"):
        img = cv2.imread(f"{path}/{folder}/{img_name}")
        images.append(img)
        classNo.append(count)
images = np.array(images)
classNo = np.array(classNo)

X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validationRatio)

print("Data Shapes")
print("Train:", X_train.shape, y_train.shape)
print("Validation:", X_validation.shape, y_validation.shape)
print("Test:", X_test.shape, y_test.shape)

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def equalize(img):
    return cv2.equalizeHist(img)

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    return img / 255.0

X_train = np.array(list(map(preprocessing, X_train)))
X_validation = np.array(list(map(preprocessing, X_validation)))
X_test = np.array(list(map(preprocessing, X_test)))

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

dataGen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, shear_range=0.1, rotation_range=10)
dataGen.fit(X_train)

y_train = to_categorical(y_train, len(myList))
y_validation = to_categorical(y_validation, len(myList))
y_test = to_categorical(y_test, len(myList))

def myModel():
    model = Sequential([
        Conv2D(60, (5, 5), input_shape=(32, 32, 1), activation='relu'),
        Conv2D(60, (5, 5), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(30, (3, 3), activation='relu'),
        Conv2D(30, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.5),
        Flatten(),
        Dense(500, activation='relu'),
        Dropout(0.5),
        Dense(len(myList), activation='softmax')
    ])
    model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if os.path.exists("model_trained.p"):
    print("Loading pre-trained model...")
    with open("model_trained.p", "rb") as f:
        model = pickle.load(f)
else:
    print("Training new model...")
    model = myModel()
    history = model.fit(dataGen.flow(X_train, y_train, batch_size=batch_size_val),
                        steps_per_epoch=steps_per_epoch_val, epochs=epochs_val,
                        validation_data=(X_validation, y_validation), shuffle=True)
    with open("model_trained.p", "wb") as f:
        pickle.dump(model, f)
    plt.figure(1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['training', 'validation'])
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.figure(2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.legend(['training', 'validation'])
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.show()

score = model.evaluate(X_test, y_test, verbose=0)
print('Test Score:', score[0])
print('Test Accuracy:', score[1])

from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score
import numpy as np

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')
accuracy = accuracy_score(y_true_classes, y_pred_classes)

print(f"Test Accuracy: {score[1] * 100:.2f}%")

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Accuracy: {accuracy:.4f}")

print("Classification Report:")
report = classification_report(y_true_classes, y_pred_classes)
print(report)

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(10, 180)

def getClassName(classNo):
    with open(labelFile, "r") as file:
        labels = pd.read_csv(file)
        return labels.iloc[classNo]["Name"]

while True:
    success, imgOriginal = cap.read()
    img = cv2.resize(imgOriginal, (32, 32))
    img = preprocessing(img).reshape(1, 32, 32, 1)
    predictions = model.predict(img)
    classIndex = np.argmax(predictions)
    probabilityValue = np.max(predictions)

    if probabilityValue > threshold:
        cv2.putText(imgOriginal, f"CLASS: {getClassName(classIndex)}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(imgOriginal, f"PROBABILITY: {probabilityValue*100:.2f}%", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Result", imgOriginal)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
