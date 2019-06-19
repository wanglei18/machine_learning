import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split

def get_data():
    df = pd.read_csv("/Users/wanglei/ML_data/flower_images/flower_labels.csv")
    files = df['file']
    labels = df['label'].values
    images = []
    for file in files:
        image = cv2.imread("/Users/wanglei/ML_data/flower_images/" + file)
        image = cv2.resize(image, (128, 128), 0, 0, cv2.INTER_LINEAR)
        image = np.multiply(image, 1.0 / 255.0)
        images.append(image)
    return np.array(images), labels

X, y = get_data()
print(X.shape)
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=3)
























