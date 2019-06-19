import numpy as np
from skimage import io
import os
import matplotlib.pyplot as plt

def read_face_images(data_folder):
    image_paths = [os.path.join(data_folder, item) for item in os.listdir(data_folder)]
    images, labels, subjects = [], [], []
    for image_path in image_paths:
        im = io.imread(image_path,as_grey=True)
        images.append(np.array(im, dtype='uint8'))
        labels.append(int(os.path.split(image_path)[1].split(".")[0].replace("subject", "")))
        subjects.append(os.path.split(image_path)[1].split(".")[1])
    return np.array(images), np.array(labels), np.array(subjects)

data_folder = "/Users/wanglei/ml_data/yalefaces"
images, labels, subjects = read_face_images(data_folder)

plt.figure(1)
plt.imshow(images[1])
plt.figure(2)
plt.imshow(images[2])
plt.figure(3)
plt.imshow(images[3])
plt.figure(4)
plt.imshow(images[4])

plt.show()





    



















