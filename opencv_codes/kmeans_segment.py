import numpy as np
import cv2
import matplotlib.pyplot as plt

img_file = ""
image = cv2.imread(img_file)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# flatten to 2D arr
pixels = image.reshape(-1,3).astype(np.float32)

criteria = (cv2.TermCriteria_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
k = 3

# kmeans
_, labels, center = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

segemented_img = np.zeros_like(image)

for i in range(k):
    segemented_img[labels==i] = center[i]

plt.figure()
plt.imshow(segemented_img)
plt.title("Segmented Image")
plt.show()