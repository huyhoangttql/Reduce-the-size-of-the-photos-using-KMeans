import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
img = plt.imread("a.jpg")

width = img.shape[0]
heigth = img.shape[1]
# print(img.shape)

img = img.reshape(width*heigth,3)
# print(img.shape)

kmeans = KMeans(n_clusters=4).fit(img)

labels = kmeans.predict(img)

clusters = kmeans.cluster_centers_
# print(clusters)
# print(labels)

img2 = np.zeros_like(img)
for i in range(len(img2)):
    img2[i] = clusters[labels[i]]
img2 = img2.reshape(width,heigth,3)

plt.imshow(img2)
plt.show()
