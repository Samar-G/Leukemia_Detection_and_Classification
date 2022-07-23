import matplotlib.pyplot as plt
import Convert
import cv2
import numpy as np


def flattenGray(images):
    readImagesGray = []
    for image in images:
        imgG = image
        imgG = cv2.imread(imgG)
        imgG = cv2.cvtColor(imgG, cv2.COLOR_BGR2GRAY)
        imgblurG = cv2.GaussianBlur(imgG, (0, 0), sigmaX=33, sigmaY=33)
        divide = cv2.divide(imgG, imgblurG, scale=255)
        thresh = cv2.threshold(divide, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        readImagesGray.append(morph)

    lenImage = len(images)
    readImagesGray = np.array(readImagesGray)
    flattenedGray = np.array(readImagesGray).reshape(lenImage, -1)

    return flattenedGray


TrainData, TestData, labelOutputTrain, labelOutputTest = Convert.readFiles("F:/Graduation/Leukemia/model5images/Train/*", "F:/Graduation/Leukemia/model5images/Test/*")

classes = []
_, axs = plt.subplots(5, 5, figsize=(20, 20))
axs = axs.flatten()
i = 0
Test = flattenGray(TestData)

for img, ax in zip(Test, axs):
    temp_image = img  # cv2.imread(img)
    if i % 2 == 0:
        ax.set_title("Class : " + str(labelOutputTest[i]))
    ax.imshow(temp_image)
    i += 1
plt.show()

# _, axarr = plt.subplots(nrows=5, ncols=5, figsize=(20, 20))
# for a in range(5):
#     for b in range(5):
#         for i in range(len(TestData)):
#             temp_image = TestData[i]
#             temp_image = cv2.imread(temp_image)
#             # temp_image *= 255
#             # temp_image = np.clip(temp_image, 0, 255).astype("uint8")
#             if b == 2:
#                 axarr[a, b].set_title("Class : " + str(labelOutputTest[i]))
#             axarr[a, b].imshow(temp_image)
#             axarr[a, b].xaxis.set_visible(False)
#             axarr[a, b].yaxis.set_visible(False)
# plt.show()


# New DATA
_, axs = plt.subplots(5, 5, figsize=(20, 20))
axs = axs.flatten()
i = 0
Test = flattenGray(TestData)
for img in Test:
    print(img)
    temp_image = cv2.imread(img)
    print(temp_image)
    if i == 2 or i == 7 or i == 12 or i == 17 or i == 22:
        axs[i].set_title("Class : " + str(labelOutputTest[i]))
    axs[i].imshow(temp_image)
    axs[i].xaxis.set_visible(False)
    axs[i].yaxis.set_visible(False)
    i += 1
plt.show()

# # Old DATA
# _, axs = plt.subplots(5, 5, figsize=(20, 20))
# axs = axs.flatten()
# i = 0
# for img in TrainData:
#     temp_image = cv2.imread(img)
#     if i == 2 or i == 7 or i == 12 or i == 17 or i == 22:
#         axs[i].set_title("Class : " + str(labelOutputTrain[i]))
#     axs[i].imshow(temp_image)
#     axs[i].xaxis.set_visible(False)
#     axs[i].yaxis.set_visible(False)
#     i += 1
# plt.show()


