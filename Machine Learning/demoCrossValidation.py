import cv2
import glob
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.naive_bayes import CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import svm, tree
from sklearn.ensemble import RandomForestClassifier
import random
import Convert
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
import pandas as pd


# Convert.convertedImagesTrain()
# Convert.convertedImagesTest()
# Convert.readFiles()


def flattenRGB(images):
    readImagesRGB = []
    readImagesRGBCanny = []

    for image in images:
        # print(image)
        imgRGB = image

        # imgBlurRGB = cv2.GaussianBlur(imgRGB, (3, 3), 0)
        # edges = cv2.Canny(image=imgBlurRGB, threshold1=50, threshold2=150)  # Canny Edge Detection
        contours, hierarchy = cv2.findContours(imgRGB, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        image2 = cv2.drawContours(imgRGB.copy(), contours, -1, (0, 255, 0), 2)
        # cv2.imshow("2", image2)
        # cv2.waitKey(0)
        # break

        readImagesRGBCanny.append(image2)
        # readImagesRGB.append(imgRGB)

    lenImage = len(images)
    readImagesRGBCanny = np.array(readImagesRGBCanny)
    flattenedRGB = np.array(readImagesRGBCanny).reshape(lenImage, -1)

    return flattenedRGB


def flattenGray(images):
    # readImagesGray = []
    readImagesGrayCanny = []

    for image in images:
        # imgG = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        imgG = image
        # cv2.imshow("before", imgG)
        # cv2.waitKey(0)
        imgG = cv2.cvtColor(imgG, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("gray", imgG)
        # readImagesGray.append(imgG)

        # imgBlurGray = cv2.GaussianBlur(imgG, (3, 3), 0)
        # cv2.imshow("blur", imgBlurGray)
        #
        # # train: 73%, test: 23% -> 500 images
        # # train: 55%, test: 33% -> AML ALL Normal
        # _, binary = cv2.threshold(imgG, 225, 255, cv2.THRESH_BINARY_INV)
        # contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # image2 = cv2.drawContours(imgG.copy(), contours, -1, (0, 255, 0), 2)
        # cv2.imshow("contour", image2)
        # cv2.waitKey(0)
        # break
        blur = cv2.GaussianBlur(imgG, (0, 0), sigmaX=33, sigmaY=33)
        divide = cv2.divide(imgG, blur, scale=255)
        thresh = cv2.threshold(divide, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        image2 = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        readImagesGrayCanny.append(image2)
        # cv2.imshow("after pre", image2)
        # cv2.waitKey(0)
        # break

    lenImage = len(images)
    readImagesGrayCanny = np.array(readImagesGrayCanny)
    flattenedGray = np.array(readImagesGrayCanny).reshape(lenImage, -1)

    return flattenedGray


def svmClass(features, output, featuresO, testO):
    clf = svm.SVC(kernel="poly")
    print("In SVM")
    clf.fit(features, output)
    print("training done")
    outA = clf.predict(featuresO)
    print("testing done")
    accS = accuracy_score(testO, outA) * 100
    preS = precision_score(testO, outA, average='micro')
    recS = recall_score(testO, outA, average='micro')
    print(testO[:5])
    print("____________________________________")
    print((outA[0:5]))
    out = accuracy_score(output, clf.predict(features)) * 100
    print(out)
    out2 = confusion_matrix(testO, outA)
    print(out2)
    return preS, recS, accS, outA


def logClass(features, output, featuresO, testO):
    classifier = LogisticRegression(max_iter=4500)
    print("In Logistic")
    classifier.fit(features, output)
    print("training done")
    outA = classifier.predict(featuresO)
    print("testing done")
    accL = accuracy_score(testO, outA) * 100
    preL = precision_score(testO, outA, average='micro')
    recL = recall_score(testO, outA, average='micro')
    print(testO[:5])
    print("____________________________________")
    print((outA[0:5]))
    out = accuracy_score(output, classifier.predict(features)) * 100
    print(out)
    out2 = confusion_matrix(testO, outA)
    print(out2)
    return preL, recL, accL, outA


def KnnClass(features, output, featuresO, testO):
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(features, output)
    y_pred = classifier.predict(featuresO)
    accL = accuracy_score(testO, y_pred) * 100
    preL = precision_score(testO, y_pred, average='micro')
    recL = recall_score(testO, y_pred, average='micro')
    return preL, recL, accL, y_pred


def naiveBayes(features, output, featuresO, testO):
    cnb = CategoricalNB()
    cnb.fit(features, output)
    y_pred = cnb.predict(featuresO)
    acc = accuracy_score(testO, y_pred) * 100
    pre = precision_score(testO, y_pred, average='micro')
    rec = recall_score(testO, y_pred, average='micro')
    return pre, rec, acc


# return confusion_matrix(testO, y_pred),classification_report(testO, y_pred)


def dtClass(features, output, featuresO, testO):
    dt = tree.DecisionTreeClassifier()
    dt = dt.fit(features, output)
    outA = dt.predict(featuresO)
    acc = accuracy_score(testO, outA) * 100
    pre = precision_score(testO, outA)
    rec = recall_score(testO, outA)
    return pre, rec, acc


def RandomForest(features, output):
    features = pd.DataFrame(features)
    output = pd.DataFrame(output)
    print("pass")
    pipeline = make_pipeline(StandardScaler(),
                             svm.SVC(kernel="poly"))
                              # RandomForestClassifier(n_estimators=100,max_depth=4))

    strtfdKFold = StratifiedKFold(n_splits=5)

    kfold = strtfdKFold.split(features, output)
    print("pass4")
    scores = []

    for k, (train, test) in enumerate(kfold):
        print("in loop")
        pipeline.fit(features.iloc[train, :], output.iloc[train].values.ravel())

        score = pipeline.score(features.iloc[test, :], output.iloc[test])

        scores.append(score)
        print("in loop")

    print(scores)

    print('\n\nCross-Validation accuracy: %.3f' % np.mean(scores))


def modelsCall(trainDir, testDir):
    TrainData, TestData, labelOutputTrain, labelOutputTest = Convert.readFiles(trainDir, testDir)

    # TrainR = flattenRGB(TrainRG)
    # print("khlast flatten RGB train")
    TrainG = flattenGray(TrainData)
    print("khlast flatten gray train")

    RandomForest(TrainG, labelOutputTrain)


dirTrain = 'D:/Iseul/Education/College/(4)_1st_Semester/Graduation Project/ModelWithAug(Train&Test)/Train&Test/*'
dirTest = 'D:/Iseul/Education/College/(4)_1st_Semester/Graduation Project/ModelWithAug(Train&Test)/Test/*'
modelsCall(dirTrain, dirTest)
