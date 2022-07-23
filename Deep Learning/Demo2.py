from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D  # , AveragePooling2D
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import Convert
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import numpy
from sklearn.metrics import recall_score, precision_score
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2


def CNNModel1(Train, labelTrain, Test, labelTest, imageDim):
    # batch_size = 64
    epochs = 3
    num_classes = 5
    num_filters = 32
    filter_size = 3
    pool_size = 2
    model = Sequential([
        Conv2D(num_filters, filter_size, input_shape=(124, 124, imageDim), padding="same"),
        MaxPooling2D(pool_size=pool_size),
        Conv2D(num_filters, filter_size, padding="same"),
        MaxPooling2D(pool_size=pool_size),

        Flatten(),
        Dense(128, activation="relu"),  # Adding the Hidden layer 10254
        Dropout(0.1, seed=2019),
        Dense(num_classes, activation='softmax'),  # ouput layer
    ])

    model.compile(
        'adam',
        loss='categorical_crossentropy',
        metrics=[tf.keras.metrics.AUC(name="accuracy"), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    )

    model.fit(
        Train,
        to_categorical(labelTrain),
        epochs=epochs,
        validation_data=(Test, to_categorical(labelTest)),

    )
    model.save("F:/Graduation/Saved Models/Model1NewData4")
    model.save_weights("F:/Graduation/Saved Models/Model1NewDataWeights4")
    model.save("my_h5_model4.h5")
    model.summary()

    classes = ['ALL', 'AML', 'CLL', 'CML', 'Normal']
    predictions = model.predict(Test)
    # print(predictions)
    predictions = numpy.argmax(predictions, axis=1)
    print(sum([labelTest[i] == predictions[i] for i in range(len(predictions))]))
    print(len(labelTest))
    precision = precision_score(labelTest, predictions, average='micro')
    recall = recall_score(labelTest, predictions, average='micro')
    print(precision, recall)
    # print(predictions)
    cm = confusion_matrix(labelTest, predictions)
    print(cm)


def CNNModel2(Train, labelTrain, Test, labelTest, imageDim):
    # batch_size = 64
    epochs = 3
    num_classes = 5
    num_filters = 32
    filter_size = 3
    pool_size = 2
    model = Sequential([
        Conv2D(num_filters, filter_size, input_shape=(124, 124, imageDim), padding="same"),
        Conv2D(num_filters, filter_size, padding="same"),
        Conv2D(num_filters, filter_size, padding="same"),
        MaxPooling2D(pool_size=pool_size),
        Conv2D(num_filters, filter_size, padding="same"),
        Conv2D(num_filters, filter_size, padding="same"),
        Conv2D(num_filters, filter_size, padding="same"),
        MaxPooling2D(pool_size=pool_size),
        Conv2D(num_filters, filter_size, padding="same"),
        MaxPooling2D(pool_size=pool_size),

        Flatten(),
        Dense(128, activation="relu"),  # Adding the Hidden layer 10254
        Dropout(0.1, seed=2019),
        Dense(num_classes, activation='softmax'),  # ouput layer
    ])

    model.compile(
        'adam',
        loss='categorical_crossentropy',
        metrics=[tf.keras.metrics.AUC(name="accuracy"), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    )

    model.fit(
        Train,
        to_categorical(labelTrain),
        epochs=epochs,
        validation_data=(Test, to_categorical(labelTest)),

    )
    model.save("F:/Graduation/Saved Models/Model1NewData4")
    model.save_weights("F:/Graduation/Saved Models/Model1NewDataWeights4")
    model.save("my_h5_model4.h5")
    model.summary()

    classes = ['ALL', 'AML', 'CLL', 'CML', 'Normal']
    predictions = model.predict(Test)
    # print(predictions)
    predictions = numpy.argmax(predictions, axis=1)
    print(sum([labelTest[i] == predictions[i] for i in range(len(predictions))]))
    print(len(labelTest))
    precision = precision_score(labelTest, predictions, average='micro')
    recall = recall_score(labelTest, predictions, average='micro')
    print(precision, recall)
    # print(predictions)
    cm = confusion_matrix(labelTest, predictions)
    print(cm)


def AlexnetBuild(Train, labelTrain, Test, labelTest, imageDim):
    num_classes = 5
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=96, kernel_size=(5, 5), strides=(2, 2), activation='relu',
                            input_shape=(124, 124, imageDim)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
        tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        loss='categorical_crossentropy',
        metrics=[tf.keras.metrics.AUC(name="accuracy"), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
        optimizer=tf.optimizers.SGD(learning_rate=0.001), )
    model.summary()
    # callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)
    model.fit(Train,
              to_categorical(labelTrain),
              epochs=2,
              validation_data=(Test, to_categorical(labelTest)), )
    # callbacks=[callback])
    model.save("F:/Graduation/Saved Models/Model1NewDataAlex2")
    model.save_weights("F:/Graduation/Saved Models/Model1NewDataWeightsAlex2")
    model.save("my_h5_model_alex2.h5")
    model.summary()

    predictions = model.predict(Test)
    # print(predictions)
    predictions = numpy.argmax(predictions, axis=1)
    print(sum([labelTest[i] == predictions[i] for i in range(len(predictions))]))
    print(len(labelTest))
    precision = precision_score(labelTest, predictions, average='micro')
    recall = recall_score(labelTest, predictions, average='micro')
    print(precision, recall)
    # print(predictions)
    cm = confusion_matrix(labelTest, predictions)
    print(cm)


def res50Built(Train, labelTrain, Test, labelTest, imageDim):
    epochs = 2
    num_classes = 5
    model = Sequential([
        ResNet50(weights=None, include_top=True, input_shape=(124, 124, imageDim), classes=5),
        Flatten(),
        Dense(500, activation="relu"),  # Adding the Hidden layer
        Dropout(0.1, seed=2022),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        'sgd',
        loss='categorical_crossentropy',
        metrics=[tf.keras.metrics.AUC(name="accuracy"), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    )
    model.fit(
        Train,
        to_categorical(labelTrain),
        epochs=epochs,
        validation_data=(Test, to_categorical(labelTest)),
    )
    model.save("F:/Graduation/Saved Models/Model1NewDataRes2")
    model.save_weights("F:/Graduation/Saved Models/Model1NewDataWeightsRes2")
    model.save("my_h5_model_RES2.h5")
    model.summary()

    predictions = model.predict(Test)
    # print(predictions)
    predictions = numpy.argmax(predictions, axis=1)
    print(sum([labelTest[i] == predictions[i] for i in range(len(predictions))]))
    print(len(labelTest))
    precision = precision_score(labelTest, predictions, average='micro')
    recall = recall_score(labelTest, predictions, average='micro')
    print(precision, recall)
    # print(predictions)
    cm = confusion_matrix(labelTest, predictions)
    print(cm)


def VGG16Built(Train, labelTrain, Test, labelTest, imageDim):
    epochs = 2
    num_classes = 5
    model = Sequential([
        VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, imageDim)),
        Flatten(),
        Dense(1010, activation="relu"),  # Adding the Hidden layer
        Dropout(0.2, seed=2022),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        'sgd',
        loss='categorical_crossentropy',
        metrics=[tf.keras.metrics.AUC(name="accuracy"), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    )
    model.fit(
        Train,
        to_categorical(labelTrain),
        epochs=epochs,
        validation_data=(Test, to_categorical(labelTest)),
    )
    model.save("F:/Graduation/Saved Models/Model1OldDataVGG")
    model.save_weights("F:/Graduation/Saved Models/Model1OldDataWeightsVGG")
    model.save("my_h5_model_VGG.h5")
    model.summary()

    predictions = model.predict(Test)
    # print(predictions)
    predictions = numpy.argmax(predictions, axis=1)
    print(sum([labelTest[i] == predictions[i] for i in range(len(predictions))]))
    print(len(labelTest))
    precision = precision_score(labelTest, predictions, average='micro')
    recall = recall_score(labelTest, predictions, average='micro')
    print(precision, recall)
    # print(predictions)
    cm = confusion_matrix(labelTest, predictions)
    print(cm)


def modelsCall(trainDir, testDir):
    TrainData, TestData, labelOutputTrain, labelOutputTest = Convert.readFiles(trainDir, testDir)

    TrainR, TrainG = Convert.ReadImagesCNN(TrainData)
    TestR, TestG = Convert.ReadImagesCNN(TestData)
    outProcessTrain = preprocessing.LabelEncoder()
    outProcessTest = preprocessing.LabelEncoder()

    outProcessTest.fit(labelOutputTest)
    LabelTest = outProcessTest.transform(labelOutputTest)

    outProcessTrain.fit(labelOutputTrain)
    LabelTrain = outProcessTrain.transform(labelOutputTrain)

    CNNModel2(TrainR, LabelTrain, TestR, LabelTest, int(3))
    # res50Built(TrainR, LabelTrain, TestR, LabelTest, int(3))
    # AlexnetBuild(TrainR, LabelTrain, TestR, LabelTest, int(3))
    # VGG16Built(TrainR, LabelTrain, TestR, LabelTest, int(3))


# dirTrain = 'F:/Graduation/newLeukemia/New Data/Data/Train/*'
# dirTest = 'F:/Graduation/newLeukemia/New Data/Data/Test/*'

# dirTrain = 'F:/Graduation/Leukemia/FinalGP_Data/ModelWithAug/Train/*'
# dirTest = 'F:/Graduation/Leukemia/FinalGP_Data/ModelWithAug/Test/*'

dirTrain = 'F:/Graduation/newLeukemia/New Data/PrePro_Data/Train/*'
dirTest = 'F:/Graduation/newLeukemia/New Data/PrePro_Data/Test/*'
modelsCall(dirTrain, dirTest)
