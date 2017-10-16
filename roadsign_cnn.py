import keras
from keras.models import Sequential
from keras.layers import Activation,Dense,Dropout,Conv2D,MaxPooling2D,Flatten
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adagrad, Adam, RMSprop
from PIL import Image
import numpy as np
from matplotlib import pylab as plt
import os

def plot_history(history):
    # 精度の履歴をプロット
    plt.plot(history.history['acc'],"o-",label="accuracy")
    plt.plot(history.history['val_acc'],"o-",label="val_acc")
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc="lower right")
    plt.show()

    # 損失の履歴をプロット
    plt.plot(history.history['loss'],"o-",label="loss",)
    plt.plot(history.history['val_loss'],"o-",label="val_loss")
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='lower right')
    plt.show()

if __name__ == "__main__":

    # CONFIG - Image
    img_w = 50
    img_h = 50
    img_ch = 3 #R,G,B
    trainimg_rootdir = "./roadsign/train"
    testimg_rootdir = "./roadsign/test"

    # CONFIG - fitting behavior
    epoch = 1500
    batch_size = 100
    validation_split = 0.2

    ## Training image and labels
    train_img=[]
    train_lbl=[]
    test_img=[]
    test_lbl=[]
    train_lblbin = []
    test_lblbin = []

    # Read train and test
    for root_dir in [trainimg_rootdir,testimg_rootdir]:
        if root_dir == ".DS_Store":
            continue

        # Read true/false image in train or test
        for tf_dir in os.listdir(root_dir + "/"):
            if img == ".DS_Store":
                continue
            print("Reading img in ", root_dir + "/" + tf_dir)

            # Labeling for read image (decided by image dir name)
            lbl = 0
            if tf_dir == "true":
                lbl = 1
            elif tf_dir == "false":
                lbl = 0
            else:
                print("[WRN] Ignoring neither true nor false dir: ",tf_dir)
                continue

            for img in os.listdir(root_dir + "/" + tf_dir):
                if img == ".DS_Store":
                    continue

                img_filepath = root_dir + "/" + tf_dir + "/" + img
                image = Image.open(img_filepath).convert("RGB").resize((img_w,img_h))

                # Convert image onto numpy
                np_img = np.array(image)

                # Convert image (0-255) to (0.0 - 1.0)
                if root_dir == trainimg_rootdir:
                    train_img.append(np_img / 255.)
                    train_lbl.append(lbl)
                elif root_dir == testimg_rootdir:
                    test_img.append(np_img / 255.)
                    test_lbl.append(lbl)

        # Labeling array
        if root_dir == trainimg_rootdir:
            train_lblbin = to_categorical(train_lbl)
        elif root_dir == testimg_rootdir:
            test_lblbin = to_categorical(test_lbl)

    np_train_img = np.array(train_img)
    np_test_img = np.array(test_img)
    np_train_img.reshape(np_train_img.shape[0],img_w,img_h,img_ch)
    np_test_img.reshape(np_test_img.shape[0],img_w,img_h,img_ch)
    print("[OK] Read Img done. \n  Train:",np_train_img.shape,"\n  Test:",np_test_img.shape)
    # print("Train:")
    # print(train_lblbin)
    # print("Test")
    # print(test_lblbin)

    #Build CNN
    model = Sequential()

    ## CNN - input layer
    model.add(Conv2D(filters=32,
                    kernel_size=(4,4),
                    strides=(1,1),
                    activation='relu',
                    padding='valid',
                    input_shape=(img_w,img_h,img_ch)))

    ## CNN - Conv-layer and Pooling-layer
    model.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu'))
    model.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=64,kernel_size=(2,2),activation='relu'))
    model.add(Conv2D(filters=64,kernel_size=(2,2),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    # Fully-connected layer
    model.add(Flatten())
    model.add(Dense(32,activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(2,activation="softmax"))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=["accuracy"])

    model.summary() #For debug. Show summary

    history = model.fit(np_train_img,train_lblbin,batch_size=batch_size,
                        validation_split=validation_split,
                        verbose=1,
                        validation_data=(np_test_img,test_lblbin)
                        )

    plot_history(history)
