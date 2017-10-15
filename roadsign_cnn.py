from keras.models import Sequential
from keras.layers import Activation,Dense,Dropout
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adagrad, Adam, RMSprop
from PIL import Image
import numpy as np
from matplotlib import pylab as plt
import os

epoch = 1500
batch_size = 100
validation_split = 0.1
img_w = 50
img_h = 50
img_elem = 3 #R,G,B

trainimg_rootdir = "./roadsign/train"
testimg_rootdir = "./roadsign/test"

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
        print("Reading img in ", root_dir + "/" + tf_dir)

#        for img_dir in os.listdir(root_dir + "/" + tf_dir):
#            if img_dir == ".DS_Store":
#                continue

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
            print("Reading img ",img_filepath)
            image = Image.open(img_filepath).convert("RGB").resize((img_w,img_h))
            # image.show()
            #Make Image URL

            # Convert image onto numpy
            np_img = np.array(image)

            # Convert Numpy (Exchange Row and Column)
            #  [R,G,B],[R,G,B],.. --> [R,R,...],[G,G,..],[B,B..]
            np_img = np_img.transpose(2,0,1)

            # Convert numpy(Combine R,G,B array)
            # [R,R,..][G,G..][B,B..] --> [R,R,..,G,G,..,B,B..]
            np_img = np_img.reshape(1,np_img.shape[0] * np_img.shape[1] * np_img.shape[2]).astype("float32")[0]

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

    print("[OK] Read Img done. Train:",len(train_img),"(" , len(train_lblbin) ,")  Test:", len(test_img), "(" , len(test_lblbin) , ")")
    print(" Train_Lbl:",train_lbl , " Test_lbl:",test_lbl)
    print(" Train_lblbin",train_lblbin, " Test_lblbin=",test_lblbin)

train_img = np.array(train_img)
test_img = np.array(test_img)

#Build CNN
model = Sequential()
model.add(Dense(200,input_dim=(img_w * img_h * img_elem)))
model.add(Activation("relu"))
model.add(Dropout(0.2))

model.add(Dense(200))
model.add(Activation("relu"))
model.add(Dropout(0.2))

model.add(Dense(2))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy",
              optimizer=RMSprop(),
              metrics=["accuracy"])

model.summary() #For debug. Show summary

model.fit(train_img,train_lblbin,batch_size=batch_size,
            validation_split=validation_split,
            verbose=1,
            validation_data=(test_img,test_lblbin))
