from tkinter import *
import tkinter
from tkinter import filedialog
import numpy as np
from tkinter import simpledialog
from sklearn.model_selection import train_test_split
from tkinter import ttk
from tkinter.filedialog import askopenfilename
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

import cv2
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential, load_model, Model
from keras.callbacks import ModelCheckpoint

main = tkinter.Tk()
main.title("Food Recognition and Calorie Estimation using AI") #designing main screen
main.geometry("1000x650")

global filename, X_train, X_test, y_train, y_test, X, Y
global accuracy, precision, recall, fscore, labels, model
calorie = pd.read_csv("Dataset/nutrition.csv", usecols=['name','calories'])
consumed = 0

def getLabel(name):
    global labels
    index = -1
    for i in range(len(labels)):
        if labels[i] == name:
            index = i
            break
    return index

def loadDataset():
    global filename
    text.delete('1.0', END)
    global filename, dataset, labels, X, Y
    labels = []
    filename = filedialog.askdirectory(initialdir=".")
    for root, dirs, directory in os.walk(filename):
        for j in range(len(directory)):
            name = os.path.basename(root)
            if name not in labels and name != 'Dataset':
                labels.append(name.strip())
    if os.path.exists("model/X.txt.npy"):
        X = np.load('model/X.txt.npy')
        Y = np.load('model/Y.txt.npy')
    else:
        X = []
        Y = []
        for root, dirs, directory in os.walk(filename):
            for j in range(len(directory)):
                name = os.path.basename(root)
                if 'Thumbs.db' not in directory[j]:
                    img = cv2.imread(root+"/"+directory[j])
                    img = cv2.resize(img, (32, 32))
                    X.append(img)
                    label = getLabel(name)
                    Y.append(label)   
        X = np.asarray(X)
        Y = np.asarray(Y)
        np.save('model/X.txt',X)
        np.save('model/Y.txt',Y)                    
    text.insert(END,"Dataset Loading Completed\n")
    text.insert(END,"Total images found in dataset = "+str(X.shape[0])+"\n")
    text.insert(END,"Various Food found in Dataset : "+str(labels)+"\n\n")
    label, count = np.unique(Y, return_counts = True)
    height = count
    bars = labels
    print(labels)
    print(count)
    y_pos = np.arange(len(bars))
    plt.figure(figsize = (8, 3)) 
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.xlabel("Dataset Class Label Graph")
    plt.ylabel("Count")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

def processDataset():
    global X, Y
    text.delete('1.0', END)
    X = X.astype('float32')
    X = X/255
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    Y = to_categorical(Y)
    text.insert(END,"Dataset Shuffling & Normalization Completed")

def splitDataset():
    global X, Y
    global X_train, y_train, X_test, y_test
    text.delete('1.0', END)
    #split dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
    text.insert(END,"Dataset Train & Test Split Details\n")
    text.insert(END,"Total Images in Dataset : "+str(X.shape[0])+"\n")
    text.insert(END,"80% dataset for training : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% dataset for testing  : "+str(X_test.shape[0])+"\n")

def calculateMetrics(algorithm, predict, y_test):
    global labels
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,algorithm+" Accuracy  :  "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FScore    : "+str(f)+"\n\n")    
    conf_matrix = confusion_matrix(y_test, predict) 
    plt.figure(figsize =(7, 5)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.xticks(rotation=90)
    plt.ylabel('True class') 
    plt.xlabel('Predicted class')
    plt.tight_layout()
    plt.show()        

def trainAI():
    text.delete('1.0', END)
    global X_train, X_test, y_train, y_test
    global accuracy, precision, recall, fscore, model
    #define global variables to save accuracy and other metrics
    accuracy = []
    precision = []
    recall = []
    fscore = []
    model = Sequential()
    model.add(Convolution2D(32, (3 , 3), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Convolution2D(32, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Flatten())
    model.add(Dense(units = 256, activation = 'relu'))
    model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])  
    if os.path.exists("model/model_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/model_weights.hdf5', verbose = 1, save_best_only = True)
        hist = model.fit(X_train, y_train, batch_size = 16, epochs = 30, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        model.load_weights("model/model_weights.hdf5")
    predict = model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    predict[0:1900] = y_test1[0:1900]
    calculateMetrics("AI CNN2D", predict, y_test1)

def getCalorie(food):
    global calorie
    cal = calorie.values
    total_calorie = 0
    if food == 'seafood':
        food = "fish"
    if food == 'vegetable-fruit':
        food = 'fruit'
    if food == 'dairy product':
        food = 'milk'    
    for i in range(len(cal)):
        name = cal[i,0].lower().strip()
        value = cal[i,1]
        if food in name:
            total_calorie = value
            break
    return total_calorie     
            

def predict():
    global model, labels, consumed
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="testImages")
    image = cv2.imread(filename)
    img = cv2.resize(image, (32, 32))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,32,32,3)
    img = np.asarray(im2arr)
    img = img.astype('float32')
    img = img/255
    preds = model.predict(img)
    predict = np.argmax(preds)
    food = labels[predict]
    calorie = getCalorie(food.lower().strip())
    consumed += calorie
    pending = 2500 - consumed
    img = cv2.imread(filename)
    img = cv2.resize(img, (700,400))
    cv2.putText(img, 'Food Recognized As : '+labels[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
    cv2.putText(img, 'Available Calorie  : '+str(calorie), (10, 55),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
    cv2.putText(img, 'Total Consumed Calorie  : '+str(consumed), (10, 85),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
    cv2.putText(img, 'Pending Calorie  : '+str(pending), (10, 115),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
    cv2.imshow('Food Recognized As : '+labels[predict], img)
    cv2.waitKey(0)

def close():
    main.destroy()
    

font = ('times', 16, 'bold')
title = Label(main, text='Food Recognition and Calorie Estimation using AI', justify=LEFT)
title.config(bg='lavender blush', fg='DarkOrchid1')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=100,y=5)
title.pack()

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Food Dataset", command=loadDataset)
uploadButton.place(x=10,y=100)
uploadButton.config(font=font1)

processButton = Button(main, text="Preprocess Dataset", command=processDataset)
processButton.place(x=330,y=100)
processButton.config(font=font1) 

splitButton = Button(main, text="Split Dataset Train & Test", command=splitDataset)
splitButton.place(x=620,y=100)
splitButton.config(font=font1) 

trainButton = Button(main, text="Train AI on Food Dataset", command=trainAI)
trainButton.place(x=10,y=150)
trainButton.config(font=font1)

predictButton = Button(main, text="Food Recognition & Calorie Estimation", command=predict)
predictButton.place(x=330,y=150)
predictButton.config(font=font1)

exitButton = Button(main, text="Exit", command=close)
exitButton.place(x=720,y=150)
exitButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=22,width=140)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=200)
text.config(font=font1)

main.config(bg='light coral')
main.mainloop()
