# -*- coding: utf-8 -*-
"""
Created on Sun May  6 14:08:08 2018

@author: anny
"""

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np


dataset_path = 'feature_label_data/dataset_mfcc_delta_13.npz'
dataset = np.load(dataset_path)
train_examples = dataset['train']
ans_train =dataset['ans_train']
test_examples = dataset['test']
ans_test = dataset['ans_test']
valid_examples = dataset['valid']
ans_valid = dataset['ans_valid']
d = {'C':0,'C+':1, 'D-':1, 'D':2, 'E-':3, 'E':4, 'F':5,'F+':6, 'G-':6,\
           'G':7, 'A-':8, 'A':9, 'B-':10, 'B':11,\
           'c':12, 'c+':13, 'd-':13,'d':14, 'd+':15,'e-':15, 'e':16, 'f':17, 'f+':18,\
           'g':19,'g+':20, 'a-':20, 'a':21, 'b-':22, 'b':23}
Key = ['C', 'D-', 'D', 'E-', 'E', 'F', 'G-',\
           'G', 'A-', 'A', 'B-', 'B',\
           'c', 'c+', 'd', 'e-', 'e', 'f', 'f+',\
           'g', 'a-', 'a', 'b-', 'b'] 

x_train = np.array([train_examples[0][0]])
y_train = np.array([d[ans_train[0][0][1]]])
print(x_train.shape)
for j in range(1,len(ans_train[0])):
    x_train = np.append(x_train,np.array([train_examples[0][j]]),axis = 0)
    y_train = np.append(y_train,np.array([d[ans_train[0][j][1]]]),axis = 0)
idx = 1

while idx < len(ans_train):
    j=0
    for j in range(1,len(ans_train[idx])):
        x_train = np.append(x_train,np.array([train_examples[idx][j]]),axis = 0)
        y_train = np.append(y_train,np.array([d[ans_train[idx][j][1]]]),axis = 0)
    idx+=1
    
x_test = np.array([test_examples[0][0]])
y_test = np.array([d[ans_test[0][0][1]]])
print(x_train.shape)
for j in range(1,len(ans_test[0])):
    x_test = np.append(x_test,np.array([test_examples[0][j]]),axis = 0)
    y_test = np.append(y_test,np.array([d[ans_test[0][j][1]]]),axis = 0)
idx = 1

while idx < len(ans_test):
    j=0
    for j in range(1,len(ans_test[idx])):
        x_test = np.append(x_test,np.array([test_examples[idx][j]]),axis = 0)
        y_test = np.append(y_test,np.array([d[ans_test[idx][j][1]]]),axis = 0)
    idx+=1
    
x_valid = np.array([valid_examples[0][0]])
y_valid = np.array([d[ans_valid[0][0][1]]])
print(x_valid.shape)
for j in range(1,len(ans_valid[0])):
    x_valid = np.append(x_valid,np.array([valid_examples[0][j]]),axis = 0)
    y_valid = np.append(y_valid,np.array([d[ans_valid[0][j][1]]]),axis = 0)
idx = 1

while idx < len(ans_valid):
    j=0
    for j in range(1,len(ans_valid[idx])):
        x_valid = np.append(x_valid,np.array([valid_examples[idx][j]]),axis = 0)
        y_valid = np.append(y_valid,np.array([d[ans_valid[idx][j][1]]]),axis = 0)
    idx+=1

    
print(x_train.shape,y_train.shape)
y_train = keras.utils.to_categorical(y_train, num_classes=24)

y_test = keras.utils.to_categorical(y_test, num_classes= 24)

y_valid = keras.utils.to_categorical(y_valid, num_classes= 24)



model = Sequential()
# Dense(64) 是一个具有 64 个隐藏神经元的全连接层。
# 在第一层必须指定所期望的输入数据尺寸：
# 在这里，是一个 20 维的向量。
model.add(Dense(64, activation='relu', input_dim=12))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(24, activation='softmax'))

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=100,
          batch_size=32,validation_data=(x_valid, y_valid))
score = model.evaluate(x_test, y_test, batch_size=128)
f = open('out/Q5.txt','w')
for t in range(0,len(ans_test)):
    
    p = model.predict(x_test[0:len(ans_test[t])-1])
    print(p.shape)
    acc = 0;
    for i in range(0, p.shape[0]):
        key = np.argmax(p[i])
        print (i,Key[key],ans_test[t][i])
        if(key<12):
            if key == d[ans_test[t][i][1]]:
                acc = acc+1
                print("correct!")
            elif (key+7)%12 == d[ans_test[t][i][1]]:
                acc = acc + 0.5
                print("fifthPerfect")
            elif (key + 9)%12+12 == d[ans_test[t][i][1]]:   
                acc = acc + 0.3
                print("Relative")
            elif key + 12 == d[ans_test[t][i][1]]:
                acc = acc + 0.2
                print("Parallel")         
        else:
            if key == d[ans_test[t][i][1]]:
                acc = acc+1
            elif (key-5)%12+12 == d[ans_test[t][i][1]]:
                acc = acc + 0.5
                print("fifthPerfect")
            elif (key-9)%12== d[ans_test[t][i][1]]:   
                acc = acc + 0.3
                print("Relative")
            elif key - 12 == d[ans_test[t][i][1]]:
                acc = acc + 0.2
    f.write(str(t)+' ')
    f.write(str(acc/p.shape[0]))
    f.write('\n')
    print (acc/p.shape[0])
f.close()