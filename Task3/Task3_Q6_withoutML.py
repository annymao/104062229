# -*- coding: utf-8 -*-
"""
Created on Fri May  4 22:44:20 2018

@author: anny
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 22:00:04 2018

@author: anny
"""
import numpy as np
import librosa.feature
import scipy.io.wavfile as wav
import scipy.signal
import matplotlib 
import matplotlib.pyplot as plt

font = {'family' : 'sans-serif', 'sans-serif':'Arial', 'size'   : 18}
matplotlib.rc('font', **font)

# Generate major chord templates
Major_template = np.array([[6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88]])
# Generate monor chord templates
Minor_template = np.array([[6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17]])

Template = Major_template
for i in range(11):
    Template = np.append(Template, np.roll(Major_template, i+1), axis=0)    
for i in range(12):
    Template = np.append(Template, np.roll(Minor_template, i), axis=0)
 
#for debug
d = {'C':0,'C+':1, 'D-':1, 'D':2, 'E-':3, 'E':4, 'F':5,'F+':6, 'G-':6,\
           'G':7, 'A-':8, 'A':9, 'B-':10, 'B':11,\
           'c':12, 'c+':13, 'd-':13,'d':14, 'd+':15,'e-':15, 'e':16, 'f':17, 'f+':18,\
           'g':19,'g+':20, 'a-':20, 'a':21, 'b-':22, 'b':23}
Key = ['C', 'D-', 'D', 'E-', 'E', 'F', 'G-',\
           'G', 'A-', 'A', 'B-', 'B',\
           'c', 'c+', 'd', 'e-', 'e', 'f', 'f+',\
           'g', 'a-', 'a', 'b-', 'b'] 
acc = 0;         
music_pieces = 100;
#manually change path to each genre and test each song     
GENRES = "metal"      

dataset_path = 'feature_label_data/ML_dataset_mfcc_delta_13.npz'
dataset = np.load(dataset_path)
train_examples = dataset['train']
ans_train =dataset['ans_train']
test_examples = dataset['test']
ans_test = dataset['ans_test']


ans_train = np.append(ans_train,ans_test)
train_examples = np.append(train_examples,test_examples)

idx = 0
while idx < len(ans_train):
    f = open('out/out_2_{0}.txt'.format(idx),'w')
    acc = 0;
    train = train_examples[idx]
    data =  ans_train[idx]
    t = 1
    idx +=1  
    print(len(train))
    print(len(data))
    while t < len(data):
        
        temp_ma = Template[0]
        temp_ma -= np.sum(temp_ma)/12
        temp_mi=Template[12]
        temp_mi -= np.sum(temp_mi)/12
        MajorCof = np.dot(temp_ma,train[t])/np.sqrt(np.multiply(np.dot(train[t],train[t]),np.dot(temp_ma,temp_ma)))
        MinCof = np.dot(temp_mi,train[t])/np.sqrt(np.multiply(np.dot(train[t],train[t]),np.dot(temp_mi,temp_mi)))
        for i in range(1,12):
            temp_ma = Template[i]
            temp_ma -= np.sum(temp_ma)/12
            temp_mi=Template[i+12]
            temp_mi -= np.sum(temp_mi)/12
            MajorCof = np.append(MajorCof,np.dot(temp_ma,train[t])/np.sqrt(np.multiply(np.dot(train[t],train[t]),np.dot(temp_ma,temp_ma))))
             
            MinCof = np.append(MinCof,np.dot(temp_mi,train[t])/np.sqrt(np.multiply(np.dot(train[t],train[t]),np.dot(temp_mi,temp_mi))))
        cof = MajorCof
        cof = np.append(cof,MinCof)
        key = np.argmax(cof)
        
        #print(t,Key[key])
        
        ansKey = data[t][1]
        t+=1
        wr = str(t) + '   ' + Key[key]+' '
        f.write(wr)
        
        
        #print("Ans: ",ansKey)
        f.write(ansKey+'\n')
        if(key<12):
            if key == d[ansKey]:
                acc = acc+1
                
                f.write("correct!\n")
            elif (key+7)%12 == d[ansKey]:
                acc = acc + 0.5
                f.write("fifthPerfect\n")
            elif (key + 9)%12+12 == d[ansKey]:   
                acc = acc + 0.3
                f.write("Relative\n")
            elif key + 12 == d[ansKey]:
                acc = acc + 0.2
                f.write("Parallel\n")         
        else:
            if key == d[ansKey]:
                acc = acc+1
                f.write("correct!\n")
            elif (key-5)%12+12 == d[ansKey]:
                acc = acc + 0.5
                f.write("fifthPerfect\n")
            elif (key-9)%12== d[ansKey]:   
                acc = acc + 0.3
                f.write("Relative\n")
            elif key - 12 == d[ansKey]:
                acc = acc + 0.2
                f.write("Parallel\n") 
    f.write('Accuracy: '+ str(acc/len(data)))
    print(acc/len(data))        
    f.close()
