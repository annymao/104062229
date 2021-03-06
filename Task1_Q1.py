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
Major_template = np.array([[1,0,1,0,1,1,0,1,0,1,0,1]])/np.sqrt(3.0)
# Generate monor chord templates
Minor_template = np.array([[1,0,1,1,0,1,0,1,1,0,1,0]])/np.sqrt(3.0)

Template = Major_template
for i in range(11):
    Template = np.append(Template, np.roll(Major_template, i+1), axis=0)    
for i in range(12):
    Template = np.append(Template, np.roll(Minor_template, i), axis=0)
 
#for debug
Key = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#',\
           'G', 'G#', 'A', 'A#', 'B',\
           'c', 'c#', 'd', 'd#', 'e', 'f', 'f#',\
           'g', 'g#', 'a', 'a#', 'b'] 
        

#manually change path to each genre and test each song     
#GENRES = "blues"      
"""
------------------------ACCURACY-----------------
gama     100      |    10     |    1
ROCK    0.2755    |  0.31633  | 0.33673
HIPHOP  0.12345   |  0.14814  | 0.13580
POP     0.35106   |  0.40426  | 0.38297
BLUES   0.0408    |  0.0408   | 0.06122
METAL   0.1935    |  0.1935   | 0.21505
-------------------------------------------------
"""

for GAMA in [1,10,100,1000]:
    f = open('output/Q1_{0}.txt'.format(GAMA),'w')
    for GENRES in ['rock','hiphop','pop','blues','metal']:
        music_pieces = 100;
        acc = 0; 
        for fileLen in range(100):
            path = 'genres/{0}/{0}.00{1:03}.au'.format(GENRES,fileLen)
            x, fs = librosa.load(path, sr=None)
            ansKey = int(open('gtzan_key-master/gtzan_key/genres/{0}/{0}.00{1:03}.lerch.txt'.format(GENRES,fileLen),'r').read())
        
            f.write('{0}.00{1:03}    '.format(GENRES,fileLen))
            if x.dtype != 'float32': # deal with the case of integer-valued data
                x = np.float32(x/32767.)
            if x.ndim > 1: # I want to deal only with single-channel signal now
                x = np.mean(x, axis = 1)
            
            Chroma = librosa.feature.chroma_stft(y=x, sr=fs)
            Chroma = Chroma/np.tile(np.sum(np.abs(Chroma)**2, axis=0)**(1./2), \
                                (Chroma.shape[0], 1))
            
            
            Chroma = np.log10(1+GAMA *np.abs(Chroma))
            #spectral smoothing
            """
            Len = 5
            for i in range(Chroma.shape[1]):
                Chroma[:,i] = np.sum(Chroma[:,i- int(Len/2):int(i+Len/2)],axis = 1)/Len
            """
            
            sumChroma =np.sum(Chroma,axis = 1)
            tonic = np.argmax(sumChroma)
            #substract mean x - x bar
            sumChroma -= np.sum(sumChroma) /sumChroma.shape[0]
        
            temp_ma = Template[tonic]
            temp_ma -= np.sum(temp_ma)/12
            temp_mi=Template[tonic+12]
            temp_mi -= np.sum(temp_mi)/12
            MajorCof = np.dot(temp_ma,sumChroma)/ np.sqrt(np.multiply(np.dot(sumChroma,sumChroma),np.dot(temp_ma,temp_ma)))
             
            MinorCof = np.dot(temp_mi,sumChroma)/np.sqrt(np.multiply(np.dot(sumChroma,sumChroma),np.dot(temp_mi,temp_mi)))
            if(ansKey == -1):
                music_pieces -=1
                f.write('\n')
                continue
            if(ansKey == 0):
                ansKey = 9
            elif(ansKey == 1):
                ansKey = 10
            elif (ansKey == 2):
                ansKey = 11
            elif(ansKey == 12):
                ansKey = 21
            elif (ansKey == 13):
                ansKey = 22
            elif (ansKey == 14):
                ansKey = 23
            else :
                ansKey = ansKey-3
        
            if(tonic >=0 and tonic <= 11):
                if(MajorCof>MinorCof):
                    print(tonic,Key[tonic]+" Major")
                    f.write(Key[tonic]+" Major\n")
                    if tonic == ansKey:
                        acc = acc+1
                else:
                    print((tonic+12),Key[tonic+12]+" Minor")
                    f.write(Key[tonic+12]+" Minor\n")
                    if tonic+12 == ansKey:
                        acc=acc+1 
                #print("Ans: ",Key[ansKey])
        print("#music_pieces: ",music_pieces)
        if music_pieces !=0:
            f.write("Accuracy: "+str(acc/music_pieces)+'\n')
            #print('accuracy: ',acc/music_pieces)
    f.close()