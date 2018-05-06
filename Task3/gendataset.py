# Using all xml file to generate dataset
# read the xml files in metainfo/
# generate a *.npz data in feature_label_data/

from util import parselabel,parseAnswer
from time import time
import numpy as np
import os

if __name__ == "__main__":
    # setting parameter for MFCCs
    n_mfcc = 13
    featuretype = 'mfcc_delta'
    # define the variables for feature and label
    if featuretype == 'mfcc_delta':
        train = []
        test = []
        valid = []
    elif featuretype == 'mfcc_cov':
        train = []
        test = []
        valid = []
    Y_train = []
    Y_test = []
    Y_valid = []
    chroma = []
    # Use os.walk to go through all the audio file under folder "audio/"
    print("Start generating dataset with feature type " + featuretype + " ...")
    t0 = time()
    
    for dirPath, _, fileNames in os.walk("Train"):
        if fileNames == []:
            continue
        for f in fileNames:
            X = parselabel(os.path.join(
                os.path.join(dirPath, f)),featuretype)
            #train = np.append(train, [X],axis = 0)
            train.append(X)
    print('train Finish')
    for dirPath, _, fileNames in os.walk("Test"):
        if fileNames == []:
            continue
        for f in fileNames:
            X = parselabel(os.path.join(
                os.path.join(dirPath, f)),featuretype)
            test.append(X)
    print('test Finish')
    for dirPath, _, fileNames in os.walk("Valid"):
        if fileNames == []:
            continue
        for f in fileNames:
            X = parselabel(os.path.join(
                os.path.join(dirPath, f)),featuretype)
            valid.append(X)  
    print('valid Finish')
    idx=0
    for dirPath, _, fileNames in os.walk("answer_train"):
        if fileNames == []:
            continue
        for f in fileNames:
            Y_train.append((parseAnswer(os.path.join(
                os.path.join(dirPath, f)),len(train[idx]))))
            idx+=1
    print('ans_train Finish')
    idx =0
    for dirPath, _, fileNames in os.walk("answer_test"):
        if fileNames == []:
            continue
        for f in fileNames:
            Y_test .append((parseAnswer(os.path.join(
                os.path.join(dirPath, f)),len(test[idx]))))
            idx+=1
    print('ans_test Finish')
    idx =0
    for dirPath, _, fileNames in os.walk("answer_valid"):
        if fileNames == []:
            continue
        for f in fileNames:
            Y_valid .append((parseAnswer(os.path.join(
                os.path.join(dirPath, f)),len(valid[idx]))))
            idx+=1
    print('ans_valid Finish')
    print(len(train),len(test),len(valid))
    print(len(Y_train),len(Y_test),len(Y_valid))
    print('Finished in {:4.2f} sec!'.format(time() - t0))
    print('Collect {:d} samples totally.'.format(len(test)+len(train)+len(valid)))
    datasetNames = 'ML_dataset_' + featuretype + '_' + str(n_mfcc) + '.npz'
    datasetpath = os.path.join('feature_label_data', datasetNames)
    print('Dataset is saved at "feature_label_data/'+ datasetNames +'".')
    
    np.savez(datasetpath, test=test,train=train,valid=valid,ans_train=Y_train,ans_test = Y_test,ans_valid = Y_valid)
    
