import os
import numpy as np
import time
import sys

from ChexnetTrainer import ChexnetTrainer_Binary_FN

def main ():
    
    runTest()
    #runTrain()

def runTrain():
    
    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%d%m%Y")
    timestampLaunch = timestampDate + '-' + timestampTime
    
    #---- Path to the directory with images
    pathDirData = '/home/dxtien/dxtien_research/COVID/CXR8'
    
    #---- Paths to the files with training, validation and testing sets.
    #---- Each file should contains pairs [path to image, output vector]
    #---- Example: images_011/00027736_001.png 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    pathFileTrain = './dataset/binary_train.txt'
    pathFileVal = './dataset/binary_validate.txt'
    pathFileTest = './dataset/binary_test.txt'
    
    #---- Neural network parameters: type of the network, is it pre-trained 
    #---- on imagenet, number of classes
    nnArchitecture = 'DENSENET-121-FN'
    nnIsTrained = True

    #---- Training settings: batch size, maximum number of epochs
    trBatchSize = 64
    trMaxEpoch = 50
    
    #---- Parameters related to image transforms: size of the down-scaled image, cropped image
    imgtransResize = 256
    imgtransCrop = 224
        
    pathModel = nnArchitecture + '-' + timestampLaunch + '.pth.tar'
    
    checkpoint = None #'DENSENET-121-FN-11042020-083506.pth.tar'

    print ('Training ...')
    ChexnetTrainer_Binary_FN.train(pathDirData, pathFileTrain, pathFileVal, nnIsTrained, trBatchSize, trMaxEpoch, imgtransResize, imgtransCrop, timestampLaunch, checkpoint=checkpoint)
    
    print ('Testing the trained model')
    ChexnetTrainer_Binary_FN.test(pathDirData, pathFileTest, pathModel, trBatchSize, imgtransResize, imgtransCrop, timestampLaunch)


def runTest():
    
    pathDirData = '/home/dxtien/dxtien_research/COVID/CXR8'
    pathFileTest = './dataset/binary_test.txt'
    nnIsTrained = True
    nnClassCount = 1
    trBatchSize = 32
    imgtransResize = 256
    imgtransCrop = 224
    
    pathModel = './DENSENET-121-FN-11042020-235423.pth.tar'
    
    timestampLaunch = ''
    
    ChexnetTrainer_Binary_FN.test(pathDirData, pathFileTest, pathModel, trBatchSize, imgtransResize, imgtransCrop, timestampLaunch)

if __name__ == '__main__':
    main()