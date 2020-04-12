import os
import numpy as np
import time
import sys

from ChexnetTrainer import ChexnetTrainer_Binary_ResNet

def main ():
    
    #runTest()
    runTrain()

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
    nnArchitecture = 'RESNET-50'
    nnIsTrained = True

    #---- Training settings: batch size, maximum number of epochs
    trBatchSize = 64
    trMaxEpoch = 100
    
    #---- Parameters related to image transforms: size of the down-scaled image, cropped image
    imgtransResize = 256
    imgtransCrop = 224
        
    pathModel = nnArchitecture + '-' + timestampLaunch + '.pth.tar'
    
    #pathModel = 'm-06042020-134822.pth.tar'
    #pathModel = 'm-06042020-172102.pth.tar'

    print ('Training ...')
    ChexnetTrainer_Binary_ResNet.train(pathDirData, pathFileTrain, pathFileVal, nnIsTrained, trBatchSize, trMaxEpoch, imgtransResize, imgtransCrop, timestampLaunch, checkpoint=None)
    
    print ('Testing the trained model')
    ChexnetTrainer_Binary_ResNet.test(pathDirData, pathFileTest, pathModel, trBatchSize, imgtransResize, imgtransCrop, timestampLaunch)


def runTest():
    
    pathDirData = '/home/dxtien/dxtien_research/COVID/CXR8'
    pathFileTest = './dataset/test_1.txt'
    nnIsTrained = True
    nnClassCount = 1
    trBatchSize = 8
    imgtransResize = 256
    imgtransCrop = 224
    
    pathModel = './RESNET-121-FN-10042020-214158.pth.tar'
    
    timestampLaunch = ''
    
    ChexnetTrainer_Binary_ResNet.test(pathDirData, pathFileTest, pathModel, trBatchSize, imgtransResize, imgtransCrop, timestampLaunch)

if __name__ == '__main__':
    main()