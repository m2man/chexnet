import os
import numpy as np
import time
import sys

from ChexnetTrainer import ChexnetTrainer, ChexnetTrainer_Binary, ChexnetTrainer_Binary_ResNet

#-------------------------------------------------------------------------------- 

def main ():
    
    runTest()
    #runTrain_Binary()
  
#--------------------------------------------------------------------------------   

def runTrain():
    
    DENSENET121 = 'DENSE-NET-121'
    DENSENET169 = 'DENSE-NET-169'
    DENSENET201 = 'DENSE-NET-201'
    
    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%d%m%Y")
    timestampLaunch = timestampDate + '-' + timestampTime
    
    #---- Path to the directory with images
    pathDirData = '/home/dxtien/dxtien_research/COVID/CXR8'
    
    #---- Paths to the files with training, validation and testing sets.
    #---- Each file should contains pairs [path to image, output vector]
    #---- Example: images_011/00027736_001.png 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    pathFileTrain = './dataset/train_1.txt'
    pathFileVal = './dataset/val_1.txt'
    pathFileTest = './dataset/test_1.txt'
    
    #---- Neural network parameters: type of the network, is it pre-trained 
    #---- on imagenet, number of classes
    nnArchitecture = DENSENET121
    nnIsTrained = True
    nnClassCount = 14
    
    #---- Training settings: batch size, maximum number of epochs
    trBatchSize = 16
    trMaxEpoch = 100
    
    #---- Parameters related to image transforms: size of the down-scaled image, cropped image
    imgtransResize = 256
    imgtransCrop = 224
        
    pathModel = 'm-' + timestampLaunch + '.pth.tar'
    
    print ('Training NN architecture = ', nnArchitecture)
    ChexnetTrainer.train(pathDirData, pathFileTrain, pathFileVal, nnArchitecture, nnIsTrained, nnClassCount, trBatchSize, trMaxEpoch, imgtransResize, imgtransCrop, timestampLaunch, None)
    
    print ('Testing the trained model')
    ChexnetTrainer.test(pathDirData, pathFileTest, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, imgtransResize, imgtransCrop, timestampLaunch)

#--------------------------------------------------------------------------------   

def runTrain_Binary():
    
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
    #nnArchitecture = 'RESNET-50'
    nnArchitecture = 'DENSENET-121'
    nnIsTrained = True
    nnClassCount = 2

    #---- Training settings: batch size, maximum number of epochs
    trBatchSize = 64
    trMaxEpoch = 50
    
    #---- Parameters related to image transforms: size of the down-scaled image, cropped image
    imgtransResize = 256
    imgtransCrop = 224
        
    pathModel = nnArchitecture + '-' + timestampLaunch + '.pth.tar'
    
    #pathModel = 'm-06042020-134822.pth.tar'
    #pathModel = 'm-06042020-172102.pth.tar'

    print ('Training ...')
    if nnArchitecture == 'RESNET-50':
        ChexnetTrainer_Binary_ResNet.train(pathDirData, pathFileTrain, pathFileVal, nnIsTrained, nnClassCount, trBatchSize, trMaxEpoch, imgtransResize, imgtransCrop, timestampLaunch, checkpoint='RESNET-50-10042020-030319.pth.tar')
    else:
        ChexnetTrainer_Binary.train(pathDirData, pathFileTrain, nnArchitecture, pathFileVal, nnIsTrained, nnClassCount, trBatchSize, trMaxEpoch, imgtransResize, imgtransCrop, timestampLaunch, checkpoint=None)
    
    print ('Testing the trained model')
    if nnArchitecture == 'RESNET-50':
        ChexnetTrainer_Binary_ResNet.test(pathDirData, pathFileTest, pathModel, nnClassCount, nnIsTrained, trBatchSize, imgtransResize, imgtransCrop, timestampLaunch)
    else:
        ChexnetTrainer_Binary.test(pathDirData, pathFileTest, nnArchitecture, pathModel, nnClassCount, nnIsTrained, trBatchSize, imgtransResize, imgtransCrop, timestampLaunch)

#-------------------------------------------------------------------------------- 

def runTest():
    
    pathDirData = '/home/dxtien/dxtien_research/COVID/CXR8'
    pathFileTest = './dataset/binary_test_14.txt'
    nnArchitecture = 'DENSE-NET-121'
    nnIsTrained = True
    nnClassCount = 14
    trBatchSize = 4
    imgtransResize = 256
    imgtransCrop = 224
    
    pathModel = './models/m-25012018-123527.pth.tar'
    #pathModel = './DENSENET-121-10042020-184520.pth.tar'

    timestampLaunch = ''
    
    ChexnetTrainer.test(pathDirData, pathFileTest, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, imgtransResize, imgtransCrop, timestampLaunch)

#-------------------------------------------------------------------------------- 

if __name__ == '__main__':
    main()





