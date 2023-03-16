import random
import numpy as np
import os, fnmatch, re



def get_sequences(dir_input, numHistory, numFuture):
    gridSize = 1086
    numTotal = numHistory + numFuture
    names = os.listdir(dir_input)
    names = [i for i in names if re.match(r'dataTarget_ABI_.*.npy',i)]
    #names = fnmatch.filter(names, 'dataTarget_ABI*')
    names.sort()
    time_stamp = np.array(list(map(lambda x:int(x.split("_")[2].split(".")[0]),names)))/900

    valid = []
    for i in range(len(time_stamp[:-numTotal+1])):
        is_valid = True
        for j in range(numTotal-1):
            step = time_stamp[i+j+1] - time_stamp[i+j]
            if step != 1:
                is_valid = False
                break
        if is_valid:
            valid.append(i)

    y = np.zeros((numFuture, gridSize, gridSize),np.uint16)
    z = np.zeros((numHistory, gridSize, gridSize),np.uint16)
    t = []

    count = -1
    for indx in valid:
        count = count + 1
        t.append(time_stamp[indx])
        for i in range(numHistory):
            z[i,:,:] = np.load(dir_input+names[indx + i])[0:1,:,:]
        for i in range(numFuture):
            y[i,:,:] = np.load(dir_input+names[indx + numHistory + i])[0:1,:,:]

        np.save(dir_output + 'y_{:04d}.npy'.format(count),y)
        np.save(dir_output + 'x_{:04d}.npy'.format(count),z)

    t = np.array(t)
    np.save(dir_output + 't.npy',t)
    print(y.shape,z.shape,t.shape)

    return

def get_sequences_reduced_overlap(dir_input, numHistory, numFuture, numSkip):
    gridSize = 1086
    numTotal = numHistory + numFuture
    names = os.listdir(dir_input)
    names = [i for i in names if re.match(r'dataTarget_ABI_.*.npy',i)]
    #names = fnmatch.filter(names, 'dataTarget_ABI*')
    names.sort()
    time_stamp = np.array(list(map(lambda x:int(x.split("_")[2].split(".")[0]),names)))/900

    valid = []
    for i in range(0,len(time_stamp[:-numTotal+1]),numSkip):
        is_valid = True
        for j in range(numTotal-1):
            step = time_stamp[i+j+1] - time_stamp[i+j]
            if step != 1:
                is_valid = False
                break
        if is_valid:
            valid.append(i)

    y = np.zeros((numFuture, gridSize, gridSize),np.uint16)
    z = np.zeros((numHistory, gridSize, gridSize),np.uint16)
    t = []

    count = -1
    for indx in valid:
        count = count + 1
        t.append(time_stamp[indx])
        for i in range(numHistory):
            z[i,:,:] = np.load(dir_input+names[indx + i])[0:1,:,:]
        for i in range(numFuture):
            y[i,:,:] = np.load(dir_input+names[indx + numHistory + i])[0:1,:,:]

        np.save(dir_output + 'y_{:04d}.npy'.format(count),y)
        np.save(dir_output + 'x_{:04d}.npy'.format(count),z)

    t = np.array(t)
    np.save(dir_output + 't.npy',t)
    print(y.shape,z.shape,t.shape)

    return



dir_input = './dataTarget_processed_2019/'
dir_output = './test/'

numHistory = 10
numFuture = 4

get_sequences(dir_input,numHistory, numFuture)

dir_input = './dataTarget_processed_2019/'
dir_output = './test/'

numHistory = 8
numFuture = 6
numSkip = (numHistory + numFuture)//2

get_sequences_reduced_overlap(dir_input,numHistory, numFuture,numSkip)

# %%
numHistory = 6
numFuture = 8
numSkip = (numHistory + numFuture)//2
numSkip = 1

gridSize = 1086
numTotal = numHistory + numFuture
names = os.listdir('dataTarget_processed_2019/')
names = [i for i in names if re.match(r'dataTarget_ABI_.*.npy',i)]
#names = fnmatch.filter(names, 'dataTarget_ABI*')
names.sort()
time_stamp = np.array(list(map(lambda x:int(x.split("_")[2].split(".")[0]),names)))/900
time_exact = np.array(list(map(lambda x:int(x.split("_")[2].split(".")[0]),names)))

valid = []
valid_train = []
valid_val = []
valid_test = []
for i in range(0,len(time_stamp[:-numTotal+1]),numSkip):
    is_valid = True
    for j in range(numTotal-1):
        step = time_stamp[i+j+1] - time_stamp[i+j]
        if step != 1:
            is_valid = False
            break
    if is_valid:
        #if i//(4*24)%4 != 3:
        #    valid_train.append(i)
        #elif i//(4*24)%8 == 3:
        #    valid_val.append(i)
        #elif i//(4*24)%8 == 7:
        #    valid_test.append(i)
        valid.append(i)
len(valid)
