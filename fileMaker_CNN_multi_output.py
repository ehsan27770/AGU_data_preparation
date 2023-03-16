import sys
import random
import numpy as np
import os, fnmatch, re
#from tqdm import tqdm
import argparse


#dir_input = '/home/emclab_epfl/data/dataTarget_processed_2019/'
#dir_output = '/home/emclab_epfl/data/dataSequenceCNN_2019_multi_output/'

parser = argparse.ArgumentParser("multi output file maker")
parser.add_argument("--input", help = "input folder address normally starts like dataTarget_processed_XXXX/")
parser.add_argument("--output", help = "output folder address normally starts like dataSequenceCNN_XXXX_multi_output/")
args = parser.parse_args()
dir_input = args.input
dir_output = args.output

print(dir_input,dir_output)
os.makedirs(dir_output, exist_ok=True)
os.makedirs(dir_output+'train/', exist_ok=True)
os.makedirs(dir_output+'val/', exist_ok=True)
os.makedirs(dir_output+'test/', exist_ok=True)

def get_sequences(dir_input, numHistory, numFuture):
    gridSize = 1086
    numTotal = numHistory + numFuture
    names = os.listdir(dir_input)
    names = [i for i in names if re.match(r'dataTarget_ABI_.*.npy',i)]
    #names = fnmatch.filter(names, 'dataTarget_ABI*')
    names.sort()
    time_stamp = np.array(list(map(lambda x:int(x.split("_")[2].split(".")[0]),names)))/900
    time_exact = np.array(list(map(lambda x:int(x.split("_")[2].split(".")[0]),names)))

    valid = []
    valid_train = []
    valid_val = []
    valid_test = []
    for i in range(len(time_stamp[:-numTotal+1])):
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
    random.shuffle(valid)
    length = len(valid)
    valid_train = valid[:int(0.7*length)]
    valid_val = valid[int(0.7*length):int(0.85*length)]
    valid_test = valid[int(0.85*length):]



    y = np.zeros((numFuture, gridSize, gridSize),np.uint16)
    z = np.zeros((numHistory, gridSize, gridSize),np.uint16)
    #t = []
    for valid_seq,folder in zip([valid_train,valid_val,valid_test],['train/','val/','test/']):
        count = -1
        t = []
        #for indx in tqdm(valid_seq):
        for indx in valid_seq:
            count = count + 1
            #t.append(time_stamp[indx])
            t.append(time_exact[indx])

            for i in range(numHistory):
                z[i,:,:] = np.load(dir_input+names[indx + i])[0:1,:,:]
            for i in range(numFuture):
                y[i,:,:] = np.load(dir_input+names[indx + numHistory + i])[0:1,:,:]

            np.save(dir_output + folder + 'y_{:04d}.npy'.format(count),y)
            np.save(dir_output + folder + 'x_{:04d}.npy'.format(count),z)

        t = np.array(t)
        np.save(dir_output + folder + 't.npy',t)

    return

def get_sequences_reduced_overlap(dir_input, numHistory, numFuture,numSkip):
    gridSize = 1086
    numTotal = numHistory + numFuture
    names = os.listdir(dir_input)
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
    random.shuffle(valid)
    length = len(valid)
    valid_train = valid[:int(0.7*length)]
    valid_val = valid[int(0.7*length):int(0.85*length)]
    valid_test = valid[int(0.85*length):]

    y = np.zeros((numFuture, gridSize, gridSize),np.uint16)
    z = np.zeros((numHistory, gridSize, gridSize),np.uint16)
    #t = []
    for valid_seq,folder in zip([valid_train,valid_val,valid_test],['train/','val/','test/']):
        count = -1
        t = []
        #for indx in tqdm(valid_seq):
        for indx in valid_seq:
            count = count + 1
            #t.append(time_stamp[indx])
            t.append(time_exact[indx])

            for i in range(numHistory):
                z[i,:,:] = np.load(dir_input+names[indx + i])[0:1,:,:]
            for i in range(numFuture):
                y[i,:,:] = np.load(dir_input+names[indx + numHistory + i])[0:1,:,:]

            np.save(dir_output + folder + 'y_{:04d}.npy'.format(count),y)
            np.save(dir_output + folder + 'x_{:04d}.npy'.format(count),z)

        t = np.array(t)
        np.save(dir_output + folder + 't.npy',t)

    return


def get_sequences_reduced_overlap_distinct_train_val_test(dir_input, numHistory, numFuture,numSkip):
    gridSize = 1086
    numTotal = numHistory + numFuture
    names = os.listdir(dir_input)
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
    #random.shuffle(valid)
    length = len(valid)
    valid_train = valid[:int(0.7*length)]
    valid_val = valid[int(0.7*length):int(0.85*length)]
    valid_test = valid[int(0.85*length):]

    y = np.zeros((numFuture, gridSize, gridSize),np.uint16)
    z = np.zeros((numHistory, gridSize, gridSize),np.uint16)
    #t = []
    for valid_seq,folder in zip([valid_train,valid_val,valid_test],['train/','val/','test/']):
        count = -1
        t = []
        #for indx in tqdm(valid_seq):
        for indx in valid_seq:
            count = count + 1
            #t.append(time_stamp[indx])
            t.append(time_exact[indx])

            for i in range(numHistory):
                z[i,:,:] = np.load(dir_input+names[indx + i])[0:1,:,:]
            for i in range(numFuture):
                y[i,:,:] = np.load(dir_input+names[indx + numHistory + i])[0:1,:,:]

            np.save(dir_output + folder + 'y_{:04d}.npy'.format(count),y)
            np.save(dir_output + folder + 'x_{:04d}.npy'.format(count),z)

        t = np.array(t)
        np.save(dir_output + folder + 't.npy',t)

    return


def get_sequences_reduced_overlap_periodic_and_distinct_train_val_test(dir_input, numHistory, numFuture, numSkip):#final and correct one
    gridSize = 1086
    numTotal = numHistory + numFuture
    names = os.listdir(dir_input)
    names = [i for i in names if re.match(r'dataTarget_ABI_.*.npy',i)]

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

            valid.append(i)
    # the special part where all of the previous functions are different
    #random.shuffle(valid)
    period = 5*24*4 # 5 days, number of timestamps in 5 days, 5days, 24 hour, every 15 min (4 times per hour)
    length = len(valid)
    index_all = np.array(list(range(length)))

    index_train = index_all[(index_all%period<0.7*period)]
    index_val = index_all[(index_all%period>=0.7*period) & (index_all%period<0.85*period)]
    index_test = index_all[(index_all%period>=0.85*period)]

    valid_train = list(np.array(valid)[index_train])
    valid_val = list(np.array(valid)[index_val])
    valid_test = list(np.array(valid)[index_test])

    y = np.zeros((numFuture, gridSize, gridSize),np.uint16)
    z = np.zeros((numHistory, gridSize, gridSize),np.uint16)
    #t = []
    for valid_seq,folder in zip([valid_train,valid_val,valid_test],['train/','val/','test/']):
        count = -1
        t = []
        #for indx in tqdm(valid_seq):
        for indx in valid_seq:
            count = count + 1
            #t.append(time_stamp[indx])
            t.append(time_exact[indx])

            for i in range(numHistory):
                z[i,:,:] = np.load(dir_input+names[indx + i])[0:1,:,:]
            for i in range(numFuture):
                y[i,:,:] = np.load(dir_input+names[indx + numHistory + i])[0:1,:,:]

            np.save(dir_output + folder + 'y_{:04d}.npy'.format(count),y)
            np.save(dir_output + folder + 'x_{:04d}.npy'.format(count),z)

        t = np.array(t)
        np.save(dir_output + folder + 't.npy',t)

    return

numHistory = 8
numFuture = 6
#numSkip = (numHistory + numFuture)//2
numSkip = 3

#get_sequences_reduced_overlap(dir_input,numHistory, numFuture,numSkip)
#get_sequences_reduced_overlap_distinct_train_val_test(dir_input,numHistory, numFuture,numSkip)
get_sequences_reduced_overlap_periodic_and_distinct_train_val_test(dir_input,numHistory, numFuture,numSkip)
