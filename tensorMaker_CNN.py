import random
import numpy as np
import os, fnmatch


gridSize = 1086
numSamples = 100
leadTime = 0

dir_Input = '/dataInput_processed_2019/'
dir_Target = '/media/ehsan/Ehsan Files/ubuntu extra space/dataTarget_processed_2019/'


inputInd = [2,4,5,7,8,9,10]
targetInd = [0]
numChannels = 22

#Physics free-related settings
numHistory = 11
numFuture = 4

def get_data_fileName (dir_Input,dir_Target):

    # X: atmospheric predictors
    # y: lightning data for future tim windows (i.e., target tensor)
    # z: lightning data for previous time windows (i.e., input tensor in the physics-free approach)
    # t: The time index of the gathered data

    X = np.zeros((numSamples, numChannels, gridSize, gridSize),np.uint16)
    y = np.zeros((numSamples, numFuture, gridSize, gridSize),np.uint16)
    z = np.zeros((numSamples, numHistory, gridSize, gridSize),np.uint16)
    t = []
    listFiles = fnmatch.filter(os.listdir(dir_Target), 'dataTarget_ABI*')
    random.shuffle(listFiles)
    #listFiles=listFiles[:numSamples]
    index=0
    for fileName in listFiles:
        if index<numSamples:
            t_index=fileName.split("_")[2]
            print("index=" + str(index + 1) + "/" + str(numSamples) + ', time slice: ' + str(t_index.split(".")[0]))
            try:
                #X[index,:,:,:]=np.load(dir_Input+"dataInput_ABI_"+str(int(t_index.split(".")[0])-leadTime*60)+".npy")

                for j in range(numFuture):
                    temp = np.load(dir_Target+"dataTarget_ABI_"+str(int(t_index.split(".")[0])+(j)*15*60)+".npy")
                    y[index,j,:,:] = temp[0:1,:,:]

                for k in range(numHistory):
                    temp = np.load(dir_Target+"dataTarget_ABI_"+str(int(t_index.split(".")[0])-(k+1)*15*60-leadTime*60)+".npy")
                    z[index, k, :, :] = temp[0,:,:]

                index = index + 1
                t.append(t_index.split(".")[0])
            except:
                print("Data not found for this time slice...Searching for an available time slice")
                #X[index, :, :, :] = np.zeros((numChannels, gridSize, gridSize),np.uint16)
                #y[index, :, :, :] = np.zeros((4, gridSize, gridSize),np.uint16)
                #print("Data not found for this time slice...Filled with zeros instead")
    return X,y,z,t


X,y,z,t = get_data_fileName (dir_Input,dir_Target)
#y=np.sign(y)
#X=X[:,inputInd,:,:]
#y=y[:,targetInd,:,:]
print(X.shape,y.shape,z.shape,len(t))

#np.save('/Users/mostajab/Desktop/preLight/dataX/XN_2018_2_'+str(leadTime)+'.npy',X)
np.save('/Users/mostajab/Desktop/preLight/datay/yN_2019_5_'+str(leadTime)+'.npy',y[:,:,:, :])
np.save('/Users/mostajab/Desktop/preLight/dataz/zN_2019_5_'+str(leadTime)+'.npy',z[:,:,:, :])
np.save('/Users/mostajab/Desktop/preLight/datat/tN_2019_5_'+str(leadTime)+'.npy',t)
