#from sunpy.image.rescale import resample
from sunpy.image import resample
import matplotlib.pyplot as plt
import numpy
import netCDF4
from netCDF4 import Dataset
import numpy as np
#from sunpy.extern.six.moves import range
import os
from netCDF4 import Dataset
import numpy as np
from numpy import empty
from numpy import zeros
from numpy import linspace
from scipy.io import loadmat
from scipy.io import savemat
import numpy as np
#from getListOfFiles import fullPath
from numpy import savetxt
import calendar
import datetime
import timeit
#from Product import Product


def dataInputMaker_CNN (offset,day,year,duration,step,gridSize,dirPath_ABI,dirSave):

    #offset=2*3600
    gridLat=loadmat("gridLatMat.mat")
    gridLat=gridLat['gridLat']

    gridLon=loadmat("gridLonMat.mat")
    gridLon=gridLon['gridLon']

    gridLat= resample(gridLat, (gridSize, gridSize), method='neighbor', center=False, minusone=True)
    gridLon = resample(gridLon, (gridSize, gridSize), method='neighbor', center=False, minusone=True)
    #filepath = '/Users/mostajab/Desktop/GOES-R/Data/214_Atmospheric/OR_ABI-L2-ACHAF-M3_G16_s20182140000417_e20182140011184_c20182140012095.nc'
    #dirPath_ABI = "/Users/mostajab/Desktop/GOES-R/Data/214_Atmospheric/"
    #dirPath_ABI ="/Volumes/LaCie/GOES-R/Data/ABI/001"

    #filesList_ABI=fullPath(dirPath_ABI)
    filesList_ABI=os.path.abspath(dirPath_ABI)

    #filesList_ABI=np.load("fullList_ABI_2.npy")
    #filesList_ABI=os.listdir(filePath_ABI)

    productName_ABI=["ACHA","ACHT","ACTP","CTP","COD","RRQPE","AOD","DSI","TPW","LST","CPS"]
    productAtt_ABI=["HT","TEMP","Phase","PRES","COD","RRQPE","AOD","CAPE","TPW","LST","PSD"]

    def JulianDate_to_MMDDYYY(y, jd):
        month = 1
        while jd - calendar.monthrange(y, month)[1] > 0 and month <= 12:
            jd = jd - calendar.monthrange(y, month)[1]
            month = month + 1
        return [y, month, jd]

    a=JulianDate_to_MMDDYYY(int(year), int(day))
    print(a)
    t0 = datetime.datetime(2000, 1, 1, 12, 00, 00)
    t1=datetime.datetime (a[0],a[1],a[2],00,00,00)
    tStart=(t1-t0).total_seconds()+offset
    tEnd=tStart+24*duration*3600
    #timeArray=linspace(586440360,586525859,24)
    timeArray = linspace(tStart, tEnd,num=(24/step)*duration,endpoint=False)
    print(abs(tStart-timeArray)/3600)
    grid=empty([gridSize,gridSize])
    for i in range(gridSize):
        for j in range(gridSize):
            grid[i,j]=i*gridSize+j

    validList_ABI=[]
    for k in range(duration):
        year_rev=int(year)+divmod(int(day)+k,365)[0]
        day_rev=divmod(int(day)+k,365)[1]
        validList_ABI += ["s" +"%03d" % (year_rev)+"%03d" % (day_rev)]
    print(validList_ABI)


    #features=empty([len(productName_ABI),len(timeArray),gridSize,gridSize])
    #features_mask=empty([len(productName_ABI),len(timeArray),gridSize,gridSize])
    #features[:,:,:,:]=np.nan
    features= np.zeros((len(productName_ABI), len(timeArray), gridSize, gridSize), np.uint16)
    features_mask=np.zeros((len(productName_ABI),len(timeArray), gridSize, gridSize), np.uint8)
    rawFileNames_ABI = [[0 for j in range(len(productName_ABI))] for i in range(len(timeArray))]

    '''
    for i in range(len(timeArray)):
        features[len(productName_ABI),i,:,:]=grid
        features_mask[len(productName_ABI), i, :, :] = grid
    '''

    def findClosest(A, target):
        #A must be sorted
        if len(A)==1:
            return 0
        idx = A.searchsorted(target)
        idx = np.clip(idx, 1, len(A)-1)
        left = A[idx-1]
        right = A[idx]
        idx -= target - left < right - target
        return idx

    def dataMaker(i,product_index,path,mainAtt,mode):
        try:
            dataset = Dataset(path)
        except:
            print("Error: Unknown file format")
            return
        if mode=="ABI":
            t = dataset.variables['t'][:]
            t_index = findClosest(timeArray, t)
            if t < (timeArray[t_index] - 7 * 60) or t > (timeArray[t_index] + 7 * 60):
                #print("...")
                return
            print(filesList_ABI[i])
            print("t_index_ABI=" + str(t_index) + "/" + str(len(timeArray) - 1))
            #print(str(i)+"/"+str(len(filesList_ABI))+ " of ABI files complete.")
            print("-----------------------------------------------------------")
            temp = []
            temp_mask=[]
            temp = dataset.variables[mainAtt][:][:]
            try:
                temp_mask=dataset.variables["DQF"][:][:]
            except:
                temp_mask = dataset.variables["DQF_Overall"][:][:]
            if mainAtt == "Phase":
                #temp = temp.filled(5)
                temp = temp.filled(fill_value=0)
            else:
                temp = temp.filled(fill_value=0)
                #temp = temp.filled(np.nan)

            tempResampled = resample(temp, (gridSize, gridSize), method='neighbor', center=False, minusone=True)
            tempResampled_mask = resample(temp_mask, (gridSize, gridSize), method='neighbor', center=False, minusone=True)
            features[product_index, t_index, :, :] =tempResampled
            features_mask[product_index, t_index, :, :] = tempResampled_mask
            rawFileNames_ABI[t_index][product_index] = path
            '''
            fig = plt.figure()
            ax1 = fig.add_subplot(121)
            ax1.title.set_text(productName_ABI[product_index] + "_Original")
            r=round(temp_mask.shape[0] / gridSize)
            im1 = ax1.imshow(temp_mask[r*850: r * 895, r * 630:r * 710])
            fig.colorbar(im1)
            ax2 = fig.add_subplot(122)
            ax2.title.set_text(productName_ABI[product_index] + "_Saved")
            im2 = ax2.imshow(tempResampled_mask[850:895, 630:710])
            fig.colorbar(im2)
            plt.show()
            features[product_index, t_index, :, :] = tempResampled
            '''

    for i in range(len(filesList_ABI)):
        if any(True for name in validList_ABI if name in filesList_ABI[i]):
            for j in range(len(productName_ABI)):
                if "-" + productName_ABI[j] in filesList_ABI[i]:
                    dataMaker(i, j, filesList_ABI[i], productAtt_ABI[j], "ABI")


    #dataInput=features.reshape((features.shape[0],features.shape[1]*features.shape[2]*features.shape[3]))
    #dataInput=numpy.transpose(dataInput)
    #dataInput=features
    #np.save("dataInput_ABI_" + day + "_" + "%03d" % (int(day) + duration - 1), dataInput)
    for t_index in range (len(timeArray)):
        print('\x1b[0;30;46m' + "Saving the corresponding dataInput file:" + "dataInput_ABI_" + "%03d" % (
        timeArray[t_index]) + ".npy" + '\x1b[0m')
        try:
            np.save(dirSave+"/dataInput_ABI_"+"%03d" % (timeArray[t_index]),np.concatenate((features[:,t_index,:,:],features_mask[:,t_index,:,:])))
            np.save(dirSave+"/rawFileNamesInput_ABI_"+"%03d" % (timeArray[t_index]),rawFileNames_ABI[t_index][:])
        except:
            print("ABI data for time #" + str(timeArray[t_index]) + " was not saved.")
            #np.save(dirSave+"/missingTimeInd_ABI_" + "%03d" % (timeArray[t_index]), (timeArray[t_index]))
    return
