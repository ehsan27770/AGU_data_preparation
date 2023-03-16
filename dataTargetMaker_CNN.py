#from sunpy.image.rescale import resample
from sunpy.image import resample
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
#from sunpy.extern.six.moves import range
import os
from netCDF4 import Dataset
import numpy as np
import scipy.interpolate
import scipy.ndimage
from numpy import empty
from numpy import zeros
from numpy import linspace
from scipy.io import loadmat
from scipy.io import savemat
#from getListOfFiles import fullPath
from numpy import savetxt
import calendar
import datetime
from numpy import meshgrid
import timeit
from Product import Product
import numpy

def dataTargetMaker_CNN (offset,day,year,duration,step,gridSize_ABI,dirPath_GLM,dirSave):
    modeCDS = 'off'
    gridSize_CDS=gridSize_ABI
    gridSize1=gridSize_ABI

    #offset =8*3600
    gridLat1=loadmat("gridLatMat.mat")
    gridLat1=gridLat1['gridLat']

    gridLon1=loadmat("gridLonMat.mat")
    gridLon1=gridLon1['gridLon']

    gridLat1= resample(gridLat1, (gridSize1, gridSize1), method='neighbor', center=False, minusone=True)
    gridLon1 = resample(gridLon1, (gridSize1, gridSize1), method='neighbor', center=False, minusone=True)

    if modeCDS=='on':
        gridSize2 = gridSize_CDS
        lat2 = np.arange(82, -82.25, -0.25)
        lon2 = np.arange(-157, 7.25, 0.25)
        [gridLon2, gridLat2] = meshgrid(lon2, lat2)

    #dirPath_GLM = "/Users/mostajab/Desktop/GOES-R/Data/214_GLM/"
    #dirPath_GLM = "/Volumes/LaCie/046"

    #filesList_GLM=os.listdir(filePath_GLM)


    filesList_GLM=[]
    for k in range(duration+1):
        try:
            #filesList_GLM+=fullPath(dirPath_GLM+"%03d" % (int(day) + k))
            filesList_GLM+=os.path.abspath(dirPath_GLM+"%03d" % (int(day) + k))
        except:
            print("GLM data for day #"+str(int(day)+k)+" was not found in the GLM directory.")
    print(len(filesList_GLM))
    #print(np.transpose(filesList_GLM))

    #filesList_GLM = np.load("fullList_GLM.npy")
    productName_GLM=["GLM"]
    productAtt_GLM=["flash"]

    def JulianDate_to_MMDDYYY(y, jd):
        month = 1
        dayy = 0
        while jd - calendar.monthrange(y, month)[1] > 0 and month <= 12:
            jd = jd - calendar.monthrange(y, month)[1]
            month = month + 1
        return [y, month, jd]

    a=JulianDate_to_MMDDYYY(int(year), int(day))
    print(int(day))
    print(a)
    t0 = datetime.datetime(2000, 1, 1, 12, 00, 00)
    t1=datetime.datetime (a[0],a[1],a[2],00,00,00)
    tStart=(t1-t0).total_seconds()+offset
    tEnd=tStart+24*duration*3600
    #timeArray=linspace(586440360,586525859,24)
    timeArray = linspace(tStart, tEnd,num=(24/step)*duration,endpoint=False)
    print(abs(tStart-timeArray)/3600)




    validList_GLM = []
    for k in range(duration):
        year_rev = int(year) + divmod(int(day) + k, 365)[0]
        day_rev = divmod(int(day) + k, 365)[1]
        validList_GLM += ["s" + "%03d" % (year_rev) + "%03d" % (day_rev)]
    print(validList_GLM)

    response_ABI = np.zeros([len(productName_GLM) + 3, len(timeArray), gridSize1, gridSize1],np.uint16)
    #response_ABI=empty([len(productName_GLM)+3,len(timeArray),gridSize1,gridSize1])
    #response_ABI[:,:,:,:]=np.nan
    rawFileNames_GLM = {}
    if modeCDS=='on':
        response_CDS = np.zeros([len(productName_GLM) + 3, len(timeArray), gridSize2, gridSize2], np.uint16)
        #response_CDS=empty([len(productName_GLM)+3,len(timeArray),gridSize2,gridSize2])
        #response_CDS[:,:,:,:]=np.nan

    flashCount = [0]
    flashEnergy = [0]



    def findClosest(A, target):
        #A must be sorted
        # A must be sorted
        if len(A) == 1:
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

        if mode=="GLM":
            t = dataset.variables['product_time'][:]
            flag=0
            t_index = findClosest(timeArray, t)
            #print(t,timeArray[t_index])
            if t >= (timeArray[t_index]) and t < (timeArray[t_index] + 15 * 60):
                flag=1
                tt=t_index
            elif t < (timeArray[t_index]) and t < (timeArray[t_index-1]+15 * 60) and t_index!=0:
                flag = 1
                tt = t_index-1
            else:
                print("...")

            if flag==1:
                t_index=tt
                print("t_index_GLM=" + str(t_index)+"/"+str(len(timeArray)-1),filesList_GLM[i])

                '''
                response_ABI[0, t_index, :, :]=numpy.nan_to_num(response_ABI[0,t_index,:,:])
                response_ABI[1, t_index, :, :]=numpy.nan_to_num(response_ABI[1,t_index,:,:])
                response_ABI[2, t_index, :, :] = numpy.nan_to_num(response_ABI[2, t_index, :, :])
                response_ABI[3, t_index, :, :] = numpy.nan_to_num(response_ABI[3, t_index, :, :])


                response_CDS[0, t_index, :, :] = numpy.nan_to_num(response_CDS[0, t_index, :, :])
                response_CDS[1, t_index, :, :] = numpy.nan_to_num(response_CDS[1, t_index, :, :])
                response_CDS[2, t_index, :, :] = numpy.nan_to_num(response_CDS[2, t_index, :, :])
                response_CDS[3,t_index, :, :] = numpy.nan_to_num(response_CDS[3, t_index, :, :])
                '''

                lat= []
                lon=[]
                distance=[]
                lat= dataset.variables["flash_lat"][:]
                lon= dataset.variables["flash_lon"][:]
                area = dataset.variables["flash_area"][:]
                energy= dataset.variables["flash_energy"][:]
                if not int(t_index) in rawFileNames_GLM.keys():
                    rawFileNames_GLM[int(t_index)] = []
                rawFileNames_GLM[int(t_index)].append(path)
                #print(rawFileNames_GLM[int(t_index)][:])
                #np.save("rawFileNames_GLM_" + "%03d" % (timeArray[t_index]), rawFileNames_GLM[int(t_index)][:])

                for flash_index in range(len(lat)):

                    distanceLat1 = abs(lat[flash_index] - gridLat1)
                    distanceLon1 = abs(lon[flash_index] - gridLon1)
                    distance1 = distanceLat1+ distanceLon1
                    #distance1_sqrt=np.sqrt(distanceLat1**2+ distanceLon1**2)
                    [lat_index1, lon_index1] = numpy.where(distance1 == numpy.nanmin(distance1))
                    #[lat_index1_sqrt, lon_index1_sqrt] = numpy.where(distance1_sqrt == numpy.nanmin(distance1_sqrt))
                    #print(gridLat1[500:505,500:505])
                    #print(lat[flash_index],abs(lat[flash_index]-gridLat1[lat_index1,lon_index1]),lon[flash_index],abs(lon[flash_index]-gridLon1[lat_index1,lon_index1]))
                    #print(abs(lat[flash_index]-gridLat1[lat_index1,lon_index1]),abs(lat[flash_index]-gridLat1[lat_index1_sqrt,lon_index1_sqrt]))
                    if modeCDS=='on':
                        distanceLat2 = abs(lat[flash_index] - gridLat2)
                        distanceLon2 = abs(lon[flash_index] - gridLon2)
                        distance2 = distanceLat2+ distanceLon2
                        [lat_index2, lon_index2] = numpy.where(distance2 == numpy.nanmin(distance2))
                        lat_index2 = lat_index2[0]
                        lon_index2 = lon_index2[0]
                        print([lat_index1,lon_index1],[lat_index2,lon_index2])

                    try:
                        response_ABI[product_index,t_index,int(lat_index1),int(lon_index1)]+=1
                        #print(response_ABI[product_index+1, t_index, int(lat_index1), int(lon_index1)], area[flash_index], flash_index)
                        response_ABI[product_index+1, t_index, int(lat_index1), int(lon_index1)] += area[flash_index]
                        response_ABI[product_index + 2, t_index, int(lat_index1), int(lon_index1)] += energy[flash_index]# /1.52597e-15
                        response_ABI[product_index + 3, t_index, int(lat_index1), int(lon_index1)] = 1
                        if modeCDS == 'on':
                            response_CDS[product_index, t_index, int(lat_index2), int(lon_index2)] += 1
                            response_CDS[product_index + 1, t_index, int(lat_index2), int(lon_index2)] += area[flash_index]
                            response_CDS[product_index + 2, t_index, int(lat_index2), int(lon_index2)] += energy[flash_index]
                            response_CDS[product_index + 3, t_index, int(lat_index2), int(lon_index2)] = 1
                    except:
                        np.save(dirSave+"/NotCountedFlash_GLM_"+"%03d" % (timeArray[t_index]), path)

                #print("NewflashCount=", str(flashCount))
                #print ("storedflashCount=",str(np.nansum(response[0,:,:,:])))
                #print(str(int(i / len(filesList_GLM)) * 100) + "% GLM complete.")
                if t > (timeArray[t_index] + 15 * 60-25):
                    #print("t_index_GLM=" + str(t_index) + "/" + str(len(timeArray) - 1), filesList_GLM[i])
                    print('\x1b[4;30;42m' +"Saving the corresponding dataTarget file:"+"dataTarget_ABI_" + "%03d" % (timeArray[t_index])+".npy"+'\x1b[0m')
                    try:
                        np.save(dirSave+"/dataTarget_ABI_" + "%03d" % (timeArray[t_index]), response_ABI[:, t_index, :, :])
                        if modeCDS=='on':
                            print('\x1b[4;30;42m' + "Saving the corresponding dataTarget file:" + "dataTarget_CDS_" + "%03d" % (timeArray[t_index]) + ".npy" + '\x1b[0m')
                            np.save(dirSave + "/dataTarget_CDS_" + "%03d" % (timeArray[t_index]),response_CDS[:, t_index, :, :])
                        # np.save(dirSave+"/dataTarget_CDS_" + "%03d" % (timeArray[t_index]), response_CDS[ :, t_index, :, :])
                        np.save(dirSave+"/rawFileNamesTarget_ABI_" + "%03d" % (timeArray[t_index]), rawFileNames_GLM[int(t_index)][:])
                    except:
                        print("GLM data for time #" + str(timeArray[t_index]) + " was not saved.")
                        np.save(dirSave+"/missingTimeInd_GLM_" + "%03d" % (timeArray[t_index]), (timeArray[t_index]))
                    print("-----------------------------------------------------------")

    products_GLM=[]
    for index in range(len(productName_GLM)):
        products_GLM.append(Product(productName_GLM[index],productAtt_GLM[index],0))

    for i in range(len(filesList_GLM)):

        if any(True for name in validList_GLM if name in filesList_GLM[i] ) and "_G16_" in filesList_GLM[i]:

            for j in range(len(productName_GLM)):
                if "_" + products_GLM[j].name+"-" in filesList_GLM[i]:
                    products_GLM[j].number += 1
                    dataMaker(i,j,filesList_GLM[i], products_GLM[j].mainAtt,"GLM")


    #print("Making the target arrays .....")

    '''
    for t_index in range (len(timeArray)):
        try:
            np.save("dataTarget_ABI_"+"%03d" % (timeArray[t_index]),response_ABI[:,t_index,:,:])
            #np.save("dataTarget_CDS_" + "%03d" % (timeArray[t_index]), response_CDS[ :, t_index, :, :])
            np.save("rawFileNames_GLM_" + "%03d" % (timeArray[t_index]), rawFileNames_GLM[int(t_index)][:])
        except:
            print("GLM data for time #" + str(timeArray[t_index]) + " was not saved.")
            np.save("missingTimeInd_GLM_" + "%03d" % (timeArray[t_index]), (timeArray[t_index]))
    '''
    return
