#from sunpy.image.rescale import resample
#from sunpy.image import resample
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
import time
#from Product import Product
import numpy
import netCDF4 as nc
import math
import random
import threading
import multiprocessing
import concurrent.futures
import itertools

from tqdm import tqdm

import logging

logging.basicConfig(level=logging.DEBUG, format="{asctime} {levelname:<8} {message}", style='{')
# %%

def JulianDate_to_MMDDYYY(y, jd):
    month = 1
    dayy = 0
    while jd - calendar.monthrange(y, month)[1] > 0 and month <= 12:
        jd = jd - calendar.monthrange(y, month)[1]
        month = month + 1
    return [y, month, jd]

def find_closest(Array, target):
    #Array must be sorted
    # A must be sorted
    if len(Array) == 1:
        return 0
    idx = Array.searchsorted(target)
    idx = np.clip(idx, 1, len(Array)-1)
    left = Array[idx-1]
    right = Array[idx]
    idx -= target - left < right - target
    return idx

def timestamp_extractor(path):
    #logging.debug(f'{path}')
    path = path.split('/')[-1]
    year = int(path.split('_')[3][1:5])
    day = int(path.split('_')[3][5:8])
    hour = int(path.split('_')[3][8:10])
    minute = int(path.split('_')[3][10:12])
    second = int(path.split('_')[3][12:14])
    milisecond = int(path.split('_')[3][14:16])

    year, month, day = JulianDate_to_MMDDYYY(year, day)

    timestamp = (datetime.datetime(year, month, day, hour, minute, second) - datetime.datetime(2000,1,1,12,00,00)).total_seconds()
    logging.info(f'{year},{month},{day},{hour},{minute},{second}')
    return timestamp

def getListOfFiles(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles

def getListOfFiles_fast(dirName):
    listOfFiles = list()
    for (dirpath, dirnames, filenames) in os.walk(dirName):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]
    return listOfFiles

def Raw_File_Name_Maker(offset,day,year,duration,step,gridSize_ABI,dirPath_GLM,dir_save):
    def name_checker(i, path, dir_save):
        #global rawFileNames_GLM
        timestamp = timestamp_extractor(path)
        flag = 0
        t_index = find_closest(timeArray, timestamp)
        #print(t_index)
        #print(datetime.timedelta(seconds=timestamp) + datetime.datetime(2000,1,1,12,00,00))

        if timestamp >= (timeArray[t_index]) and timestamp < (timeArray[t_index] + 15 * 60):
            flag = 1
            tt = t_index
        elif timestamp < (timeArray[t_index]) and timestamp < (timeArray[t_index-1]+15 * 60) and t_index!=0:
            flag = 1
            tt = t_index-1

        if flag == 1:
            t_index = tt
            #print("t_index_GLM=" + str(t_index)+"/"+str(len(timeArray)-1),filesList_GLM[i])
            if not int(t_index) in rawFileNames_GLM.keys():
                rawFileNames_GLM[int(t_index)] = []
            rawFileNames_GLM[int(t_index)].append(path)

            if timestamp > (timeArray[t_index] + 15 * 60-25):
                #print('\x1b[4;30;42m' +"Saving the corresponding dataTarget file:"+"dataTarget_ABI_" + "%03d" % (timeArray[t_index])+".npy"+'\x1b[0m')
                np.save(dir_save+"/rawFileNamesTarget_ABI_" + "%03d" % (timeArray[t_index]), rawFileNames_GLM[int(t_index)][:])

    filesList_GLM=[]
    for k in range(duration):
        try:
            filesList_GLM+=[getListOfFiles_fast(dirPath_GLM+"%03d" % (int(day) + k))]
        except Exception as e:
            logging.critical('Exeption occured: ', exc_info=True)
            print("GLM data for day #"+str(int(day)+k)+" was not found in the GLM directory.")

    a=JulianDate_to_MMDDYYY(int(year), int(day))
    t0 = datetime.datetime(2000, 1, 1, 12, 00, 00)
    t1=datetime.datetime (a[0],a[1],a[2],00,00,00)
    tStart=(t1-t0).total_seconds()+offset
    tEnd=tStart+24*duration*3600
    timeArray = linspace(tStart, tEnd,num=int(24/step)*duration,endpoint=False)
    #print(abs(tStart-timeArray)/3600)


    rawFileNames_GLM = {}
    for i,files_in_day in enumerate(filesList_GLM):
        for j,file in enumerate(files_in_day):
            try:
                name_checker(i, file, dir_save)
            except Exception as e:
                pass
                #logging.info(f'Exeption occured for file: {file}', exc_info=True)

def Data_Target_Maker(path_to_raw_files_name_file,dir_save):
    logging.debug('entered Data_Target_Maker')
    def find_index_fast(lat,lon): # x = lon, y = lat
        #logging.debug('entered find_index')
        #print(lat, lon)
        lat = lat * math.pi / 180
        lon = lon * math.pi / 180
        #print(lat, lon)
        x, y = latlon_to_xy(lat, lon)
        #i_x = np.searchsorted(gridX, x)
        i_x = find_closest(gridX, x)

        #i_y = np.searchsorted(gridY, y)
        i_y = find_closest(gridY, y)
        #print(i_x, i_y)
        #print('----------------')
        #logging.debug('exited find_index')
        return 1086 - i_y, i_x

    def find_index(lat,lon): # x = lon, y = lat
        dist_lon = lon - gridLon
        dist_lat = lat - gridLat
        distance = abs(dist_lon) + abs(dist_lat)
        #distance = dist_lon**2+ dist_lat**2
        #distance = dist_lon*dist_lon + dist_lat*dist_lat
        i_x,i_y = np.unravel_index(np.nanargmin(distance),(1086,1086))
        #[lat_index1, lon_index1] = numpy.where(distance1 == numpy.nanmin(distance1))
        #print(t1-t0,t2-t1,t3-t2)
        return i_x,i_y

    def calculate_circle_values(energy, area):
        # should consider 10*10 km per pixel and non uniform lat-lon meshgrid
        #logging.debug('entered calculate_circle_values')
        #logging.debug(f'{energy},{area}')
        r = np.sqrt(area/math.pi/(10*2))
        #r = np.sqrt(area/math.pi)
        val = energy/(area/(10*10))
        return r, val

    def draw_circle(map, i_x,i_y,r,val):
        #logging.debug('entered draw_circle')
        #logging.debug(f'{i_x},{i_y},{r},{val}')
        margin = int(np.round(r))
        for i in range(-margin,margin+1):
            for j in range(-margin,margin+1):
                if i**2 + j**2 <= r**2:
                    try:
                        map[i_x+i,i_y+j] += val
                    except:
                        continue
        #logging.debug('exited draw_circle')
        return


    def latlon_to_xy(lat,lon):# lat=\phi, lon=\lambda should be in radians
        H = 35786023.0
        r_eq = 6378137.0
        r_pol = 6356752.31414
        e = np.sqrt(1-(r_pol/r_eq)**2)
        lon_0 = -75 * math.pi/180
        #lat = lat * math.pi/180
        #lon = lon * math.pi/180

        phi_c = np.arctan((r_pol/r_eq)**2 * np.tan(lat))
        r_c = r_pol/np.sqrt(1- e**2 * np.cos(phi_c)**2)

        s_x = H - r_c * np.cos(phi_c) * np.cos(lon-lon_0)
        s_y = -r_c * np.cos(phi_c) * np.sin(lon-lon_0)
        s_z = r_c * np.sin(phi_c)

        y = np.arctan(s_z/s_x)
        x = np.arcsin(-s_y/np.sqrt(s_x**2 + s_y**2 + s_z**2))

        return (x, y)

    def xy_to_latlon(x,y):#x,y in radians
        H = 35786023.0
        r_eq = 6378137.0
        r_pol = 6356752.31414
        e = np.sqrt(1-(r_pol/r_eq)**2)
        lon_0 = -75 * math.pi/180

        a = np.sin(x)**2 + np.cos(x)**2 * (np.cos(y)**2 + (r_eq/r_pol)**2 * np.sin(y)**2)
        b = -2*H* np.cos(x) *np.cos(y)
        c = H**2 - r_eq**2

        r_s = (-b-np.sqrt(b**2 - 4*a*c))/2/a

        s_x = r_s * np.cos(x) * np.cos(y)
        s_y = -r_s * np.sin(x)
        s_z = r_s * np.cos(x) * np.sin(y)

        lat = np.arctan((r_eq/r_pol)**2 * s_z / (np.sqrt( (H-s_x)**2 + s_y**2 )))
        lon = lon_0 - np.arctan(s_y/(H-s_x))
        return (lat, lon)

    def draw_flashes(GLM,map,finder,r=None):
        #logging.debug('entered draw_flashes')
        for i in range(GLM.dimensions['number_of_flashes'].size):
            #print('x=',GLM.variables['flash_lon'][i],'y=',GLM.variables['flash_lat'][i])
            lon = GLM.variables['flash_lon'][i]
            lat = GLM.variables['flash_lat'][i]
            i_x, i_y = finder(lat, lon)
            energy = GLM.variables['flash_energy'][i]/GLM.variables['flash_energy'].scale_factor + GLM.variables['flash_energy'].add_offset
            area = GLM.variables['flash_area'][i]/GLM.variables['flash_area'].scale_factor + GLM.variables['flash_area'].add_offset
            if r !=None:
                val = energy
            else:
                r, val = calculate_circle_values(energy, area)

            try:
                draw_circle(map,i_x,i_y,r,val)
            except:
                #print(GLM.variables['flash_area'][i],GLM.variables['flash_area'].scale_factor,GLM.variables['flash_area'].add_offset)
                print(energy,',',area,',', r,',', val)
        #logging.debug('exited draw_flashes')

    def draw_groups(GLM,map,finder,r=None):
        for i in range(GLM.dimensions['number_of_groups'].size):
            #print('x=',GLM.variables['flash_lon'][i],'y=',GLM.variables['flash_lat'][i])
            lon = GLM.variables['group_lon'][i]
            lat = GLM.variables['group_lat'][i]
            i_x, i_y = finder(lat,lon)
            energy = GLM.variables['group_energy'][i]/GLM.variables['group_energy'].scale_factor + GLM.variables['group_energy'].add_offset
            area = GLM.variables['group_area'][i]/GLM.variables['group_area'].scale_factor + GLM.variables['group_area'].add_offset
            if r !=None:
                val = energy
            else:
                r, val = calculate_circle_values(energy, area)
            try:
                draw_circle(map,i_x,i_y,r,val)
            except:
                print(energy,',',area,',', r,',', val)

    def draw_events(GLM,map, finder):
        for i in range(GLM.dimensions['number_of_events'].size):
            #print('x=',GLM.variables['flash_lon'][i],'y=',GLM.variables['flash_lat'][i])
            lon = GLM.variables['event_lon'][i]
            lat = GLM.variables['event_lat'][i]
            i_x, i_y = finder(lat,lon)
            energy = GLM.variables['event_energy'][i]/GLM.variables['event_energy'].scale_factor + GLM.variables['event_energy'].add_offset
            #area = GLM.variables['group_area'][i]
            #r, val = calculate_circle_values(energy, area)
            r = 1
            val = energy
            #print(r, val)
            draw_circle(map,i_x,i_y,r,val)


    raw_file_names = np.load(path_to_raw_files_name_file)


    #gridSize=gridSize_ABI
    gridX = np.load('gridX.npy')
    gridY = np.load('gridY.npy')
    gridLat = np.load('gridLat.npy')
    gridLon = np.load('gridLon.npy')

    #productName_GLM=["GLM"]
    #productAtt_GLM=["flash"]

    lightning_map = np.zeros((1086,1086))
    for file in raw_file_names:
        #print(f'starting process of {file} ...')
        GLM = nc.Dataset(file,'r')
        draw_flashes(GLM,lightning_map,find_index_fast)
        draw_groups(GLM,lightning_map,find_index_fast)
        draw_events(GLM,lightning_map,find_index_fast)

    temp = path_to_raw_files_name_file.split('/')[-1]
    output_index = temp.split('_')[-1].strip('.npy')
    #np.save(dir_save+f"dataTarget_ABI_{timeArray[t_index]:03d}", response_ABI[:, t_index, :, :])
    np.save(dir_save+f"dataTarget_ABI_{output_index}", lightning_map)
    logging.debug('exited Data_Target_Maker')



def do_something():
    t = random.random()
    time.sleep(t)
    #print(f'waited for {t} sec')


def Parallel_Data_Target_Maker(path_to_raw_files_name_folder,dir_save):
    raw_file_names = os.listdir(path_to_raw_files_name_folder)
    raw_file_path = list(map(lambda name: path_to_raw_files_name_folder + name, raw_file_names))

    lim = len(raw_file_path)
    with tqdm(total=lim) as pbar:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {executor.submit(Data_Target_Maker, arg, dir_save): arg for arg in raw_file_path}
            #futures = {executor.submit(do_something): arg for arg in range(lim)}
            results = {}
            for future in concurrent.futures.as_completed(futures):
                arg = futures[future]
                results[arg] = future.result()
                pbar.update(1)

    #with  concurrent.futures.ProcessPoolExecutor() as executer:
    #    lim = 20
        #results = list(tqdm(executer.map(Data_Target_Maker, raw_file_path[:lim],itertools.repeat(dir_save)), total=lim))
        #results = list(tqdm(executer.map(do_something), total=lim))
        #results = [executer.submit(do_something) for i in range(lim)]
    return results

# %%
def main():
    mode='target'    #'input' or 'target'
    day='1' #day corresponding to the starting time [julian calender]
    year='2019'   #year corresponding to the starting time
    duration=366  #number of days from the starting time
    step=0.25   #time step [hour]
    offset=0  #offset in seconds added to the starting time
    gridSize_ABI=1086   #images width/height

    dirPath_GLM= f'/Volumes/LaCie/GLM_{year}/'  #directory where the raw files are stored
    dirSave_rawfilesname = f'/Volumes/LaCie/Ehsan/grouped_{year}/'
    dirSave_DataTarget = f'/Volumes/LaCie/Ehsan/dataTarget_{year}/'


    #Raw_File_Name_Maker(offset,day,year,duration,step,gridSize_ABI,dirPath_GLM,dirSave_rawfilesname)
    results = Parallel_Data_Target_Maker(dirSave_rawfilesname,dirSave_DataTarget)


if __name__ == '__main__':
    main()


timestamp_extractor('001/OR_ABI-L2-CPSF-M6_G16_s20222130000206_e20222130009514_c20222130012537.nc')
timestamp_extractor('001/OR_ABI-L2-RRQPEF-M6_G16_s20222440650206_e20222440659514_c20222440700006.nc')
