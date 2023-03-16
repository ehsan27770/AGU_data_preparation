from dataInputMaker_CNN import dataInputMaker_CNN
from dataTargetMaker_CNN import dataTargetMaker_CNN
import timeit

'''**********Define the input arguments**********'''

mode='target'    #'input' or 'target'
day_start='250' #day corresponding to the starting time [julian calender]
year_start='2018'   #year corresponding to the starting time
duration=9  #number of days from the starting time
step=0.25   #time step [hour]
offset=0  #offset in seconds added to the starting time
gridSize=1086   #images width/height
dir_read= '/Volumes/LaCie/GLM_2018/'  #directory where the raw files are stored
dir_save= '/Volumes/LaCie/dataTarget_processed_2018/' #directory where to save the processed files

dir_save= '/Volumes/LaCie/test_for_ehsan/' #directory where to save the processed files
mode = 'input'


'''********** Making the processed data (input/target) **********'''

t0 = timeit.default_timer()
if (mode=='input'):
        dataInputMaker_CNN(offset,day_start,year_start,duration,step,gridSize,dir_read,dir_save)
elif (mode=='target'):
    dataTargetMaker_CNN (offset,day_start,year_start,duration,step,gridSize,dir_read,dir_save)
t1 = timeit.default_timer()
print(t1-t0)
