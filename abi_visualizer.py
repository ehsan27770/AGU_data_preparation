# %% imports
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import cartopy.crs as ccrs
import math
from sunpy.image import resample
from scipy.io import loadmat
import time
import datetime
# %% data
ABI = nc.Dataset('OR_ABI-L2-ACHAF-M3_G16_s20183200000334_e20183200011101_c20183200012129.nc','r')
GLM = nc.Dataset('OR_GLM-L2-LCFA_G16_s20180572159400_e20180572200000_c20180572200028.nc','r')
#GLM = nc.Dataset('OR_GLM-L2-LCFA_G16_s20180462355000_e20180462355200_c20180462355226.nc','r')

'''
gridLat=loadmat("gridLatMat.mat")
gridLat=gridLat['gridLat']

gridLon=loadmat("gridLonMat.mat")
gridLon=gridLon['gridLon']

gridX = np.array(ABI.variables['x'])
gridY = np.flip(np.array(ABI.variables['y']))
'''

# %%
'''
np.save('gridX.npy',gridX,allow_pickle=False)
np.save('gridY.npy',gridY,allow_pickle=False)
np.save('gridLat.npy',gridLat,allow_pickle=False)
np.save('gridLon.npy',gridLon,allow_pickle=False)
'''

gridX = np.load('gridX.npy')
gridY = np.load('gridY.npy')
gridLat = np.load('gridLat.npy')
gridLon = np.load('gridLon.npy')

# %%
import os
def get_types(dirName):
    types = set()
    for (dirpath, dirnames, filenames) in os.walk(dirName):
        #types.add() += [os.path.join(dirpath, file) for file in filenames]
        map(lambda x: types.add(x.split('G16')[0]),filenames)
    return types
#get_types('.')

#names = os.listdir('.')
#temp = map(lambda x : x.split('G16')[0], names)
#types = set(temp)
#types = ['OR_ABI-L2-ACHTF-M3_', 'OR_ABI-L2-CTPF-M3_', 'OR_ABI-L2-ACHAF-M3_', 'OR_ABI-L2-AODF-M3_', 'OR_ABI-L2-CODF-M3_', 'OR_ABI-L2-ACTPF-M3_', 'OR_ABI-L2-LSTF-M3_', 'OR_ABI-L2-CPSF-M3_', 'OR_ABI-L2-DSIF-M3_', 'OR_ABI-L2-TPWF-M3_', 'OR_ABI-L2-RRQPEF-M3_']
types = ['.BC.T_ITAjLj', '.BC.T_kTl1a5', '.BC.T_bMBABA', '.BC.T_VjWlCH', 'OR_ABI-L2-AODF-M4_', 'OR_ABI-L2-DSIF-M4_', '.BC.T_n8Op1v', 'OR_ABI-L2-CPSF-M4_', 'OR_ABI-L2-TPWF-M3_', '.BC.T_fsvBE5', 'OR_ABI-L2-ACHTF-M3_', 'OR_ABI-L2-ACHAF-M4_', '.BC.T_82EHYR', 'OR_ABI-L2-RRQPEF-M3_', 'OR_ABI-L2-CODF-M3_', 'OR_ABI-L2-CTPF-M4_', 'OR_ABI-L2-ACTPF-M3_', '.DS_Store', '.BC.T_xfyWam', '.BC.T_hDnoEK', 'OR_ABI-L2-LSTF-M4_', 'OR_ABI-L2-AODF-M3_', '.BC.T_5IdHFe', '.BC.T_uQgJqU', 'OR_ABI-L2-ACHTF-M4_', '.BC.T_9hq1Vh', 'OR_ABI-L2-DSIF-M3_', 'OR_ABI-L2-TPWF-M4_', '.BC.T_syo9fp', 'OR_ABI-L2-ACTPF-M4_', 'OR_ABI-L2-CTPF-M3_', '.BC.T_dfWHOC', 'OR_ABI-L2-ACHAF-M3_', 'OR_ABI-L2-RRQPEF-M4_', 'OR_ABI-L2-LSTF-M3_', '.BC.T_JXcaQa', '.BC.T_GaBHvM', '.BC.T_D75gSn', 'OR_ABI-L2-CPSF-M3_', '.BC.T_CvqPjy', 'OR_ABI-L2-CODF-M4_']

types = ['OR_ABI-L2-AODF-M4_', 'OR_ABI-L2-DSIF-M4_', 'OR_ABI-L2-CPSF-M4_', 'OR_ABI-L2-TPWF-M3_', 'OR_ABI-L2-ACHTF-M3_', 'OR_ABI-L2-ACHAF-M4_', 'OR_ABI-L2-RRQPEF-M3_', 'OR_ABI-L2-CODF-M3_', 'OR_ABI-L2-CTPF-M4_', 'OR_ABI-L2-ACTPF-M3_','OR_ABI-L2-LSTF-M4_', 'OR_ABI-L2-AODF-M3_','OR_ABI-L2-ACHTF-M4_','OR_ABI-L2-DSIF-M3_', 'OR_ABI-L2-TPWF-M4_','OR_ABI-L2-ACTPF-M4_', 'OR_ABI-L2-CTPF-M3_', 'OR_ABI-L2-ACHAF-M3_', 'OR_ABI-L2-RRQPEF-M4_', 'OR_ABI-L2-LSTF-M3_', 'OR_ABI-L2-CPSF-M3_', 'OR_ABI-L2-CODF-M4_']
# %%

#ax = plt.axes(projection=ccrs.Geostationary(central_longitude=0.0, satellite_height=35785831, false_easting=0, false_northing=0, globe=None))
ax = plt.axes(projection=ccrs.Geostationary(central_longitude=-75.0, satellite_height=35786023.0, false_easting=0, false_northing=0, globe=None))
plt.gcf().set_size_inches(10, 10)
#ax = plt.axes(projection=ccrs.PlateCarree())
#ax.set_extent([105, 135, 20, 40])
ax.stock_img()
ax.coastlines()
#ax.scatter(df['Longitude'],df['Latitude'],alpha=0.1,transform=ccrs.PlateCarree(),zorder=100)
ax.scatter(gridLon[::25,::25],gridLat[::25,::25],alpha=0.5,transform=ccrs.PlateCarree(),zorder=100,marker='.')
#plt.savefig('linear_latlon.png')
plt.show()


# %%
ax = plt.axes()
plt.gcf().set_size_inches(10, 10)
#ax = plt.axes(projection=ccrs.PlateCarree())
#ax.set_extent([105, 135, 20, 40])
#ax.stock_img()
#ax.coastlines()
#ax.scatter(df['Longitude'],df['Latitude'],alpha=0.1,transform=ccrs.PlateCarree(),zorder=100)
ax.scatter(gridLon[::25,::25],gridLat[::25,::25],alpha=0.5,zorder=100,marker='.')
#plt.savefig('linear_latlon.png')
plt.show()


# %%
ax = plt.axes()
plt.gcf().set_size_inches(10, 10)

X, Y = latlon_to_xy(gridLat*math.pi/180, gridLon*math.pi/180)
ax.scatter(X[::25,::25]*180/math.pi,Y[::25,::25]*180/math.pi,alpha=0.5,zorder=100,marker='.')
#plt.savefig('linear_latlon.png')
plt.show()


# %%
X,Y = np.meshgrid(gridX,gridY)
lat,lon = xy_to_latlon(X,Y)
ax = plt.axes(projection=ccrs.Geostationary(central_longitude=-75.0, satellite_height=35786023.0, false_easting=0, false_northing=0, globe=None))
plt.gcf().set_size_inches(10, 10)
#ax = plt.axes(projection=ccrs.PlateCarree())
#ax.set_extent([105, 135, 20, 40])
ax.stock_img()
ax.coastlines()
#ax.scatter(df['Longitude'],df['Latitude'],alpha=0.1,transform=ccrs.PlateCarree(),zorder=100)
ax.scatter(lon[::25,::25]*180/math.pi,lat[::25,::25]*180/math.pi,alpha=0.5,transform=ccrs.PlateCarree(),zorder=100,marker='.')
#plt.savefig('linear_xy.png')
plt.show()

# %%
X,Y = np.meshgrid(gridX,gridY)
lat,lon = xy_to_latlon(X,Y)
ax = plt.axes()
plt.gcf().set_size_inches(10, 10)
ax.scatter(lon[::25,::25]*180/math.pi,lat[::25,::25]*180/math.pi,alpha=0.5,zorder=100,marker='.')
#plt.savefig('linear_xy.png')
plt.show()

# %%
X,Y = np.meshgrid(gridX,gridY)
#lat,lon = xy_to_latlon(X,Y)
ax = plt.axes()
plt.gcf().set_size_inches(10, 10)
ax.scatter(X[::25,::25]*180/math.pi,Y[::25,::25]*180/math.pi,alpha=0.5,zorder=100,marker='.')
plt.savefig('linear_xy.png')
plt.show()
