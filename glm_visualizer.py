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

# %% test
#gridLat1= resample.resample(gridLat, (1086, 1086), method='nearest', center=False, minusone=True)


# %%



t_p1 = 0
t_p2 = 0
t_p3 = 0
t_p4 = 0
# %% helper functions
def find_closest(Array, target):
    #Array must be sorted
    # A must be sorted
    if len(Array) == 1:
        return 0
    idx = Array.searchsorted(target)
    #idx = np.clip(idx, 1, len(A)-1)
    left = Array[idx-1]
    right = Array[idx]
    idx -= target - left < right - target
    return idx

def find_index_fast(lat,lon): # x = lon, y = lat
    global t_p1
    global t_p2
    global t_p3
    t0 = time.perf_counter()
    #print(lat, lon)
    lat = lat * math.pi / 180
    lon = lon * math.pi / 180
    #print(lat, lon)
    x, y = latlon_to_xy(lat, lon)
    #i_x = np.searchsorted(gridX, x)
    i_x = find_closest(gridX, x)




    t1 = time.perf_counter()
    #i_y = np.searchsorted(gridY, y)
    i_y = find_closest(gridY, y)
    t2 = time.perf_counter()
    #print(i_x, i_y)
    #print('----------------')
    t_p1 += (t1-t0)
    t_p2 += (t2-t1)
    return 1086 - i_y, i_x


def find_index(lat,lon): # x = lon, y = lat
    global t_p1
    global t_p2
    global t_p3
    t0 = time.perf_counter()
    dist_lon = lon - gridLon
    dist_lat = lat - gridLat
    t1 = time.perf_counter()
    distance = abs(dist_lon) + abs(dist_lat)
    #distance = dist_lon**2+ dist_lat**2
    #distance = dist_lon*dist_lon + dist_lat*dist_lat
    t2 = time.perf_counter()
    i_x,i_y = np.unravel_index(np.nanargmin(distance),(1086,1086))
    t3 = time.perf_counter()
    #[lat_index1, lon_index1] = numpy.where(distance1 == numpy.nanmin(distance1))
    #print(t1-t0,t2-t1,t3-t2)
    t_p1 += (t1-t0)
    t_p2 += (t2-t1)
    t_p3 += (t3-t2)
    return i_x,i_y

def draw_flashes(map,finder,r=None):
    for i in range(GLM.dimensions['number_of_flashes'].size):
        #print('x=',GLM.variables['flash_lon'][i],'y=',GLM.variables['flash_lat'][i])
        lon = GLM.variables['flash_lon'][i]
        lat = GLM.variables['flash_lat'][i]
        i_x, i_y = finder(lat, lon)
        energy = GLM.variables['flash_energy'][i]/GLM.variables['flash_energy'].scale_factor
        area = GLM.variables['flash_area'][i]
        if r !=None:
            val = energy
        else:
            r, val = calculate_circle_values(energy, area)
        #print(r, val)
        draw_circle(map,i_x,i_y,r,val)

def draw_groups(map,finder,r=None):
    for i in range(GLM.dimensions['number_of_groups'].size):
        #print('x=',GLM.variables['flash_lon'][i],'y=',GLM.variables['flash_lat'][i])
        lon = GLM.variables['group_lon'][i]
        lat = GLM.variables['group_lat'][i]
        i_x, i_y = finder(lat,lon)
        energy = GLM.variables['group_energy'][i]/GLM.variables['group_energy'].scale_factor
        area = GLM.variables['group_area'][i]
        if r !=None:
            val = energy
        else:
            r, val = calculate_circle_values(energy, area)
        #print(r, val)
        draw_circle(map,i_x,i_y,r,val)

def draw_events(map, finder):
    for i in range(GLM.dimensions['number_of_events'].size):
        #print('x=',GLM.variables['flash_lon'][i],'y=',GLM.variables['flash_lat'][i])
        lon = GLM.variables['event_lon'][i]
        lat = GLM.variables['event_lat'][i]
        i_x, i_y = finder(lat,lon)
        energy = GLM.variables['event_energy'][i]/GLM.variables['event_energy'].scale_factor
        #area = GLM.variables['group_area'][i]
        #r, val = calculate_circle_values(energy, area)
        r = 1
        val = energy
        #print(r, val)
        draw_circle(map,i_x,i_y,r,val)

def calculate_circle_values(energy, area):
    # should consider 10*10 km per pixel and non uniform lat-lon meshgrid
    r = np.sqrt(area/math.pi/(10*2))
    r = np.sqrt(area/math.pi)
    val = energy/(area/(10*10))
    return r, val

def draw_circle(map, i_x,i_y,r,val):
    global t_p4
    t0 = time.perf_counter()
    margin = int(np.round(r))
    for i in range(-margin,margin+1):
        for j in range(-margin,margin+1):
            if i**2 + j**2 <= r**2:
                try:
                    map[i_x+i,i_y+j] += val
                except:
                    continue
    t1 = time.perf_counter()
    t_p4 += (t1-t0)
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


# %%
t_p1 = 0
t_p2 = 0
t_p3 = 0
t_p4 = 0
map1 = np.zeros((1086,1086))

t_start = time.perf_counter()
draw_flashes(map1,find_index_fast)
#draw_groups(map1,find_index_fast)
#draw_events(map1,find_index_fast)
t_end = time.perf_counter()

plt.figure(figsize=(20,15))
plt.imshow(map1)
plt.savefig('flashes.png')
plt.figure()

print(t_p1,t_p2,t_p3, t_p4,'total =',t_p1+t_p2+t_p3+ t_p4)
print('total=',t_end-t_start)
map1
        # %%
t_p1 = 0
t_p2 = 0
t_p3 = 0
t_p4 = 0
map2 = np.zeros((1086,1086))

t_start = time.perf_counter()
#draw_flashes(map2,find_index_fast)
draw_groups(map2,find_index_fast)
#draw_events(map1,find_index_fast)
t_end = time.perf_counter()

plt.figure(figsize=(20,15))
plt.imshow(map2)
plt.savefig('groups.png')
plt.figure()

print(t_p1,t_p2,t_p3, t_p4,'total =',t_p1+t_p2+t_p3+ t_p4)
print('total=',t_end-t_start)

# %%
t_p1 = 0
t_p2 = 0
t_p3 = 0
t_p4 = 0
map3 = np.zeros((1086,1086))

t_start = time.perf_counter()
#draw_flashes(map1,find_index_fast)
#draw_groups(map2,find_index_fast)
draw_events(map3,find_index_fast)
t_end = time.perf_counter()

plt.figure(figsize=(20,15))
plt.imshow(map3)
plt.savefig('events.png')
plt.figure()

print(t_p1,t_p2,t_p3, t_p4,'total =',t_p1+t_p2+t_p3+ t_p4)
print('total=',t_end-t_start)

# %%
t_p1 = 0
t_p2 = 0
t_p3 = 0
t_p4 = 0
map4 = np.zeros((1086,1086))

t_start = time.perf_counter()
draw_flashes(map4,find_index_fast)
draw_groups(map4,find_index_fast)
draw_events(map4,find_index_fast)
t_end = time.perf_counter()

plt.figure(figsize=(20,15))
plt.imshow(map4)
plt.savefig('total.png')
plt.figure()

print(t_p1,t_p2,t_p3, t_p4,'total =',t_p1+t_p2+t_p3+ t_p4)
print('total=',t_end-t_start)


# %%
t_p1 = 0
t_p2 = 0
t_p3 = 0
t_p4 = 0
map5 = np.zeros((1086,1086))

t_start = time.perf_counter()
draw_flashes(map5,find_index)
#draw_groups(map,find_index)
#draw_events(map,find_index)
t_end = time.perf_counter()

plt.figure(figsize=(20,15))
plt.imshow(map5)
plt.savefig('flashes_slow.png')
plt.figure()

print(t_p1,t_p2,t_p3, t_p4,'total =',t_p1+t_p2+t_p3+ t_p4)
print('total=',t_end-t_start)

# %%
t_p1 = 0
t_p2 = 0
t_p3 = 0
t_p4 = 0
map6 = np.zeros((1086,1086))

t_start = time.perf_counter()
#draw_flashes(map,find_index)
draw_groups(map6,find_index)
#draw_events(map,find_index)
t_end = time.perf_counter()

plt.figure(figsize=(20,15))
plt.imshow(map6)
plt.savefig('groups_slow.png')
plt.figure()

print(t_p1,t_p2,t_p3, t_p4,'total =',t_p1+t_p2+t_p3+ t_p4)
print('total=',t_end-t_start)


# %%
t_p1 = 0
t_p2 = 0
t_p3 = 0
t_p4 = 0
map7 = np.zeros((1086,1086))

t_start = time.perf_counter()
#draw_flashes(map,find_index)
#draw_groups(map,find_index)
draw_events(map7,find_index)
t_end = time.perf_counter()

plt.figure(figsize=(20,15))
plt.imshow(map7)
plt.savefig('events_slow.png')
plt.figure()

print(t_p1,t_p2,t_p3, t_p4,'total =',t_p1+t_p2+t_p3+ t_p4)
print('total=',t_end-t_start)

# %%
t_p1 = 0
t_p2 = 0
t_p3 = 0
t_p4 = 0
map8 = np.zeros((1086,1086))

t_start = time.perf_counter()
draw_flashes(map8,find_index)
draw_groups(map8,find_index)
draw_events(map8,find_index)
t_end = time.perf_counter()

plt.figure(figsize=(20,15))
plt.imshow(map8)
plt.savefig('total_slow.png')
plt.figure()

print(t_p1,t_p2,t_p3, t_p4,'total =',t_p1+t_p2+t_p3+ t_p4)
print('total=',t_end-t_start)

















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


# %%
GLM.dimensions['number_of_events'].size
import os
os.listdir('/Volumes/LaCie/GLM_2018/046' )
for i in range(46,47):
    for j in range(24):
        for name in os.listdir(f'/Volumes/LaCie/GLM_2018/{i:03d}/{j:02d}'):
            #print(name)
            temp = nc.Dataset(f'/Volumes/LaCie/GLM_2018/{i:03d}/{j:02d}/'+name,'r')
            print(i,j,temp.dimensions['number_of_events'].size)
GLM.variables['flash_area']
