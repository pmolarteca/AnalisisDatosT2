# -*- coding: utf-8 -*-
"""
Created on Tue Sep 05 08:26:51 2017

@author: palom
"""


from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
import datetime

from scipy import stats
import scipy.stats
from statsmodels import robust




Pathout='C:/Users/palom/Documents/AnalisisDatos/Tarea02/'
Data=Dataset('precip.mon.mean.nc','r')
print Data.variables

print Data.variables.keys()
print Data.variables['precip']
print Data.variables['time']

precip = np.array (Data.variables['precip'][:])
precip[precip==-9.96921e+36]=np.nan
lat = np.array (Data.variables['lat'][:])
lon = np.array (Data.variables['lon'][:])
time = np.array (Data.variables['time'][:])
lat_bnds = np.array (Data.variables['lat_bnds'][:])
lon_bnds= np.array (Data.variables['lon_bnds'][:])
time_bnds = np.array (Data.variables['time_bnds'][:])

time = time.astype(float)


fechas = []
for i in range(len(time)):
#print i
#print time[i]
    fechas.append(datetime.datetime(1800,01,01)+datetime.timedelta(days = time[i]))

fechas = np.array(fechas)
print fechas

#-----------Estadisticos Anuales---------------------------------

Mapa_Media_Anual = np.zeros([len(lat),len(lon)]) * np.NaN
Mapa_Mediana_Anual = np.zeros([len(lat),len(lon)]) * np.NaN
Mapa_Desviacion_Anual= np.zeros([len(lat),len(lon)]) * np.NaN
Mapa_IQR_Anual= np.zeros([len(lat),len(lon)]) * np.NaN
Mapa_Asimetria_Anual= np.zeros([len(lat),len(lon)]) * np.NaN
Mapa_YK_Anual= np.zeros([len(lat),len(lon)]) * np.NaN


iqr = np.subtract(*np.percentile(precip, [75, 25]))

per25=np.percentile(precip,25)
per75=np.percentile(precip,75)
IQR=per25-per75



for i in range(len(lat)):
    for j in range(len(lon)):
        Mapa_Media_Anual[i,j] = np.mean(precip[:,i,j])
        Mapa_Mediana_Anual[i,j] = np.median(precip[:,i,j])
        Mapa_Desviacion_Anual[i,j] = np.std(precip[:,i,j])
        Mapa_IQR_Anual[i,j] = np.subtract(*np.percentile(precip[:,i,j], [75, 25]))
        Mapa_Asimetria_Anual[i,j] = stats.skew(precip[:,i,j])
        Mapa_YK_Anual[i,j] = ((np.percentile(precip[:,i,j],25))-2*\
        (np.median(precip[:,i,j])) + (np.percentile(precip[:,i,j],75))) /\
        ((np.percentile(precip[:,i,j],75)) -  (np.percentile(precip[:,i,j],25)))



#-----------------------plotear Mapa media anual--------------------
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
m = Basemap(llcrnrlat=np.min(lat),urcrnrlat=np.max(lat), \
llcrnrlon=np.min(0),urcrnrlon=np.max(360),\
rsphere=6371200.,resolution='l',area_thresh=10000)
ny = lat.shape[0]; nx = lon.shape[0]
lons, lats = m.makegrid(nx, ny)
x,y = m(lons, lats)
#cs = m.contourf(x,y,np.flipud(Mapa))
cs = m.contourf(x,y,np.roll(Mapa_Media_Anual, len(lon)/2))
#m.colorbar(location='bottom',pad="10%")
m.colorbar()
m.drawparallels(np.arange(-90.,90,30.), labels=[1,0,0,0], size=11,\
linewidth=0.1)
m.drawmeridians(np.arange(0, 360, 30.),labels=[0,1,0,1], size=11, \
linewidth=0.1)
m.drawcoastlines()
m.drawmapboundary()
plt.savefig(Pathout+'Mapa Media Anual'+'.png')   
       

#-----------------------plotear Mapa mediana anual--------------------
        
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
m = Basemap(llcrnrlat=np.min(lat),urcrnrlat=np.max(lat), \
llcrnrlon=np.min(0),urcrnrlon=np.max(360),\
rsphere=6371200.,resolution='l',area_thresh=10000)
ny = lat.shape[0]; nx = lon.shape[0]
lons, lats = m.makegrid(nx, ny)
x,y = m(lons, lats)
#cs = m.contourf(x,y,np.flipud(Mapa))
cs = m.contourf(x,y,np.roll(Mapa_Mediana_Anual, len(lon)/2))
#m.colorbar(location='bottom',pad="10%")
m.colorbar()
m.drawparallels(np.arange(-90.,90,30.), labels=[1,0,0,0], size=11,\
linewidth=0.1)
m.drawmeridians(np.arange(0, 360, 30.),labels=[0,1,0,1], size=11, \
linewidth=0.1)
m.drawcoastlines()
m.drawmapboundary()
plt.savefig(Pathout+'Mapa Mediana Anual'+'.png')


#-----------------------plotear Mapa desviacion anual--------------------


fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
m = Basemap(llcrnrlat=np.min(lat),urcrnrlat=np.max(lat), \
llcrnrlon=np.min(0),urcrnrlon=np.max(360),\
rsphere=6371200.,resolution='l',area_thresh=10000)
ny = lat.shape[0]; nx = lon.shape[0]
lons, lats = m.makegrid(nx, ny)
x,y = m(lons, lats)
#cs = m.contourf(x,y,np.flipud(Mapa))
cs = m.contourf(x,y,np.roll(Mapa_Desviacion_Anual, len(lon)/2))
#m.colorbar(location='bottom',pad="10%")
m.colorbar()
m.drawparallels(np.arange(-90.,90,30.), labels=[1,0,0,0], size=11,\
linewidth=0.1)
m.drawmeridians(np.arange(0, 360, 30.),labels=[0,1,0,1], size=11, \
linewidth=0.1)
m.drawcoastlines()
m.drawmapboundary()
plt.savefig(Pathout+u'Mapa Desviación Estandar Anual'+'.png')

#-----------------------plotear Mapa Asimetria anual-------------------------
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
m = Basemap(llcrnrlat=np.min(lat),urcrnrlat=np.max(lat), \
llcrnrlon=np.min(0),urcrnrlon=np.max(360),\
rsphere=6371200.,resolution='l',area_thresh=10000)
ny = lat.shape[0]; nx = lon.shape[0]
lons, lats = m.makegrid(nx, ny)
x,y = m(lons, lats)
#cs = m.contourf(x,y,np.flipud(Mapa))
cs = m.contourf(x,y,np.roll(Mapa_Asimetria_Anual, len(lon)/2))
#m.colorbar(location='bottom',pad="10%")
m.colorbar()
m.drawparallels(np.arange(-90.,90,30.), labels=[1,0,0,0], size=11,\
linewidth=0.1)
m.drawmeridians(np.arange(0, 360, 30.),labels=[0,1,0,1], size=11, \
linewidth=0.1)
m.drawcoastlines()
m.drawmapboundary()
plt.savefig(Pathout+'Mapa coeficiente de asimetria anual'+'.png')



#----------------------------------------------------
#----------------------------------------------------


#medias mensuales

Meses = np.array([fechas[i].month for i in range(len(fechas))])

Mapa_Media_mensual= np.zeros([12,len(lat),len(lon)]) * np.NaN
Mapa_Mediana_mensual = np.zeros([12,len(lat),len(lon)]) * np.NaN
Mapa_Desviacion_mensual= np.zeros([12,len(lat),len(lon)]) * np.NaN
Mapa_IQR_mensual= np.zeros([12,len(lat),len(lon)]) * np.NaN
Mapa_Asimetria_mensual= np.zeros([12,len(lat),len(lon)]) * np.NaN
Mapa_YK_mensual= np.zeros([12,len(lat),len(lon)]) * np.NaN

 
for k in range(1,13): 
    tmp= (np.where(Meses == k)) [0]
    Temp_meses = precip[tmp,:,:]
    
    for i in range(len(lat)):
        for j in range(len(lon)):
    
            Mapa_Media_mensual[k - 1,i,j] = np.mean(Temp_meses[:,i,j])
            Mapa_Mediana_mensual[k-1,i,j] = np.median(Temp_meses[:,i,j])
            Mapa_Desviacion_mensual[k-1,i,j] = np.std(Temp_meses[:,i,j])
            Mapa_IQR_mensual[k-1, i,j] = np.subtract(*np.percentile(Temp_meses[:,i,j], [75, 25]))
            Mapa_Asimetria_mensual[k-1,i,j] = stats.skew(Temp_meses[:,i,j])
            Mapa_YK_mensual[k-1, i,j] = ((np.percentile(precip[:,i,j],25))-2*\
            (np.median(precip[:,i,j])) + (np.percentile(precip[:,i,j],75))) / \
            ((np.percentile(precip[:,i,j],75)) -  (np.percentile(precip[:,i,j],25)))



fig = plt.figure(figsize=(10,8))

for i in range(12):
    
    ax = fig.add_subplot(4,3,i+1)
    ax.set_title(u'Mes = ' +str(i + 1))
    # Basemap es el paquete que dibuja las líneas del mapa
    m = Basemap(llcrnrlat=np.min(lat),urcrnrlat=np.max(lat), \
    llcrnrlon=np.min(0),urcrnrlon=np.max(360),\
    rsphere=6371200.,resolution='l',area_thresh=10000, lon_0 =179)
    ny = lat.shape[0]; nx = lon.shape[0]
    lons, lats = m.makegrid(nx, ny)
    x,y = m(lons, lats)
    #cs = m.contourf(x,y,np.flipud(Mapa))
    cs = m.contourf(x,y,np.roll(Mapa_Media_mensual[i,:,:],len(lon)/2)) 
    #m.colorbar(location='bottom',pad="10%")
    m.colorbar()
    m.drawparallels(np.arange(-90.,90,30.), labels=[1,0,0,0], size=11,\
    linewidth=0.1)
    m.drawmeridians(np.arange(0, 360, 30.),labels=[0,1,0,1], size=11, \
    linewidth=0.1)
    m.drawcoastlines()
    m.drawmapboundary()

plt.savefig('test.png')



#Colombia ------------------------------
#---------------------------------------


Colombia_Lat = np.where((lat > 0) & (lat < 11))[0]
Colombia_Lon = np.where((lon > 283) & (lon < 290))[0]


Media_Colombia=[]
Mediana_Colombia=[]
Desviacion_Colombia=[]
Curtosis_Colombia=[]
Asimetria_Colombia=[]
MAD_Colombia=[]
TriMd=[]
YK_Colombia=[]

for i in range(len(time)):
    
    Mapa_Colombia= precip[i,Colombia_Lat,:]
    Mapa_Colombia = Mapa_Colombia[:,Colombia_Lon]
    Mapa_NoNaN_Colombia = Mapa_Colombia[np.isfinite(Mapa_Colombia)]
    Media_Colombia.append(np.mean(Mapa_NoNaN_Colombia))
    Mediana_Colombia.append(np.median(Mapa_NoNaN_Colombia))
    Desviacion_Colombia.append(np.std(Mapa_NoNaN_Colombia))
    Curtosis_Colombia.append(scipy.stats.kurtosis(Mapa_NoNaN_Colombia))
    Asimetria_Colombia.append(stats.skew(Mapa_NoNaN_Colombia))
    MAD_Colombia.append(robust.mad(Mapa_NoNaN_Colombia))
    #TriMd_Colombia.append(np.median(Mapa_NoNaN_Colombia))
    #YK_Colombia.append(np.median(Mapa_NoNaN_Colombia))
    
    
Media_Colombia=np.array(Media_Colombia)
Mediana_Colombia=np.array(Mediana_Colombia)
Desviacion_Colombia=np.array(Desviacion_Colombia)
Curtosis_Colombia=np.array(Curtosis_Colombia)
Asimetria_Colombia=np.array(Asimetria_Colombia)
MAD_Colombia=np.array(MAD_Colombia)

    
Meses = np.array([fechas[i].month for i in range(len(fechas))])

Colombia_Media_mensual= np.zeros([12]) * np.NaN
Colombia_Mediana_mensual =  np.zeros([12]) * np.NaN
Colombia_Desviacion_mensual=  np.zeros([12]) * np.NaN
Colombia_IQR_mensual= np.zeros([12]) * np.NaN
Colombia_Asimetria_mensual=  np.zeros([12]) * np.NaN
Colombia_MAD_mensual= np.zeros([12]) *np.NaN
Colombia_YK_mensual=  np.zeros([12]) * np.NaN


for k in range(1,13):
    tmpp = np.where(Meses == k)[0]
    
    TempM = Media_Colombia[tmpp]
    TempMd= Mediana_Colombia[tmpp]
    TempD=  Desviacion_Colombia[tmpp]
    TempC=  Curtosis_Colombia[tmpp]
    TempA=  Asimetria_Colombia[tmpp]    
    TempMAD= MAD_Colombia [tmpp]
    #TempYK= YK_Colombia [tmpp] 
    
    
    
    Colombia_Media_mensual[k-1]= np.mean(TempM)
    Colombia_Mediana_mensual[k-1]= np.mean(TempMd)
    Colombia_Desviacion_mensual[k-1]= np.mean(TempD)
  # Colombia_Curtosis_mensual[k-1]= np.mean(TempC)
  #  Colombia_IQR_mensual[k]= np.mean(TempC)
    Colombia_Asimetria_mensual[k-1]= np.mean(TempA)
    Colombia_MAD_mensual[k-1]= np.mean (TempMAD)
    #Colombia_YK_mensual[k]= np.mean(TempYK)

    
    
    
#gráficas ciclo anual estadisticos Colombia
    

fig = plt.figure(figsize=(10,8))
plt.plot(Colombia_Media_mensual, color= 'g', label= 'Media')
plt.plot(Colombia_Mediana_mensual, color= 'm', label='Mediana' )
plt.legend()

plt.savefig('test2.png')


#pacifico--------------------------------------------------
#-----------------------------------------------------------



Pacifico_Lat = np.where((lat > -4) & (lat < 4))[0]
Pacifico_Lon = np.where((lon > 210) & (lon < 230))[0]

Media_Pacifico=[]
Mediana_Pacifico=[]
Desviacion_Pacifico=[]
Curtosis_Pacifico=[]
Asimetria_Pacifico=[]
MAD_Pacifico=[]
TriMd_Pacifico=[]
YK_Pacifico=[]



for i in range(len(time)):

    Mapa_Pacifico= precip[i,Pacifico_Lat,:]
    Mapa_Pacifico = Mapa_Pacifico[:,Pacifico_Lon]
    Mapa_NoNaN_Pacifico = Mapa_Pacifico[np.isfinite(Mapa_Pacifico)]
    Media_Pacifico.append(np.mean(Mapa_NoNaN_Pacifico))
    Mediana_Pacifico.append(np.median(Mapa_NoNaN_Pacifico))
    Desviacion_Pacifico.append(np.std(Mapa_NoNaN_Pacifico))
    Curtosis_Pacifico.append(scipy.stats.kurtosis(Mapa_NoNaN_Pacifico))
    Asimetria_Pacifico.append(stats.skew(Mapa_NoNaN_Pacifico))
    MAD_Pacifico.append(robust.mad(Mapa_NoNaN_Pacifico))
    #TriMd_Colombia.append(np.median(Mapa_NoNaN_Colombia))
    #YK_Colombia.append(np.median(Mapa_NoNaN_Colombia))




Media_Pacifico=np.array(Media_Pacifico)
Mediana_Pacifico=np.array(Mediana_Pacifico)
Desviacion_Pacifico=np.array(Desviacion_Pacifico)
Curtosis_Pacifico=np.array(Curtosis_Pacifico)
Asimetria_Pacifico=np.array(Asimetria_Pacifico)
MAD_Pacifico=np.array(MAD_Pacifico)

    

Pacifico_Media_mensual= np.zeros([12]) * np.NaN
Pacifico_Mediana_mensual =  np.zeros([12]) * np.NaN
Pacifico_Desviacion_mensual=  np.zeros([12]) * np.NaN
Pacifico_IQR_mensual= np.zeros([12]) * np.NaN
Pacifico_Asimetria_mensual=  np.zeros([12]) * np.NaN
Pacifico_MAD_mensual= np.zeros([12]) *np.NaN
Pacifico_YK_mensual=  np.zeros([12]) * np.NaN


for k in range(1,13):
    tmpp = np.where(Meses == k)[0]
    
    PTempM = Media_Pacifico[tmpp]
    PTempMd= Mediana_Pacifico[tmpp]
    PTempD=  Desviacion_Pacifico[tmpp]
    PTempC=  Curtosis_Pacifico[tmpp]
    PTempA=  Asimetria_Pacifico[tmpp]    
    PTempMAD= MAD_Pacifico [tmpp]
    #TempYK= YK_Pacifico [tmpp] 
    
    
    
    Pacifico_Media_mensual[k-1]= np.mean(PTempM)
    Pacifico_Mediana_mensual[k-1]= np.mean(PTempMd)
    Pacifico_Desviacion_mensual[k-1]= np.mean(PTempD)
  # Pacifico_Curtosis_mensual[k-1]= np.mean(PTempC)
  # Pacifico_IQR_mensual[k]= np.mean(PTempC)
    Pacifico_Asimetria_mensual[k-1]= np.mean(PTempA)
    Pacifico_MAD_mensual[k-1]= np.mean (PTempMAD)
   #Pacifico_YK_mensual[k]= np.mean(PTempYK)

   
#gráficas ciclo anual estadisticos Colombia
    

fig = plt.figure(figsize=(10,8))
plt.plot(Pacifico_Media_mensual, color= 'g', label= 'Media')
plt.plot(Pacifico_Mediana_mensual, color= 'm', label='Mediana' )
plt.legend()



#INDIA-----------------
#---------------------



India_Lon = np.where((lon > 70) & (lon < 85))[0]
India_Lat= np.where ((lat>6) & (lat< 20)) [0]


Media_India=[]
Mediana_India=[]
Desviacion_India=[]
Curtosis_India=[]
Asimetria_India=[]
MAD_India=[]
TriMd_India=[]
YK_India=[]



for i in range(len(time)):

    Mapa_India= precip[i,India_Lat,:]
    Mapa_India = Mapa_India[:,India_Lon]
    Mapa_NoNaN_India = Mapa_India[np.isfinite(Mapa_India)]
    Media_India.append(np.mean(Mapa_NoNaN_India))
    Mediana_India.append(np.median(Mapa_NoNaN_India))
    Desviacion_India.append(np.std(Mapa_NoNaN_India))
    Curtosis_India.append(scipy.stats.kurtosis(Mapa_NoNaN_India))
    Asimetria_India.append(stats.skew(Mapa_NoNaN_India))
    MAD_India.append(robust.mad(Mapa_NoNaN_India))
    #TriMd_Colombia.append(np.median(Mapa_NoNaN_Colombia))
    #YK_Colombia.append(np.median(Mapa_NoNaN_Colombia))




Media_India=np.array(Media_India)
Mediana_India=np.array(Mediana_India)
Desviacion_India=np.array(Desviacion_India)
Curtosis_India=np.array(Curtosis_India)
Asimetria_India=np.array(Asimetria_India)
MAD_India=np.array(MAD_India)

    

India_Media_mensual= np.zeros([12]) * np.NaN
India_Mediana_mensual =  np.zeros([12]) * np.NaN
India_Desviacion_mensual=  np.zeros([12]) * np.NaN
India_IQR_mensual= np.zeros([12]) * np.NaN
India_Asimetria_mensual=  np.zeros([12]) * np.NaN
India_MAD_mensual= np.zeros([12]) *np.NaN
India_YK_mensual=  np.zeros([12]) * np.NaN


for k in range(1,13):
    tmpp = np.where(Meses == k)[0]
    
    ITempM = Media_India[tmpp]
    ITempMd= Mediana_India[tmpp]
    ITempD=  Desviacion_India[tmpp]
    ITempC=  Curtosis_India[tmpp]
    ITempA=  Asimetria_India[tmpp]    
    ITempMAD= MAD_India [tmpp]
    #TempYK= YK_India[tmpp] 
    
    
    
    India_Media_mensual[k-1]= np.mean(ITempM)
    India_Mediana_mensual[k-1]= np.mean(ITempMd)
    India_Desviacion_mensual[k-1]= np.mean(ITempD)
  # India_Curtosis_mensual[k-1]= np.mean(ITempC)
  # India_IQR_mensual[k]= np.mean(ITempC)
    India_Asimetria_mensual[k-1]= np.mean(ITempA)
    India_MAD_mensual[k-1]= np.mean (ITempMAD)
   #India_YK_mensual[k]= np.mean(ITempYK)

   
#gráficas ciclo anual estadisticos India
    

fig = plt.figure(figsize=(10,8))
plt.plot(India_Media_mensual, color= 'g', label= 'Media')
plt.plot(India_Mediana_mensual, color= 'm', label='Mediana' )
plt.legend()


##Distribucion de probabilidades--------------------
#----------------------------------------------------



#histograma 

num_bins = 8
hist,bins= np.histogram(Mapa_NoNaN_Colombia, bins=num_bins)
hist =hist.astype(float)
pdf= hist/np.sum(hist)
print len(bins)
print len(pdf)
print pdf
print np.sum(pdf)*np.diff(bins)[0]
print bins
print np.diff(bins)

fig = plt.figure()
ax = fig.add_subplot(111)
temp=pdf/np.sum(pdf)
print temp
bincenters=(bins[1:]+bins[:-1])/2
plt.plot(bincenters,pdf,'o-',color='b',lw=2, markersize=10)
plt.ylim((0,0.8))


num_bins =8
hist,bins= np.histogram(Mapa_NoNaN_Pacifico, bins=num_bins)
hist =hist.astype(float)
pdf= hist/np.sum(hist)
print len(bins)
print len(pdf)
print pdf
print np.sum(pdf)*np.diff(bins)[0]
print bins
print np.diff(bins)

fig = plt.figure()
ax = fig.add_subplot(111)
temp=pdf/np.sum(pdf)
print temp
bincenters=(bins[1:]+bins[:-1])/2
plt.plot(bincenters,pdf,'o-',color='b',lw=2, markersize=10)
plt.ylim((0,0.8))



num_bins = 10
hist,bins= np.histogram(Mapa_NoNaN_India , bins=num_bins)
hist =hist.astype(float)
pdf= hist/np.sum(hist)
print len(bins)
print len(pdf)
print pdf
print np.sum(pdf)*np.diff(bins)[0]
print bins
print np.diff(bins)

fig = plt.figure()
ax = fig.add_subplot(111)
temp=pdf/np.sum(pdf)
print temp
bincenters=(bins[1:]+bins[:-1])/2
plt.plot(bincenters,pdf,'o-',color='b',lw=2, markersize=10)
plt.ylim((0,0.5))




###Cuartiles-------------

Q25_Colombia=np.percentile(Mapa_NoNaN_Colombia,25)
Q75_Colombia=np.percentile(Mapa_NoNaN_Colombia,75)
Q25_Pacifico=np.percentile(Mapa_NoNaN_Pacifico,25)
Q75_Pacifico=np.percentile(Mapa_NoNaN_Pacifico,75)
Q25_India=np.percentile(Mapa_NoNaN_India,25)
Q75_India=np.percentile(Mapa_NoNaN_India,75)



####Deciles------------------
#---------------------------
#Deciles
Deciles_Colombia=[]
Deciles_Pacifico=[]
Deciles_India=[]


for i in range(10,100,10):
    Deciles_Colombia.append(np.percentile(Mapa_NoNaN_Colombia,i))    
    Deciles_Pacifico.append(np.percentile(Mapa_NoNaN_Pacifico,i))   
    Deciles_India.append(np.percentile(Mapa_NoNaN_India,i))   

fig=plt.figure()
plt.plot( Deciles_Pacifico,Deciles_Colombia)


fig=plt.figure()
plt.plot(Deciles_Colombia,Deciles_India)
plt.plot()

fig=plt.figure()
plt.plot(Deciles_India,Deciles_Pacifico)



#Scatter Plot------------------------------------------------------------------------

fig = plt.figure(figsize = [20,10])
ax = fig.add_subplot(121)
ax.set_title(u'Scatterplot Colombia-Pacífico')
ax.plot(Media_Colombia, Media_Pacifico, 'o',color='darkmagenta')
ax.set_xlabel('Media Colombia', fontsize=13)
ax.set_ylabel(u'Media Pacífico', fontsize=13)
ax.axhline(0, color='k')
ax.axvline(0, color='k')
ax.text(-5, 2.5, u'Corr = '+str(scp.pearsonr(Media_Colombia, Media_Pacifico)[0])[:5],\
fontsize=13)
ax.set_ylim(-8,18)
ax.set_xlim(-8,14)


fig = plt.figure(figsize = [20,10])
ax = fig.add_subplot(121)
ax.set_title(u'Scatterplot Colombia-India')
ax.plot(Media_Colombia, Media_India, 'o',color='b')
ax.set_xlabel('Media Colombia', fontsize=13)
ax.set_ylabel(u'Media India', fontsize=13)
ax.axhline(0, color='k')
ax.axvline(0, color='k')
ax.text(-5, 2.5, u'Corr = '+str(scp.pearsonr(Media_Colombia, Media_India)[0])[:5],\
fontsize=13)
ax.set_ylim(-8,18)
ax.set_xlim(-8,14)


fig = plt.figure(figsize = [20,10])
ax = fig.add_subplot(121)
ax.set_title(u'Scatterplot India-Pacífico')
ax.plot(Media_India, Media_Pacifico, 'o',color='c')
ax.set_xlabel('Media India', fontsize=13)
ax.set_ylabel(u'Media Pacífico', fontsize=13)
ax.axhline(0, color='k')
ax.axvline(0, color='k')
ax.text(-5, 2.5, u'Corr = '+str(scp.pearsonr(Media_India, Media_Pacifico)[0])[:5],\
fontsize=13)
ax.set_ylim(-8,18)
ax.set_xlim(-8,14)


#Histograma bivariado-------------
fig=plt.figure()
ax=plt.axes()
h=ax.hist2d(Media_Colombia,Media_India, bins=(10,10))
cb=fig.colorbar(h[3])
plt.xlabel('Media Colombia')
plt.ylabel('Media India')
plt.title('Histograma Bivariado')


fig=plt.figure()
ax=plt.axes()
plt.hist2d(Media_Colombia,Media_Pacifico, bins=(10,10))
cb=fig.colorbar(h[3])
plt.xlabel('Media Colombia')
plt.ylabel(u'Media Pacífico')
plt.title('Histograma Bivariado')


fig=plt.figure()
ax=plt.axes()
h=ax.hist2d(Media_India,Media_Pacifico, bins=(10,10))
cb=fig.colorbar(h[3])
cb.ax.set_label('counts in bin')
plt.xlabel('Media India')
plt.ylabel(u'Media Pacífico')
plt.title('Histograma Bivariado')

#---------------Correlacion

Meses = np.array([fechas[i].month for i in range(len(fechas))])
Mapa_Probabilidad = np.zeros([12,len(lat),len(lon)]) * np.NaN

for k in range(1,13):
    tmp = np.where(Meses == k)[0]
    Temp_mes = precip[tmp,:,:]
    Temp_Colombia = Media_Colombia[tmp]
    for i in range(len(lat)):
        for j in range(len(lon)):
            Mapa_Probabilidad[k-1,i,j] = len(np.where((Temp_mes[:,i,j])\
                                         >Temp_Colombia) [0])/np.float(len(Temp_mes))





fig = plt.figure(figsize=(35,18))

for i in range(12):
    
    ax = fig.add_subplot(4,3,i+1)
    ax.set_title(u'Mes = ' +str(i + 1))
    # Basemap es el paquete que dibuja las líneas del mapa
    m = Basemap(llcrnrlat=np.min(lat),urcrnrlat=np.max(lat), \
    llcrnrlon=np.min(0),urcrnrlon=np.max(360),\
    rsphere=6371200.,resolution='l',area_thresh=10000, lon_0 =179)
    ny = lat.shape[0]; nx = lon.shape[0]
    lons, lats = m.makegrid(nx, ny)
    x,y = m(lons, lats)
    #cs = m.contourf(x,y,np.flipud(Mapa))
    cs = m.contourf(x,y,np.roll(Mapa_Probabilidad[i,:,:],len(lon)/2)) 
    #m.colorbar(location='bottom',pad="10%")
    m.colorbar()
    m.drawparallels(np.arange(-90.,90,30.), labels=[1,0,0,0], size=11,\
    linewidth=0.1)
    m.drawmeridians(np.arange(0, 360, 30.),labels=[0,1,0,1], size=11, \
    linewidth=0.1)
    m.drawcoastlines()
    m.drawmapboundary()

plt.savefig('test3.png')
