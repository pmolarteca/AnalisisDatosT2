# -*- coding: utf-8 -*-
"""
Created on Fri Sep 01 17:28:18 2017

@author: palom
"""

from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
import datetime
import pandas as pd
import scipy.stats

path='./Lottery_Take_5_Winning_Numbers.csv'
Data = np.genfromtxt(path,delimiter=',',dtype=str,skip_header=1)
Pathout='C:/Users/palom/Documents/AnalisisDatos/Tarea02/'


from datetime import datetime 

Fechas = []
Numeros = np.zeros([len(Data), 5])
for i in range(len(Data)):
    Fechas.append(datetime.strptime(Data[i,0], '%m/%d/%Y'))
    Numeros[i,:] = np.array(Data[i,1].split(' ')).astype(np.int)
Fechas = np.array(Fechas)


Loteria = pd.DataFrame(Numeros, index=Fechas)
print Loteria

#probabilidades ultimo digito-----------------------------------------

plt.plot(Fechas,Loteria[4])

binnum=8

hist,bins= np.histogram(Loteria[4], bins=binnum)
hist =hist.astype(float)
pdf= hist/np.sum(hist)
print len(bins)
print len(pdf)
print pdf
print bins


fig = plt.figure(figsize=[9,6])
ax = fig.add_subplot(111)
temp=pdf/np.sum(pdf)
print temp
bincenters=(bins[1:]+bins[:-1])/2
plt.plot(bincenters,pdf,'o-',color='b',lw=2, markersize=10)
plt.ylabel(u'Probabilidad', fontsize=15)
plt.xlabel(u'Número de Balota', fontsize=15)
plt.title(u'Distribución de Probabilidades', fontsize=20)
plt.savefig('Ultimodigito')
#---------------------------------------------------------------------


#probabilidad de que todos los numeros sean pares dado que es un dia par



todospares= (np.where((Loteria[0]%2==0) & (Loteria [1]%2==0) & (Loteria [2] %2==0) \
& (Loteria [3] %2==0) & (Loteria [4] %2==0) & (Loteria.index.day %2==0)) [0])


Prob_pares= len(todospares)/np.float(len(Loteria[Loteria.index.day %2==0]))

#probabilidad de que los dias pares caigan todos los numeros pares

Prob_numPares=[]

for i in range(2,32):
    Prob_numPares.append(len(np.where((Loteria[0]%2==0) & (Loteria [1]%2==0) & (Loteria [2] %2==0) \
& (Loteria [3] %2==0) & (Loteria [4] %2==0) & (Loteria.index.day%2==0)\
& (Loteria.index.day==i) ) [0]) /np.float(len(Loteria[Loteria.index.day %2==0])))
    

   
    
plt.figure(figsize=[14,6])
plt.rcParams.update({'font.size':14})
plt.plot(range(2,32), Prob_numPares, color='deeppink')
plt.title(u'Probabilidad Condicional A', fontsize=20)
plt.xlabel(u'Día',fontsize=15)
plt.ylabel(u'Probabilidad', fontsize=15)
#plt.ylabel('',fontsize=18)
axes = plt.gca()
axes.set_xlim([2,31])
axes.set_ylim([0.0,0.008])
plt.xticks(np.arange(2, 32, 1.0))
plt.savefig(Pathout+'PCA'+'.png')







#----------------------------------------------------------------------------------


#probabilidad de que caiga 1 dado que es el primer lunes del mes

PrimerLunes= (np.where((Loteria [0] ==1)  & \
 (Loteria.index.day <8) & (Loteria.index.weekday == 0)  ) [0])
 

Prob_PrimerLunes= np.float(len(PrimerLunes))\
/len(Loteria[(Loteria.index.weekday == 0) & (Loteria.index.day <8)])
 

Prob_PL=[]

for i in range(1,13):
    Prob_PL.append(len(np.where((Loteria [0] ==1)  & \
 (Loteria.index.day <8) & (Loteria.index.weekday == 0) & (Loteria.index.month==i)) [0])\
 /np.float(len(Loteria[(Loteria.index.weekday == 0) & (Loteria.index.day <8)]))) 
    

plt.figure(figsize=[10,5])
plt.rcParams.update({'font.size':14})
plt.plot(range(1,13),Prob_PL, color='deeppink')
plt.title(u'Probabilidad Condicional B',fontsize=20)
plt.ylabel(u'Probabilidad', fontsize=15)
#plt.xlabel(u'Mes',fontsize=20)
axes = plt.gca()
axes.set_xlim([1,12])
axes.set_ylim([0.0,0.08])
ax=plt.gca()#grab the current axis
ax.set_xticks([1,2, 3, 4, 5, 6, 7,8, 9, 10 ,11,12]) #choose which x locations to have ticks
ax.set_xticklabels(['Ene','Feb','Mar','Abr','May','Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic' ]) 
plt.savefig(Pathout+'PCB'+'.png')

aa=(Loteria [0] ==1)  &  (Loteria.index.day <8) & (Loteria.index.weekday == 0)
print Loteria[aa]

#-----------------------------------------------------------------------------


#Probabilidad de que la suma de los primeros cuatro digitos
# sea menor que el ultimo digito dado que es un año del siglo XX


Psuma4= Loteria[0] + Loteria[1] + Loteria[2] + Loteria[3] 

PsumaXX=(np.where( (Loteria[4] > Psuma4) & (Loteria.index.year < 2000)) [0])

Prob_PsumaXX= len(PsumaXX)/np.float(len(Loteria[Loteria.index.year <2000]))
 


ProbXX=[]

for i in range(1992,2000):
    ProbXX.append(len(np.where((Loteria[4] > Psuma4) & \
    (Loteria.index.year==i)) [0]) /np.float(len(Loteria[Loteria.index.year <2000]))) 
    
    
        
plt.figure(figsize=[10,5])
plt.rcParams.update({'font.size':14})
plt.plot(range(1,9),ProbXX, color='deeppink')
plt.title(u'Probabilidad Condicional C', fontsize=20)
plt.xlabel(u'Año',fontsize=15)
plt.ylabel(u'Probabilidad', fontsize=15)
ax=plt.gca()#grab the current axis
ax.set_xlim([1,8])
ax.set_ylim([0.0,0.015])
ax.set_xticks([1,2, 3, 4, 5, 6, 7,8]) #choose which x locations to have ticks
ax.set_xticklabels(['1992','1993','1994','1995','1996','1997', '1998', '1999']) 
plt.savefig(Pathout+'PCC'+'.png')







#------------------------------------------------------------------



#clase----------------------------------------------
