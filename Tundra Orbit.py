#!/usr/bin/env python
# coding: utf-8

# ### Pol Riera González     UGR     Máster en Electrónica Industrial  03/2021  v5 endpoint

# ## Propagación de una órbita TUNDRA y obtener trayectoria del punto subsatélite.

# ### Librerias

# In[40]:


import numpy as np
import math
import matplotlib.pyplot as plt

from PIL import Image
from scipy import optimize

get_ipython().run_line_magic('matplotlib', 'notebook')


# ### Datos

# In[41]:


a=42164  # Semieje mayor de la elipse (km)
e=0.25   # Excentricidad
Ω=0 * np.pi / 180      # Ascension recta del nodo ascendente (rad)
i=63.4 * np.pi / 180   # Inclinacion (rad)
ω=270 * np.pi / 180    # Argumento del perigeo (rad)

Le=38    # Latitud Norte estación (º)
le=0.7   # Longitud Oeste estación

# fechas perigeo
h = 0
min = 0


#fechas estacion (fecha de los calculos)
dia = 18
mes = 10
año = 1993 

re = 6371  #radio de la Tierra


# #### Periodo de la orbita

# In[42]:


μE = 3.98601352 * 10**5
T = 2 * np.pi * np.sqrt(a**3/μE)
print('periodo de la orbita', T, '(s)')


# #### Calculo dia Juliano

# In[43]:


A = año
DTA = 291   # dia 18 de octubre
NAB1900 = 23   # numero de años bisiestos
TU = 0
min_sidereo = 23 * 60 + 56 + 4.89 / 60  # minutos de un dia sidereo
s_sidereo = min_sidereo * 60
h_sidereo = (4.89/60 + 56) / 60 + 23

JD = 2415020 + 365 * (A-1900) + DTA + NAB1900 + TU/24 - 0.5  # dia Juliano
print('Dia Juliano: ', JD, 'dias')   # dias desde 4713 aC


# #### Velocidad angular

# In[78]:


μ = 2 * np.pi / T    # velocidad angular media
print('Velocidad angular media, μ: ', μ, '(rad/s)')

tp = JD * s_sidereo
print ('Tiempo de paso por el perigeo, tp: ', tp, '(s) \n' )

t = np.linspace(tp, tp + T, 100)
print ('t: ', t, '\n')

M = μ * (t - tp) 
print ('Anomalia media, M: \n', M, '\n')

t_dias = np.linspace(JD, JD + T/60/60/h_sidereo, 100, endpoint=False)

Mnueva = μ * (t_dias - JD) *60*60*h_sidereo
print ('Anomalia media nueva, Mnueva: \n', Mnueva, '\n')


# #### Matriz de transformacion de cooordenadas orbitales a inerciales

# In[79]:


Ti = np.zeros((3,3))
Ti[0,0] = np.cos(Ω) * np.cos(ω) - np.sin(Ω) * np.cos(i) * np.sin(ω)
Ti[0,1] = -np.cos(Ω) * np.sin(ω) - np.sin(Ω) * np.cos(i) * np.cos(ω)
Ti[0,2] = np.sin(Ω) * np.sin(i)
Ti[1,0] = np.sin(Ω) * np.cos(ω) + np.cos(Ω) * np.cos(i) * np.sin(ω)
Ti[1,1] = -np.sin(Ω) * np.sin(ω) + np.cos(Ω) * np.cos(i) * np.cos(ω)
Ti[1,2] = -np.cos(Ω) * np.sin(i)
Ti[2,0] = np.sin(i) * np.sin(ω)
Ti[2,1] = np.sin(i) * np.cos(ω)
Ti[2,2] = np.cos(i)

print ('Matriz de transformacion, Ti: \n', Ti, '\n')


# #### Anomalia excentrica

# In[80]:


Es = []  # Almacena la anomalia excentrica de cada posicion

for Mi in M:
    def f(x):
        return (x - e * np.sin(x) - Mi)  #calculo anomalia excentrica
    
    E = optimize.newton(f, 1.5)    # calculo de E por Newton-Raphson  
    Es.append(E)
    #print (E, ' ')
#print('Es:', Es)


# #### Coordenadas orbitales en polares

# In[81]:


r = a * (1 - e * np.cos(Es)) #km    
φ = np.arccos(1/e * (1 - (a * (1 - e**2)) / r)) #rad    

φ_old = np.pi

for i in range(len(φ)):
    if(φ_old < φ[i]):
        φ[i]=-φ[i]
    else:
        φ[i]=φ[i]
    φ_old = φ[i]
    
φ=-φ    
φ=np.pi + φ

plt.subplot(211)
plt.title('φ (rad)')
plt.grid()
plt.plot(φ)

plt.subplot(212)
plt.title('r (km)')
plt.grid()
plt.plot(r)


# #### Calculo coordenadas orbitales

# In[82]:


Xo = r * np.cos(φ)
Yo = r * np.sin(φ)
Zo = np.zeros(len(Xo))
coord_o=[Xo, Yo, Zo]

plt.figure()
plt.plot(Xo, Yo)
plt.title('Coordenadas orbitales')
plt.grid()
plt.show()


# #### Calculo coordenadas inerciales

# In[83]:


coord_i=np.matmul(Ti,coord_o)
#print('coord_inerciales: ', coord_i)


# #### Representacion coordenadas inerciales

# In[84]:


graf=plt.figure()
plano=graf.add_subplot(111,projection='3d')
x_ci=np.array([coord_i[0]])
y_ci=np.array([coord_i[1]])
z_ci=np.array([coord_i[2]])
plano.plot_wireframe(x_ci, y_ci, z_ci)


# #### Representacion coordenadas inerciales

# In[85]:


figu=plt.figure()
ax = figu.add_subplot(111, projection='3d')
p=ax.scatter(coord_i[0], coord_i[1], coord_i[2], c=r)

def sphere(r):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = r * np.outer(np.cos(u), np.sin(v)) #multiplicacion vectores
    y = r * np.outer(np.sin(u), np.sin(v))
    z = r * np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z

x_sph, y_sph, z_sph = sphere(re)
ax.scatter(x_sph, y_sph, z_sph, c='blue')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

figu.colorbar(p, ax=ax, label='radio')


ax.set_xlim(-40000,40000)
ax.set_ylim(-40000,40000)


# #### Funcion de transformacion coordenadas cartesianas - esfericas

# In[86]:


def cart2sph(x,y,z):
    azimuth = np.arctan2(y,x)
    elevation = np.arctan2(z,np.sqrt(x**2 + y**2))
    r = np.sqrt(x**2 + y**2 + z**2)
    return r, elevation, azimuth

def sph2cart(r,elevation,azimuth):
    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)
    return x, y, z


# In[87]:


r_i, ele_i, azi_i=cart2sph(coord_i[0], coord_i[1], coord_i[2])
#print(azi_i)
#print(ele_i)


# #### Proyeccion coordenadas inerciales

# In[88]:


plt.figure()
plt.scatter(azi_i, ele_i, c=r_i, cmap='plasma')
plt.title('Coordenadas inerciales')
plt.xlabel('azimute')
plt.ylabel('elevacion')
plt.colorbar(label='radio')
plt.grid()
plt.show()


#  #### Matriz de transformacion inercial rotatoria

# In[74]:


#velocidad de rotacion de la Tierra Ωe y tiempo transcurrido Te desde que Xr=Xi 
Tc = (JD - 2415020)/36525
αgo= 99.6909833 + 36000.7689 * Tc + 3.8708 * 10**-4  * Tc**2
ΩeTes = (αgo + 0.25068447 * t_dias * min_sidereo) * np.pi/180

coord_r=np.empty(coord_i.shape)
for i, ΩeTe in enumerate(ΩeTes):
    Tir = np.zeros((3,3))
    Tir[0,0] = np.cos(ΩeTe)
    Tir[0,1] = np.sin(ΩeTe)
    Tir[1,0] = -np.sin(ΩeTe)
    Tir[1,1] = np.cos(ΩeTe)
    Tir[2,2] = 1
    coord_r[0][i], coord_r[1][i], coord_r[2][i] = np.matmul(Tir, [coord_i[0][i], coord_i[1][i], coord_i[2][i]])


# #### Coordenadas rotacionales esfericas

# In[75]:


r_r, ele_r, azi_r = cart2sph(coord_r[0], coord_r[1], coord_r[2])
r_r = r_r * 180/np.pi
ele_r = ele_r * 180/np.pi
azi_r = azi_r * 180/np.pi


# #### Representacion coordenadas rotacionales Ground Track

# In[89]:


plt.figure()
plt.scatter(azi_r, ele_r, c=r_r)
plt.xlim(-180, 180)
plt.ylim(-90, 90)
plt.grid()
plt.title('Ground Track')
plt.xlabel('Longitud')
plt.ylabel('Latitud')
plt.colorbar(label='radio')
plt.show()


# In[77]:


print(t_dias)
print('\n \n')
print(tp)


# In[ ]:





# In[ ]:





# In[ ]:




