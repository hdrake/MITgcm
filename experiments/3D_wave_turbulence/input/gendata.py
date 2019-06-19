import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.fftpack import fftn, ifftn

# create grid
nx = 256
ny = 128
nz = 210

dx = 500
dy = 500
dz = 6
nz_vary = np.int(nz*0.65)
dz = dz * np.ones((nz))[:,np.newaxis,np.newaxis]*np.ones((nz,ny,nx))
dz[nz_vary:,:,:] = dz[nz_vary,0,0]*1.0275**np.arange(0,nz-nz_vary,1.0)[:,np.newaxis,np.newaxis]*np.ones((nz-nz_vary,ny,nx))

Lx = nx*dx
Ly = ny*dy
Hz = sum(dz[:,0,0])

x = np.arange(dx/2.0,Lx,dx)[np.newaxis,np.newaxis,:]*np.ones((nz,ny,nx))
y = np.arange(dy/2.0,Ly,dy)[np.newaxis,:,np.newaxis]*np.ones((nz,ny,nx))
z = (-Hz + np.cumsum(dz,axis=0) - dz/2.0)

# Create topography
hill_mode = False
canyon_mode = False
mean_slope = 2.e-3
x0_slope = 8000.
x0_idx = np.argmin(np.abs(x[0,0,:]-x0_slope))
canyon_width = ny*dy/5. # from Thurnherr 2005
canyon_height = 500. # from Thurnherr 2005
canyon_slope = canyon_height/canyon_width
hill_slope = 2.e-2
hill_length = 8000.
hill_height = hill_length * hill_slope
zC_offset = 0
Hbot = np.zeros_like(x)
Hbot[0,:,x0_idx:] = (x[0,:,x0_idx:]-x[0,0,x0_idx])*mean_slope
if canyon_mode:
    Hbot[0,np.int(ny*1.5/5.):np.int(ny*2.5/5.),:] -= (y[0,np.int(ny*1.5/5.):np.int(ny*2.5/5.),:]-y[0,np.int(ny*1.5/5.),:])*canyon_slope
    Hbot[0,np.int(ny*2.5/5.):np.int(ny*3.5/5.),:] -= (y[0,np.int(ny*3.5/5.),:]-y[0,np.int(ny*2.5/5.):np.int(ny*3.5/5.),:])*canyon_slope
    zC_offset -= canyon_height
Hbot += canyon_height
if hill_mode:
    nxhill = 2*int(hill_length/dx)
    nhill = int((nx-x0_idx*2)/nxhill)
    Hhill = np.zeros_like(x)
    for i in range(nhill):
        Hhill[0,:,x0_idx+i*nxhill:x0_idx+i*nxhill+nxhill//2] = (x[0,:,x0_idx+i*nxhill:x0_idx+i*nxhill+nxhill//2]-x[0,0,x0_idx+i*nxhill])*hill_slope
        Hhill[0,:,x0_idx+i*nxhill+nxhill//2:x0_idx+i*nxhill+nxhill] = (Hhill[0,0,x0_idx+i*nxhill+nxhill//2-1]
                                                      - (x[0,:,x0_idx+i*nxhill+nxhill//2:x0_idx+i*nxhill+nxhill]
                                                         -x[0,0,x0_idx+i*nxhill+nxhill//2])*hill_slope)
    Hbot[0,:,:] += Hhill[0,:,:]
Hbot[0,:,-x0_idx:] = Hbot[0,:,-x0_idx-1][:,np.newaxis]

Hbot = Hbot - Hz
Hbot[0,:,0] = 0
Hbot[0,:,-1] = 0

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.pcolor(x[0,:,:]*1e-3,y[0,:,:]*1e-3,Hbot[0,:,:])
plt.xlabel('zonal distance [km]')
plt.ylabel('meridional distance [km]')
plt.colorbar()
plt.clim([np.min(z),np.min(z)+900])

plt.subplot(1,2,2)
plt.plot(x[0,0,:]*1e-3,Hbot[0,ny//2,:])
plt.xlim([0,nx*dx*1e-3])
plt.xlabel('zonal distance [km]')
plt.ylabel('depth [m]')

plt.tight_layout()
plt.show()

# generate vertical eddy diffusivity field
d = 230
k0 = 5.2e-5
k1 = 1.8e-3
K = np.zeros((nz,ny,nx))
for i in range(nx):
    for j in range(ny):
        K[:,j,i] = k0 + k1*np.exp(-(z[:,0,0]-Hbot[0,j,i])/d)
K[K>(k1+k0)] = k1+k0

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.pcolor(x[:,ny//2,:]*1e-3,z[:,ny//2,:],K[:,ny//2,:],norm=colors.LogNorm(vmin=5.e-5,vmax=2.e-3))
plt.fill_between(x[0,ny//2,:]*1e-3,-3500+np.zeros_like(x[0,ny//2,:]),Hbot[0,ny//2,:],color=(0.8,0.8,0.8))
plt.colorbar()
plt.ylim([np.min(z),-1000+zC_offset])
plt.xlim([0,nx*dx*1e-3]);

plt.subplot(1,2,2)
plt.pcolor(y[:,:,nx//2]*1e-3,z[:,:,nx//2],K[:,:,nx//2],norm=colors.LogNorm(vmin=5.e-5,vmax=2.e-3))
plt.fill_between(y[0,:,nx//2]*1e-3,-3500+np.zeros_like(y[0,:,nx//2]),Hbot[0,:,nx//2],color=(0.8,0.8,0.8))
plt.colorbar()
plt.ylim([np.min(z),-1000])
plt.xlim([0,ny*dy*1e-3]);

plt.tight_layout()
plt.show()

# generate initial conditions and temperature restoring
N = 1.3e-3
U = np.zeros((nz,ny,nx))
V = np.zeros((nz,ny,nx))
T = (N**2)/(9.81*2e-4) * (z+Hz) + 1.e-7*(np.random.random((nz,ny,nx))-0.5)
R = T[:,:1,0]

# Initial tracer concentration
xC_idx = np.argmin(np.abs(x[0,0,:]-40000))
yC_idx = ny//2-1
xC = x[0,0,xC_idx]; yC = y[0,yC_idx,0]; zC = -1600 + zC_offset;
delx = 900.; dely = 900.; delz = 30.;

C = np.exp(-((x-xC)/delx)**2 - ((z-zC)/delz)**2 - ((y-yC)/dely)**2)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.pcolor(x[:,yC_idx,:]*1e-3,z[:,yC_idx,:],C[:,yC_idx,:])
plt.colorbar(label="concentration")
plt.fill_between(x[0,0,:]*1e-3,-3500+np.zeros_like(x[0,0,:]),Hbot[0,yC_idx,:],color=(0.8,0.8,0.8))
plt.xlim([0,nx*dx*1e-3])
plt.ylim([-2000+zC_offset,-1000+zC_offset])
plt.xlabel("zonal distance [km]")
plt.ylabel("elevation [m]")
plt.clim([0,1])

plt.subplot(1,2,2)
plt.pcolor(y[:,:,xC_idx]*1e-3,z[:,:,xC_idx],C[:,:,xC_idx])
plt.fill_between(y[0,:,xC_idx]*1e-3,-3500+np.zeros_like(y[0,:,xC_idx]),Hbot[0,:,xC_idx],color=(0.8,0.8,0.8))
plt.xlim([0,ny*dy*1e-3])
plt.ylim([-1750+zC_offset,-1000])
plt.xlabel("meridional distance [km]")
plt.ylabel("elevation [m]")
plt.clim([0,1])

plt.tight_layout()
plt.show()

# Reverse vertical axis so first index is at the surface and transpose axes
U = U[::-1,:,:]
V = V[::-1,:,:]
T = T[::-1,:,:]
K = K[::-1,:,:]
C = C[::-1,:,:]

# Reverse vertical axis
R = R[::-1,:]
dz = dz[::-1,:1,:1]

# save input data as binary files
newFile = open("U.init", "wb")
newFile.write(bytes(U.astype('>f8')))
newFile = open("T.init", "wb")
newFile.write(bytes(T.astype('>f8')))
newFile.close()

newFile = open("kappa.init", "wb")
newFile.write(bytes(K.astype('>f8')))
newFile.close()

newFile = open("ptracer01_init.bin", "wb")
newFile.write(bytes(C.astype('>f8')))
newFile.close()

newFile = open("topog.init", "wb")
newFile.write(bytes(Hbot[0,:].astype('>f8')))
newFile.close()

newFile = open("delZ.init", "wb")
newFile.write(bytes(dz[:,0].astype('>f8')))
newFile.close()

# save restoring as a ascii / text file
np.savetxt('R.init',R)
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.fftpack import fftn, ifftn

