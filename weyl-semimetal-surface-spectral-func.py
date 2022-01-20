import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, sin
from numpy import linalg as LA
from scipy import signal
import scipy
# %matplotlib qt

ax = np.newaxis

# define Pauli matrices
# here we use python broadcasting
# the indexing convention will be [kx,ky,kz,i,j]
# where i,j are the matrix indices of the Hamiltonian
s0 = np.array([[1,0],[0,1]])[ax,ax,ax,:,:]
sx = np.array([[0,1],[1,0]])[ax,ax,ax,:,:]
sy = np.array([[0,-1j],[1j,0]])[ax,ax,ax,:,:]
sz = np.array([[1,0],[0,-1]])[ax,ax,ax,:,:]

# array of all pauli patrices
ss = np.array(
    [ [[1,0],[0,1]],
    [[0,1],[1,0]],
    [[0,-1j],[1j,0]],
    [[1,0],[0,-1]] ]
)

def s(a,b):
    # Kronecker product between two Pauli matrices ss[a] and ss[b]
    return np.kron(ss[a],ss[b])[ax,ax,ax,:,:]

# points in momentum space to consider
Nkx = 150
Nky = 150
Nkz = 150
eta = 0.05

# Potential of the single impurity at the surface used in QPI
U_imp = 0.1

def build_k(N):
    # build a momentum space array [-Pi/2, Pi/2]
    return np.arange(-N//2, N//2)/N*2*np.pi

# momentum space arrays
kx_ = build_k(Nkx)
ky_ = build_k(Nky)
kz_ = build_k(Nkz)

# spacing between k-points
dkx = kx_[1] - kx_[0]
dky = ky_[1] - ky_[0]
dkz = kz_[1] - kz_[0]
dk = [dkx, dky, dkz]

# adopt python broadcasting with convention [kx,ky,kz,i,j]
kx = kx_[:,ax,ax, ax,ax]
ky = ky_[ax,:,ax, ax,ax]
kz = kz_[ax,ax,:, ax,ax]
k = np.array([kx,ky,kz])

# define axis normal to surface termination
# 0 - x axis (yz surface termination)
# 1 - y axis (xz surface termination)
# 2 - z axis (xy surface termination)
bound_axis = 1
dkb = dk[bound_axis]
dk2 = np.prod(dk)/dkb
kb = k[bound_axis] # 


########## H1 #################
# a=1
# b=1
# t=-1.5
# d=0
# m=0
# beta=0.9
# alpha=0.3
# g0 = 2*d*(2-cos(kx)-cos(ky))
# g1 = a*sin(kx)
# g2 = a*sin(ky)
# g3 = m + t*cos(kz) + 2*b*(2-cos(kx)-cos(ky))
# E = 0

# H = g1*s(1,3) + g2*s(2,0) + g3*s(3,0) \
#     + g0*s(0,0) + beta*s(2,2) + alpha*sin(ky)*s(1,2)

########## H2 #################

a=1
b=1
t=-1.5
l=0.5
d=0.1
alpha=0.3
beta=0.7
m=0
E = 0

g0 = 2*d*(2-cos(kx)-cos(ky))
g1 = a*sin(kx)
g2 = a*sin(ky)
g3 = m + t*cos(kz) + 2*b*(2-cos(kx)-cos(ky))
H = g1*s(1,3) + g2*s(2,0) * g3*s(3,0) + d*s(2,3) +beta*s(2,2) + alpha*sin(ky)*s(1,2) + l*sin(kz)*s(0,1)


########## H3 #################

# t = 1
# k0 = np.pi/2
# tx = 0.5
# m = 2
# g = 1.4
# E = 0.25

# H = g*(sin(2*kx)-cos(k0)) * (cos(kz) - cos(k0)) * s0 \
#     -m*(1-cos(kz)**2-cos(ky)+2*tx*(cos(kx)-cos(k0))) * sx \
#         -2*t*sin(ky) * sy - 2*t*cos(kz)*sz
# # H = g*(cos(2*kx)-cos(k0)) * (cos(kz) - cos(k0)) * s0 \
# #     -m*(1-cos(kz)**2-cos(ky))* sx +2*tx*(cos(kx)-cos(k0)) * sx \
# #         -2*t*sin(ky) * sy - 2*t*cos(kz)*sz

###########################
id_ = np.eye(H.shape[-1])
id = id_[ax,ax,ax,:,:] # identity matrix

w = E # set the frequency to the Fermi energy

# define the inverse Green's function
Ginv = (w + 1j*eta)*id - H

# matrix of impurity
# adjust this to define the position of the impurity on the sublattice
U_imp_m = np.eye(Ginv.shape[-1])

def inv(x):
    # inverse of the matrix corresponding to the last indices of array x
    return np.linalg.inv(x)

# compute the Green's function
G = inv(Ginv)

# compute the transfer matrix
T = -inv(dkb/(2*np.pi)*np.sum(G, axis=bound_axis))

dist = -1
G0_ft = dkb/(2*np.pi)*np.sum(np.exp(1j*dist*kb)*G, axis=bound_axis)
G0_ft_ = dkb/(2*np.pi)*np.sum(np.exp(-1j*dist*kb)*G, axis=bound_axis)

Gsurface_correction = np.einsum('xzij,xzjk,xzkl->xzil', G0_ft,T,G0_ft_)


Gbulk_int = dkb/(2*np.pi)*np.sum(G,axis=bound_axis)
Gsurface = Gbulk_int + Gsurface_correction

Asurface_correction = -np.imag(np.trace(Gsurface_correction, axis1=2, axis2=3))
Abulk_int = -np.imag(np.trace(Gbulk_int, axis1=2, axis2=3))
Asurface = Abulk_int + Asurface_correction


#%% Plot the bounadry spectral function

plt.figure('bound spect f')
plt.clf()
plt.pcolormesh(kx_/np.pi,kz_/np.pi,Asurface.T, cmap='inferno', shading='auto')
plt.xlabel('$k_x/\pi$')
plt.ylabel('$k_z/\pi$')
plt.title('Boundary Spectral function')
plt.colorbar()
plt.axis('equal')
plt.tight_layout()


#%% Compute and plot QPI spectrum

T_imp = inv(id_ - U_imp * dk2 / (2*np.pi)**2 * U_imp_m @ np.sum(Gsurface, axis=(0,1)) ) @ U_imp_m * U_imp
Gsurface2 = np.flip(np.roll(np.roll(Gsurface, -1, axis=0), -1, axis=1), axis=(0,1))

ifft = np.fft.ifftn
fft = np.fft.fftn
fftshift = scipy.fft.fftshift

qpi = dk2/(2*np.pi)**2*np.einsum('xzij,jk,xzki->xz', fft(Gsurface, axes=(0,1)), T_imp, fft(Gsurface2, axes=(0,1)))
qpi = np.imag(fftshift(ifft(qpi, axes=(0,1)), axes=(0,1)))

qpi0 = dk2/(2*np.pi)**2*np.sum(Asurface, axis=(0,1)) # what is the correct power of 2 Pi


# Plot QPI spectrum
plt.figure('QPI')
plt.clf()
plt.pcolormesh(kx_/np.pi,kz_/np.pi,(qpi0+qpi).T, cmap='inferno', shading='auto')
plt.xlabel('$k_x/\pi$')
plt.ylabel('$k_z/\pi$')
plt.title('QPI spectrum')
plt.colorbar()
plt.axis('equal')
plt.tight_layout()

#%% Compute the JDOS

plt.figure('JDOS')
plt.clf()

def fft_convolve(a,b, axes=None):
    return scipy.fft.fftshift(np.fft.ifftn(np.fft.fftn(a, axes=axes) * np.fft.fftn(b, axes=axes), axes=axes), axes=axes)

conv = fft_convolve(Asurface, Asurface, axes=(0,1)).T
plt.pcolormesh(kx_/np.pi,kz_/np.pi, conv.real, cmap='inferno', shading='auto')
plt.xlabel('$k_x/\pi$')
plt.ylabel('$k_z/\pi$')
plt.title('JDOS')
plt.colorbar()
plt.axis('equal')
plt.tight_layout()