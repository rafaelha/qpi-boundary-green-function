import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, sin
from numpy import linalg as LA
from scipy import signal
import scipy
# %matplotlib qt

ax = np.newaxis
s0 = np.array([[1,0],[0,1]])[ax,ax,ax,:,:]
sx = np.array([[0,1],[1,0]])[ax,ax,ax,:,:]
sy = np.array([[0,-1j],[1j,0]])[ax,ax,ax,:,:]
sz = np.array([[1,0],[0,-1]])[ax,ax,ax,:,:]

ss = np.array(
    [ [[1,0],[0,1]], 
    [[0,1],[1,0]], 
    [[0,-1j],[1j,0]], 
    [[1,0],[0,-1]] ]
)

def s(a,b):
    return np.kron(ss[a],ss[b])[ax,ax,ax,:,:]


Nkx = 150
Nky = 450
Nkz = 150
eta = 0.05

E = 0

U_imp = 0.1

def build_k(N):
    return np.arange(-N//2, N//2)/N*2*np.pi

kx_ = build_k(Nkx)
ky_ = build_k(Nky)
kz_ = build_k(Nkz)
dkx = kx_[1] - kx_[0]
dky = ky_[1] - ky_[0]
dkz = kz_[1] - kz_[0]
dk = [dkx, dky, dkz]

kx = kx_[:,ax,ax, ax,ax]
ky = ky_[ax,:,ax, ax,ax]
kz = kz_[ax,ax,:, ax,ax]

w = E + 1j*eta

bound_axis = 1
dkb = dk[bound_axis]
dk2 = np.prod(dk)/dkb




########## H1 #################
a=1
b=1
t=-1.5
d=0
m=0
beta=0.9
alpha=0.3
g0 = 2*d*(2-cos(kx)-cos(ky))
g1 = a*sin(kx)
g2 = a*sin(ky)
g3 = m + t*cos(kz) + 2*b*(2-cos(kx)-cos(ky))

H = g1*s(1,3) + g2*s(2,0) + g3*s(3,0) \
    + g0*s(0,0) + beta*s(2,2) + alpha*sin(ky)*s(1,2)

########## H2 #################

# a=1
# b=1
# t=-1.5
# l=0.5
# d=0.1
# alpha=0.3
# beta=0.7
# m=1 # ??

# g0 = 2*d*(2-cos(kx)-cos(ky))
# g1 = a*sin(kx)
# g2 = a*sin(ky)
# g3 = m + t*cos(kz) + 2*b*(2-cos(kx)-cos(ky))
# H = g1*s(1,3) + g2*s(2,0) * g3*s(3,0) + d*s(2,3)\
#     +beta*s(2,2) + alpha*sin(ky)*s(1,2) + l*sin(kz)*s(0,1)


########## H3 #################

# t = 1
# k0 = np.pi/2
# tx = 0.5
# m = 2
# g = 1.4

# # H = g*(sin(2*kx)-cos(k0)) * (cos(kz) - cos(k0)) * s0 \
# #     -m*(1-cos(kz)**2-cos(ky)+2*tx*(cos(kx)-cos(k0))) * sx \
# #         -2*t*sin(ky) * sy - 2*t*cos(kz)*sz
# H = g*(cos(2*kx)-cos(k0)) * (cos(kz) - cos(k0)) * s0 \
#     -m*(1-cos(kz)**2-cos(ky))* sx +2*tx*(cos(kx)-cos(k0)) * sx \
#         -2*t*sin(ky) * sy - 2*t*cos(kz)*sz

###########################

Ginv = w - H

id = np.eye(Ginv.shape[-1])
U_imp_m = np.eye(Ginv.shape[-1])

def inv(x):
    return np.linalg.inv(x)
G = inv(Ginv)

T = -inv(dkb/(2*np.pi)*np.sum(G, axis=bound_axis))

dist = -1

G0 = dkb/(2*np.pi)*np.sum(np.exp(1j*dist*ky)*G, axis=bound_axis)
G0_ = dkb/(2*np.pi)*np.sum(np.exp(-1j*dist*ky)*G, axis=bound_axis)

Gb = dkb/(2*np.pi)*np.sum(G,axis=bound_axis)
Gs_ = np.einsum('xzij,xzjk,xzkl->xzil', G0,T,G0_)
Gs = Gb + Gs_

As_ = -np.imag(np.trace(Gs_, axis1=2, axis2=3))
Ab = -np.imag(np.trace(Gb, axis1=2, axis2=3))
As = Ab + As_


#%%
plt.figure('bound spect f')
plt.clf()
plt.pcolormesh(kx_/np.pi,kz_/np.pi,As.T, cmap='inferno', shading='auto')
plt.xlabel('$k_x/\pi$')
plt.ylabel('$k_y/\pi$')
plt.title('Boundary Spectral function')
plt.colorbar()
plt.axis('equal')
plt.tight_layout()


#%% QPI

T_imp = inv(id - U_imp * dk2 / (2*np.pi)**2 * U_imp_m @ np.sum(Gs, axis=(0,1)) ) @ U_imp_m * U_imp
Gs2 = np.flip(np.roll(np.roll(Gs, -1, axis=0), -1, axis=1), axis=(0,1))

ifft = np.fft.ifftn
fft = np.fft.fftn
fftshift = scipy.fft.fftshift

qpi = dk2/(2*np.pi)**2*np.einsum('xzij,jk,xzki->xz', fft(Gs, axes=(0,1)), T_imp, fft(Gs2, axes=(0,1)))
qpi = np.imag(fftshift(ifft(qpi, axes=(0,1)), axes=(0,1)))

qpi0 = dk2/(2*np.pi)**2*np.sum(As, axis=(0,1)) # what is the correct power of 2 Pi


plt.figure('qpi')
plt.clf()
plt.pcolormesh(kx_/np.pi,kz_/np.pi,(qpi0+qpi).T, cmap='inferno', shading='auto')
plt.xlabel('$k_x/\pi$')
plt.ylabel('$k_y/\pi$')
plt.title('QPI signal')
plt.colorbar()
plt.axis('equal')
plt.tight_layout()

#%%
plt.figure('conv')
plt.clf()

def fft_convolve(a,b, axes=None):
    return scipy.fft.fftshift(np.fft.ifftn(np.fft.fftn(a, axes=axes) * np.fft.fftn(b, axes=axes), axes=axes), axes=axes)

conv = fft_convolve(As, As, axes=(0,1)).T
plt.pcolormesh(kx_/np.pi,kz_/np.pi, conv.real, cmap='inferno', shading='auto')
# plt.pcolormesh(conv.real, cmap='inferno', shading='auto')
plt.xlabel('$k_x/\pi$')
plt.ylabel('$k_y/\pi$')
plt.title('JDOS')
plt.colorbar()
plt.axis('equal')
plt.tight_layout()

#%%
plt.figure('test')
plt.clf()
Gt = np.trace(Gs, axis1=2, axis2=3)
# Gt = ifft(np.imag(fft(Gt,axes=(0,1))), axes=(0,1))
plt.pcolormesh(ifft(np.imag(fft(Gt))).real)
plt.colorbar()