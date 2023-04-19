from cProfile import label
import numpy as np
import h5py
import matplotlib.pyplot as plt
import astropy
from astropy.io import ascii
import astropy.units as u
from matplotlib import units
from numpy import histogram2d
import matplotlib.colors as colors
import bisect
import math
from math import sin, cos, tan, atan
import time
import warnings
import sys
import os
from numpy import array, sqrt, sin, cos, tan

import numpy as np
from scipy.optimize import root_scalar

# 宇宙学参数
Omega_m = 0.31  # 物质密度参数
Omega_Lambda = 0.69  # 暗能量密度参数
H_0 = 67.66  # 哈勃常数，单位为 km/s/Mpc
c = 299792.458  # 光速，单位为 km/s

# 计算共动距离和尺度因子的函数
def comoving_distance(z):
    from scipy.integrate import quad
    integrand = lambda zp: 1.0 / np.sqrt(Omega_m * (1.0+zp)**3 + Omega_Lambda)
    integral, _ = quad(integrand, 0, z)
    return c/H_0 * integral

def scale_factor(z):
    return 1.0 / (1.0 + z)

# 已知共动距离，求红移和尺度因子的函数
def find_redshift_and_scale_factor(dC):
    # 定义被求解的方程
    equation = lambda z: comoving_distance(z) - dC

    # 使用root_scalar函数求解方程
    sol = root_scalar(equation, bracket=[0, 10])

    # 检查解是否收敛，如果不收敛则返回None
    if not sol.converged:
        return None, None

    # 返回红移和尺度因子
    z = sol.root
    a = scale_factor(z)
    return z, a

# 假设共动距离为5330 Mpc
dC = 5330

# 求解红移和尺度因子
z, a = find_redshift_and_scale_factor(dC)

# 打印结果
if z is None:
    print("无法求解红移和尺度因子。")
else:
    print("共动距离 dC = {:.2f} Mpc 对应的红移为 z = {:.4f}，尺度因子为 a = {:.4f}".format(dC, z, a))


warnings.simplefilter('always', RuntimeWarning)
#########
me = 9.1*10**-28
mH = 1.67*10**-24
solartog = 2.0*10**33
def M6(r,r0, h, D):
    import numpy as np
    from numpy import pi, sqrt, exp
    ##########D = dimension#######
    ##########h = smoothing length########
    d = np.sqrt((r-r0)**2)
    q = d/h
    V = [2.0, 2*pi*d, 4*pi*d**2]
    sigma = np.array([1.0/120*3, 7.0/(478.0*np.pi)*3**2, 1.0/(120*np.pi)*3**3])
    if q >= 0 and q < 1.0/3:
        w = (3.0)**5*(1.0 - q)**5 - 6*(3.0)**5*(2.0/3-q)**5+15*(3)**5*(1.0/3-q)**5
    elif q >= 1.0/3 and q < 2.0/3:
        w = (3.0)**5*(1.0 - q)**5 - 6*(3.0)**5*(2.0/3-q)**5
    elif q >= 2.0/3 and q < 1.0:
        w = (3.0)**5*(1.0 - q)**5
    else:
        w = 0
    W =  h**(-D)*sigma[D-1]*w*V[D-1]
    return W


def W(qz,qxy,D):
    import numpy as np
    from numpy import pi, sqrt, exp
    ##########D = dimension#######
    ##########h = smoothing length########
    q = np.sqrt(qxy**2 + qz**2)
    sigma = np.array([1.0/120*3, 7.0/(478.0*np.pi)*3**2, 1.0/(120*np.pi)*3**3])
    if q >= 0 and q < 1.0/3:
        w = (3.0)**5*(1.0 - q)**5 - 6*(3.0)**5*(2.0/3-q)**5+15*(3)**5*(1.0/3-q)**5
    elif q >= 1.0/3 and q < 2.0/3:
        w = (3.0)**5*(1.0 - q)**5 - 6*(3.0)**5*(2.0/3-q)**5
    elif q >= 2.0/3 and q < 1.0:
        w = (3.0)**5*(1.0 - q)**5
    else:
        w = 0
    W =  sigma[D-1]*w
    return W

    

def Gauss(r, r0, h, D):
    from numpy import pi, sqrt, exp
    d = sqrt((r-r0)**2)
    V = [2.0, 2*pi*d, 4*pi*d**2]
    sigma = [1/sqrt(pi), 1/pi, 1.0/(pi*sqrt(pi))]
    W = h**(-D)*sigma[D-1]*exp(-(r-r0)**2/h**2)
    W = W*V[D-1]
    return W



def F(qxy):
    from scipy import integrate
    v ,err = integrate.quad(W,  -np.sqrt(1-qxy**2), np.sqrt(1-qxy**2), args = (qxy,3))
    return v
qxy =np.linspace(0, 1, 1000)
Fxy = np.array(list(map(F, qxy)))

def F_xy(q):
    qxy =np.linspace(0, 1, 1000)
    q_xy = bisect.bisect(qxy, q) - 1 
    W = Fxy[q_xy]
    return W
plt.plot(qxy,Fxy)

def handle_boundary_conditions(pos, L):
    """
    处理周期性边界条件
    """
    # 检查每个维度是否超出边界
    for i in range(3):
        idx = pos[:, i] < 0
        pos[idx, i] += L[i]
        idx = pos[:, i] >= L[i]
        pos[idx, i] -= L[i]
    return pos

###############################L50
# sim name, snapshot file, FoF file.
snapfile = [["/home/oku/work/L50N256/snapshot_020/snapshot_020"]]
# set alias name
alias_snap = {
    "rho": "PartType0/Density",
    "u": "PartType0/InternalEnergy",
    "sphpos": "PartType0/Coordinates",
    "dmpos": "PartType1/Coordinates",
    "starpos": "PartType4/Coordinates",
    "smoothlen": "PartType0/SmoothingLength"
}

alias_frac = {    
    "n_HI": "PartType0/HI",
    "n_HII": "PartType0/HII",
    'n_DI': "PartType0/DI",
    'n_DII': "PartType0/DII",
    "n_e": "PartType0/ElectronAbundance",
    "n_HeI": "PartType0/HeI",
    "n_HeII": "PartType0/HeII",
    "n_HeIII": "PartType0/HeIII",
    "n_H2I": "PartType0/H2I",
    "n_H2II": "PartType0/H2II",
    
 }

alias_T = {
     "T": "PartType0/Temperature"
}  

alias_mass = {
    "sphmass": 0,
    "dmmass": 1,
    "starmass": 4
}
alias_fof = {
    "halomass": "Group/GroupMass",
    "halopos": "Group/GroupPos",
    "halonum": "Group/GroupNsubs",
    "halo0": "Group/GroupFirstSub",
    "halolen": "Group/GroupLen",
    "haloRV" : "Group/Group_R_Crit200",
    "haloMV": "Group/Group_M_Crit200"
}


alias_subfind = {
    "subhalopos": "Subhalo//SubhaloPos",
    "subhalomass": "Subhalo/SubhaloMass",
    "subhaloGrNr": "Subhalo/SubhaloGrNr",
    "subhaloparent": "Subhalo/SubhaloParent",
    "subhalolen" : "Subhalo/SubhaloLen"
    
}

# radial bin of log(r[Mpc])
Xbins = np.linspace(-2, 1, num=20)
Xpoints = (Xbins + np.roll(Xbins, -1))[0:-1] / 2
# ----------------------------------------#
# initialization
data = [{} for i in range(len(snapfile))]
for i in range(len(snapfile)):
    for key in alias_snap.keys():
        data[i][key] = np.empty((0, 3))
    for key in alias_mass.keys():
        data[i][key] = np.empty((0, 3))
    for key in alias_frac.keys():
        data[i][key] = np.empty((0, 3))
    for key in alias_T.keys():
        data[i][key] = np.empty((0, 3))
    for key in alias_fof.keys():
        data[i][key] = np.empty((0, 3))
# load header, particle and halo data
for i in range(len(snapfile)):
    print("\nstart reading data of {}...".format(snapfile[i][0]))
    try:
        fname = "{}.hdf5".format(snapfile[i][0])
        f = h5py.File(fname, "r")
    except:
        fname = "{}.0.hdf5".format(snapfile[i][0])
        f = h5py.File(fname, "r")
    # load header
    a = f["Header"].attrs["Time"]
    h = f["Parameters"].attrs["HubbleParam"]
    Omega_b = f["Parameters"].attrs["OmegaBaryon"]
    comoving = f["Parameters"].attrs["ComovingIntegrationOn"]
    if comoving == 1:
        print("This is cosmological run. Scaling code values to physical value.")
        print("a = {}, h = {}".format(a, h))
    else:
        print("This is non-cosmological run. Setting a = h = 1.")
        a = h = 1
    MassTable = f["Header"].attrs["MassTable"]
    print("Mass table: {}".format(MassTable))
    NumPart_Total = f["Header"].attrs["NumPart_Total"]
    Boxsize = f["Header"].attrs["BoxSize"]
    print("Total number of particles: {}".format(NumPart_Total))
    nfile = f["Header"].attrs["NumFilesPerSnapshot"]
    SPHdensitytocgs = f['PartType0']['Density'].attrs['to_cgs']
    SPHmasstocgs = f['PartType0']['Masses'].attrs['to_cgs']
    f.close()
    # load particle data
    for j in range(nfile):
        if nfile == 1:
            fname = "{}.hdf5".format(snapfile[i][0])
        else:
            fname = "{}.{}.hdf5".format(snapfile[i][0], j)
        f = h5py.File(fname, "r")
        #d[i] = f
        NumPart_ThisFile = f["Header"].attrs["NumPart_ThisFile"]
        print("\nreading {}th file.".format(j))
        print("Number of particles in this file: {}".format(NumPart_ThisFile))
        for key in alias_snap.keys():
            print("loading {}.\n\tmultipling by a^{} h^{} to get physical value".format(key,f[alias_snap[key]].attrs["a_scaling"], f[alias_snap[key]].attrs["h_scaling"]))
            array = np.array(f[alias_snap[key]])
            array *= a**f[alias_snap[key]].attrs["a_scaling"]
            array *= h**f[alias_snap[key]].attrs["h_scaling"]
            if len(np.shape(array)) == 1:
                # scalar value
                data[i][key] = np.append(data[i][key], array)
            else:
                # vector value
                data[i][key] = np.vstack((data[i][key], array))
                print("loading {}".format(key))
        for key in alias_frac.keys():
            array = np.array(f[alias_frac[key]])
            if len(np.shape(array)) == 1:
                array = np.array(f[alias_frac[key]])
                # scalar value
                data[i][key] = np.append(data[i][key], array)
            else:
                # vector value
                array = np.array(f[alias_frac[key]])
                data[i][key] = np.vstack((data[i][key], array))
        for key in alias_T.keys():
            print("loading {}.\n\tmultipling by a^0 h^-0 to get physical value".format(key))
                # load particle mass from particle data
            array = np.array(f["{}".format(alias_T[key])])
            data[i][key] = np.append(data[i][key], array)

        for key in alias_mass.keys():
            print(
                "loading {}.\n\tmultipling by a^0 h^-1 to get physical value".format(key))
            if MassTable[alias_mass[key]] == 0:
                # load particle mass from particle data
                array = np.array(f["PartType{}/Masses".format(alias_mass[key])])
                array *= h**-1
                data[i][key] = np.append(data[i][key], array)
            else:
                array = np.full(
                    NumPart_ThisFile[alias_mass[key]], MassTable[alias_mass[key]])
                data[i][key] = np.append(data[i][key], array)

        #f.close()
        
   
# computing radial profiles
# log radius
for i in range(len(snapfile)):
    r2 \
        = (data[i]["sphpos"].T[0] )**2 \
        + (data[i]["sphpos"].T[1] )**2\
        + (data[i]["sphpos"].T[2] )**2
    data[i]["sphlogr"] = np.log10(np.sqrt(r2))
    data[i]["sphX"] = data[i]["sphpos"].T[0] 
    data[i]["sphY"] = data[i]["sphpos"].T[1] 
    data[i]["sphZ"] = data[i]["sphpos"].T[2] 
    r2 \
        = (data[i]["dmpos"].T[0])**2 \
        + (data[i]["dmpos"].T[1] )**2\
        + (data[i]["dmpos"].T[2] )**2
    data[i]["dmlogr"] = np.log10(np.sqrt(r2))
    r2 \
        = (data[i]["starpos"].T[0])**2 \
        + (data[i]["starpos"].T[1] )**2\
        + (data[i]["starpos"].T[2] )**2
    data[i]["starlogr"] = np.log10(np.sqrt(r2))
    data[i]["starX"] = data[i]["starpos"].T[0] 
    data[i]["starY"] = data[i]["starpos"].T[1] 
    data[i]["starZ"] = data[i]["starpos"].T[2] 


# initialize profile dictionary
profile = [{} for i in range(len(snapfile))]
# density profile
for i in range(len(snapfile)):
    # mass in shell
    mshell = np.histogram(data[i]["sphlogr"], bins=Xbins,
                          weights=1e10 * data[i]["sphmass"])[0]
    # volume of shell
    vshell = 4/3 * np.pi * (np.power(10, Xbins))**3
    vshell = (np.roll(vshell, -1) - vshell)[0:-1]
    profile[i]["sphrho"] = mshell / vshell
    # mass in shell
    mshell = np.histogram(data[i]["dmlogr"], bins=Xbins,
                          weights=1e10 * data[i]["dmmass"])[0]
    # volume of shell
    vshell = 4/3 * np.pi * (np.power(10, Xbins))**3
    vshell = (np.roll(vshell, -1) - vshell)[0:-1]
    profile[i]["dmrho"] = mshell / vshell
# temperature profile
# conversion factor from u[km^2/s^2] to T[K]
# T[K] = (gamma - 1) * mu * mp / kB * 1e10 * u[km^2/s^2]
u2t = 2/3*0.588*1.67e-24/1.38e-16 * 1e10
#for i in range(len(snapfile)):
    #NT = np.histogram(data[i]["sphlogr"], bins=Xbins,
                      #weights=data[i]["u"] * u2t)[0]
    #N = np.histogram(data[i]["sphlogr"], bins=Xbins)[0]
    #profile[i]["temperature"] = NT / N

for i in range(len(snapfile)):
    NT = np.histogram(data[i]["sphlogr"], bins=Xbins,
                      weights=data[i]["T"])[0]
    N = np.histogram(data[i]["sphlogr"], bins=Xbins)[0]
    profile[i]["temperature"] = NT / N
# pressure profile
# P[Msun km^2/s^2 / Mpc^3] = (gamma - 1) * 1e10 * rho[10^10 Msun/Mpc^3] * u[km^2/s^2]
for i in range(len(snapfile)):
    NP = np.histogram(data[i]["sphlogr"], bins=Xbins,
                      weights=2/3 * 1e10 * data[i]["rho"] * data[i]["u"])[0]
    N = np.histogram(data[i]["sphlogr"], bins=Xbins)[0]
    profile[i]["pressure"] = NP / N
# entropy profile
# S = (gamma - 1) * rho^(1 - gamma) * u
for i in range(len(snapfile)):
    NS = np.histogram(data[i]["sphlogr"], bins=Xbins,
                      weights=2/3 * data[i]["rho"]**(-2/3) * data[i]["u"])[0]
    N = np.histogram(data[i]["sphlogr"], bins=Xbins)[0]
    profile[i]["entropy"] = NS / N
#ionization fraction profile
for i in range(len(snapfile)):
    NII = np.histogram(data[i]["sphlogr"], bins=Xbins,
                      weights= data[i]["n_HII"])[0]
    NI = np.histogram(data[i]["sphlogr"], bins=Xbins,
                      weights=data[i]["n_HI"])[0]
    profile[i]["ionization"] = NII / (NI+NII)  # %%
# plot density profile
plt.figure(facecolor="white")
cmap = plt.get_cmap("tab10")
for i in range(len(snapfile)):
    plt.plot(Xpoints, profile[i]["sphrho"], color=cmap(
        i), label="{}, gas".format(snapfile[i][0]))
    plt.plot(Xpoints, profile[i]["dmrho"], linestyle="dashed", color=cmap(
        i), label="{}, dark matter".format(snapfile[i][0]))
plt.yscale("log")
plt.xlabel(r"log$_{10}$ r [Mpc]")
plt.ylabel(r"$\rho$ [M$_\odot$ Mpc$^{-3}$]")
plt.legend()
plt.savefig("density.png", bbox_inches="tight")
plt.close()
# plot temperature profile
plt.figure(facecolor="white")
cmap = plt.get_cmap("tab10")
for i in range(len(snapfile)):
    plt.plot(Xpoints, profile[i]["temperature"],
             color=cmap(i), label="{}".format(snapfile[i][0]))
plt.yscale("log")
plt.xlabel(r"log$_{10}$ r [Mpc]")
plt.ylabel(r"T [K]")
plt.legend()
plt.savefig("temperature.png", bbox_inches="tight")
plt.close()
# plot pressure profile
plt.figure(facecolor="white")
cmap = plt.get_cmap("tab10")
for i in range(len(snapfile)):
    plt.plot(Xpoints, profile[i]["pressure"], color=cmap(
        i), label="{}".format(snapfile[i][0]))
plt.yscale("log")
plt.xlabel(r"log$_{10}$ r [Mpc]")
plt.ylabel(r"P [M$_\odot$ km$^2$ s$^{-2}$ Mpc$^{-3}$]")
plt.legend()
plt.savefig("pressure.png", bbox_inches="tight")
plt.close()
# plot entropy profile
plt.figure(facecolor="white")
cmap = plt.get_cmap("tab10")
for i in range(len(snapfile)):
    plt.plot(Xpoints, profile[i]["entropy"], color=cmap(
        i), label="{}".format(snapfile[i][0]))
plt.yscale("log")
plt.xlabel(r"log$_{10}$ r [Mpc]")
plt.ylabel(r"S [(10$^{10}$ M$_\odot$ Mpc$^{-3}$)$^{-2/3}$ km$^2$ s$^{-2}$]")
plt.legend()
plt.savefig("entropy.png", bbox_inches="tight")
plt.close()
# plot temperature profile
plt.figure(facecolor="white")
cmap = plt.get_cmap("tab10")
for i in range(len(snapfile)):
    plt.plot(profile[i]["temperature"], profile[i]["sphrho"], color=cmap(
        i), label="{}".format(snapfile[i][0]))
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"T [K]")
plt.ylabel(r"$\rho$ [M$_\odot$ Mpc$^{-3}$]")
plt.legend()
plt.savefig("temperature_density.png", bbox_inches="tight")
plt.close()
# plot ionization fraction profile
plt.figure(facecolor="white")
cmap = plt.get_cmap("tab10")
for i in range(len(snapfile)):
    plt.plot(Xpoints, profile[i]["ionization"], color=cmap(
        i), label="{}".format(snapfile[i][0]))
plt.yscale("log")
plt.xlabel(r"log$_{10}$ r [Mpc]")
plt.ylabel(r"N_{HII}/(N_{HI}+N_{HII)")
plt.legend()
plt.savefig("temperature_density.png", bbox_inches="tight")
plt.close()
#make Star coordinate bin
sphX = data[0]["sphX"]*h
sphY = data[0]["sphX"]*h
sphZ = data[0]["sphZ"]*h
sphX_min = np.min(data[0]["sphX"])
sphX_max = np.max(data[0]["sphX"])
sphZ_min = np.min(data[0]["sphZ"])
sphZ_max = np.max(data[0]["sphZ"])
Bin_num = 50
sphX_bins = np.linspace(sphX_min, sphX_max, Bin_num)
sphZ_bins = np.linspace(sphZ_min, sphZ_max, Bin_num)

#made Star coordinate bin
starX = data[i]["starX"]
starZ = data[i]["starZ"]
starX_min = np.min(data[0]["starX"])
starX_max = np.max(data[0]["starX"])
starZ_min = np.min(data[0]["starZ"])
starZ_max = np.max(data[0]["starZ"])

starX_bins = np.linspace(starX_min, starX_max, Bin_num)
starZ_bins = np.linspace(starZ_min, starZ_max, Bin_num)

SPHMasses, xedges, zedges = histogram2d(sphX, sphZ, bins = [sphX_bins, sphZ_bins], weights = (data[0]["sphmass"]))
X, Z = np.meshgrid(xedges, zedges)
hh = plt.pcolormesh(X, Z, SPHMasses, cmap='rainbow')
plt.xlabel(r"X [Mpc]")
plt.ylabel(r"Y [Mpc]")
cb = plt.colorbar(hh)
cb.set_label('SPH Mass [M$\odot$]')  #设置colorbar的标签字体
plt.savefig("SPHMass-Cordinates_20.png", bbox_inches="tight")
plt.close()

StarMasses, xedges, zedges = histogram2d(starX, starZ, bins = [starX_bins, starZ_bins], weights = (data[0]["starmass"]))

X, Z = np.meshgrid(xedges, zedges)
plt.pcolormesh(X, Z, StarMasses, cmap='rainbow')
plt.xlabel(r"X [Mpc]")
plt.ylabel(r"Y [Mpc]")
hh = plt.pcolormesh(X, Z, StarMasses, cmap='rainbow')
cb = plt.colorbar(hh)
cb.set_label('Star Mass [M$\odot$]')#设置colorbar的标签字体
plt.savefig("StarMass-Cordinates_20.png", bbox_inches="tight")
plt.close()

#Temperature Bin


#############################3D+2D mass_distribution################
me = 9.1*10**-28
mH = 1.67*10**-24
solartog = 2.0*10**33
def M6(r,r0, h, D):
    import numpy as np
    from numpy import pi, sqrt, exp
    ##########D = dimension#######
    ##########h = smoothing length########
    d = np.sqrt((r-r0)**2)
    q = d/h
    V = [2.0, 2*pi*d, 4*pi*d**2]
    sigma = np.array([1.0/120*3, 7.0/(478.0*np.pi)*3**2, 1.0/(120*np.pi)*3**3])
    if q >= 0 and q < 1.0/3:
        w = (3.0)**5*(1.0 - q)**5 - 6*(3.0)**5*(2.0/3-q)**5+15*(3)**5*(1.0/3-q)**5
    elif q >= 1.0/3 and q < 2.0/3:
        w = (3.0)**5*(1.0 - q)**5 - 6*(3.0)**5*(2.0/3-q)**5
    elif q >= 2.0/3 and q < 1.0:
        w = (3.0)**5*(1.0 - q)**5
    else:
        w = 0
    W =  h**(-D)*sigma[D-1]*w
    return W


def Gauss(r, r0, h, D):
    from numpy import pi, sqrt, exp
    d = sqrt((r-r0)**2)
    V = [2.0, 2*pi*d, 4*pi*d**2]
    sigma = [1/sqrt(pi), 1/pi, 1.0/(pi*sqrt(pi))]
    W = h**(-D)*sigma[D-1]*exp(-(r-r0)**2/h**2)
    W = W*V[D-1]
    return W
MpcTocm = 3.08567758*10**24
sphX = data[0]["sphX"]*h
sphY = data[0]["sphX"]*h
sphZ = data[0]["sphZ"]*h
F_e = data[0]["n_e"]
F_HI = data[0]["n_HI"]
F_HII = data[0]["n_HII"]
F_HeI = data[0]["n_HeI"]
F_HeII = data[0]["n_HeII"]
F_HeIII = data[0]["n_HeIII"]



#sphX_min = np.min(sphX)
#sphX_max = np.max(sphX)
#sphY_min = np.min(sphY)
#sphY_max = np.max(sphY)
#sphZ_min = np.min(sphZ)
#sphZ_max = np.max(sphZ)

#sphX_bins = np.linspace(sphX_min, sphX_max, Bin_num)
#sphY_bins = np.linspace(sphY_min, sphY_max, Bin_num)
#sphZ_bins = np.linspace(sphZ_min, sphZ_max, Bin_num)
sphX_bins = np.linspace(0, Boxsize, Bin_num)
sphY_bins = np.linspace(0, Boxsize, Bin_num)
sphZ_bins = np.linspace(0, Boxsize, Bin_num)
sphpos = data[0]["sphpos"]*h
Smoothlen = data[0]['smoothlen']
bin_size = Boxsize / Bin_num
dx = sphX_bins[1] - sphX_bins[0]
dy = sphY_bins[1] - sphY_bins[0]
dz = sphZ_bins[1] - sphZ_bins[0]
diag = np.sqrt(2)/2*dx
for i in range(len(Smoothlen)):
    Smoothlen[i] = max(Smoothlen[i], diag)   
#h_L_max = np.ceil(np.max(data[0]['smoothlen']/diag)) ######向上取整
#gird_num = h_L_max + 1
#dS = dx*dz
#dV = dx*dy*dz
dS = bin_size**2
dV = (bin_size*MpcTocm)**3
sphX_bin_mid = (sphX_bins + np.roll(sphX_bins, -1))[0:-1] / 2      ###
sphY_bin_mid = (sphY_bins + np.roll(sphY_bins, -1))[0:-1] / 2   
sphZ_bin_mid = (sphZ_bins + np.roll(sphZ_bins, -1))[0:-1] / 2
sph_pos_mid = [sphX_bin_mid[0:-1], sphZ_bin_mid[0:-1]]

SPHmass_o = data[0]["sphmass"] #################The SPHmass not considering the Smoothing length###########
SPHmass_cgs = SPHmass_o*SPHmasstocgs
SPHrho = data[0]["rho"]*SPHdensitytocgs###density
#SPHmasses0 = np.zeros(len(sphZ_bin_mid) - 1)###########Non-normalization
#SPHmasses = np.zeros(len(sphZ_bin_mid) - 1)###########Normalization
SPHrho_column0 =  np.zeros([len(sphZ_bin_mid), len(sphZ_bin_mid) ])###########Non-normalization
SPHrho_column =  np.zeros([len(sphZ_bin_mid), len(sphZ_bin_mid) ])###########Normalization
#d = np.zeros(len(sphZ_bin_mid))
D = 3###################Dimension
#for i in range(len(SPHmass_o)):
    #for j in range(len(SPHmasses)):
        #d0 = np.sqrt((sphZ[i] - sphZ_bin_mid[j])**2 + (sphX[i] - sphX_bin_mid[j])**2) ##########line distance
        #k = (sphZ[i] - sphZ_bin_mid[j])/(sphX[i] - sphX_bin_mid[j])     ########slope
        #if k <= 0.5:
            #z0 = sphZ[i] - k*(sphX[i] - sphX_min)
            #ze = sphZ[i] - k*(sphX[i] - sphX_max)
            #d_tot = np.sqrt((ze - z0)**2+ (sphX_max - sphX_min)**2)
        #if k > 0.5:
            #x0 = sphX[i] - (sphZ[i] - sphZ_min)/k
            #xe = sphX[i] - (sphZ[i] - sphZ_max)/k
            #d_tot = np.sqrt((xe - x0)**2+ (sphZ_max - sphZ_min)**2)
        #if d_tot > 0.5*d0:
            #d[j] = d0
        #else:
            #d[j] = d_tot - d0             #############distance across the bondary
        #SPHmasses0[j] = SPHmass_o[i]*M6(d[j],0,Smoothlen[i], D)*dS
    #for j in range(len(SPHmasses)):
        #SPHmasses[j] = SPHmass_o[i]*M6(d[j],0,Smoothlen[i], D)*dS/sum(SPHmasses0)
    #if i%100000 == 0:
        #print("it's  the %d step"  %i )
Ne = SPHmass_cgs*F_e/mH
NHI = SPHmass_cgs*F_HI/mH
NHII = SPHmass_cgs*F_HII/mH
NHeI = SPHmass_cgs*F_HeI/(4*mH)
NHeII = SPHmass_cgs*F_HeII/(4*mH)
NHeIII = SPHmass_cgs*F_HeIII/(4*mH)
ne = SPHrho**F_e/mH
nHI = SPHrho*F_HI/mH
nHII = SPHrho*F_HII/mH
nHeI = SPHrho*F_HeI/(2*mH)
nHeII = SPHrho*F_HeII/(2*mH)
nHeIII = SPHrho*F_HeIII/(2*mH)
mass_sum = np.zeros((Bin_num, Bin_num, Bin_num))
Ne_sum = np.zeros((Bin_num, Bin_num, Bin_num))
NHI_sum = np.zeros((Bin_num, Bin_num, Bin_num))
NHII_sum = np.zeros((Bin_num, Bin_num, Bin_num))
NHeI_sum = np.zeros((Bin_num, Bin_num, Bin_num))
NHeII_sum = np.zeros((Bin_num, Bin_num, Bin_num))
NHeIII_sum = np.zeros((Bin_num, Bin_num, Bin_num))
rho_sum = np.zeros((Bin_num, Bin_num, Bin_num))
ne_sum = np.zeros((Bin_num, Bin_num, Bin_num))
nHI_sum = np.zeros((Bin_num, Bin_num, Bin_num))
nHII_sum = np.zeros((Bin_num, Bin_num, Bin_num))
nHeI_sum = np.zeros((Bin_num, Bin_num, Bin_num))
nHeII_sum = np.zeros((Bin_num, Bin_num, Bin_num))
nHeIII_sum = np.zeros((Bin_num, Bin_num, Bin_num))

mass_sum0 = np.zeros((Bin_num, Bin_num, Bin_num))
NHI_sum0 = np.zeros((Bin_num, Bin_num, Bin_num))
NHII_sum0 = np.zeros((Bin_num, Bin_num, Bin_num))
NHeI_sum0 = np.zeros((Bin_num, Bin_num, Bin_num))
NHeII_sum0 = np.zeros((Bin_num, Bin_num, Bin_num))
NHeIII_sum0 = np.zeros((Bin_num, Bin_num, Bin_num))
rho_sum0 = np.zeros((Bin_num, Bin_num, Bin_num))
nHI_sum0 = np.zeros((Bin_num, Bin_num, Bin_num))
nHII_sum0 = np.zeros((Bin_num, Bin_num, Bin_num))
nHeI_sum0 = np.zeros((Bin_num, Bin_num, Bin_num))
nHeII_sum0 = np.zeros((Bin_num, Bin_num, Bin_num))
nHeIII_sum0 = np.zeros((Bin_num, Bin_num, Bin_num))
tot_mm = 0

start_time = time.time()  # 记录程序开始时间
# 程序代码
# ...
for hh in range(len(SPHmass_o)):
    total_mass = 0
    total_mass_new = 0
    total_mass_3D = 0
    total_mass_3D_new = 0
    total_Ne = 0
    total_NHI = 0
    total_NHII = 0
    total_NHeI = 0
    total_NHeII = 0
    total_NHeIII = 0
    total_Ne_new = 0
    total_NHI_new = 0
    total_NHII_new = 0
    total_NHeI_new = 0
    total_NHeII_new = 0
    total_NHeIII_new = 0
    total_mass2_3D = 0
    total_Ne2 = 0
    total_NHI2 = 0
    total_NHII2 = 0
    total_NHeI2 = 0
    total_NHeII2 = 0
    total_NHeIII2 = 0
    total_mass2_3D_new = 0
    total_Ne2_new = 0
    total_NHI2_new = 0
    total_NHII2_new = 0
    total_NHeI2_new = 0
    total_NHeII2_new = 0
    total_NHeIII2_new = 0
    p_x = bisect.bisect(sphX_bins, sphX[hh])-1
    p_z = bisect.bisect(sphZ_bins, sphZ[hh])-1
    Smoothlen[hh] = max(diag, Smoothlen[hh])
    si = np.ceil(Smoothlen[hh]/diag)
    if si%2 == 0:          ##########全部取奇数格子
        grid_num = (si + 1)**2
        Mi = si + 1
    if si%2 == 1:
        grid_num = (si + 2)**2
        Mi = si + 2
    lb = int((Mi- 1)/2)        ##########边界距离中心的格子数量
    X_index = np.linspace(p_x-lb,p_x+lb,2*lb+1)
    Z_index = np.linspace(p_z-lb,p_z+lb,2*lb+1)
    X_index = X_index.astype(int)
    Z_index = Z_index.astype(int)
    box_len = 2*lb+1
    grid_xb_max = Bin_num-1
    grid_xb_min = 0
    grid_zb_max = Bin_num-1
    grid_zb_min = 0
    d = np.zeros([len(X_index), len(Z_index)])
    if (p_x <=lb) or (p_x >= grid_xb_max - lb) or (p_z <= lb) or (p_z >= grid_zb_max - lb): ########边界处的格子
        for i in range(box_len):
            if X_index[i]< 0:
                X_index[i] = grid_xb_max+X_index[i]
            if Z_index[i]< 0:
                Z_index[i] = grid_zb_max+Z_index[i]
            if X_index[i]>= grid_xb_max:
                X_index[i] = X_index[i] - grid_xb_max
            if Z_index[i]>= grid_zb_max:
                Z_index[i] = Z_index[i] - grid_zb_max
        for i in range(len(X_index)):
            for j in range(len(Z_index)):
                d0 = np.sqrt((sphZ[hh] - sphZ_bin_mid[Z_index[j]])**2 + (sphX[hh] - sphX_bin_mid[X_index[i]])**2) ##########line distance
                if sphX[hh] - sphX_bin_mid[X_index[i]] == 0: ###########垂直于网格的连线
                    d_tot = sphZ_max - sphZ_min
                else: 
                    k = abs((sphZ[hh] - sphZ_bin_mid[Z_index[j]])/(sphX[hh] - sphX_bin_mid[X_index[i]]))   ########slope
                    if k <= 1:
                        theta = atan(k)
                        d_tot = (sphX_max - sphX_min)/cos(theta)
                    if k > 1:
                        theta = atan(k)
                        d_tot = (sphZ_max - sphZ_min)/sin(theta)
                if d0<0.5*d_tot:
                    d[i][j] = d0
                else:
                    d[i][j] = d_tot - d0             #############distance across the bondary
                SPHrho_column0[X_index[i]][Z_index[j]] += SPHmass_o[hh]/Smoothlen[hh]**2*F_xy(d[i][j]/Smoothlen[hh])
                total_mass += SPHmass_o[hh]/Smoothlen[hh]**2*F_xy(d[i][j]/Smoothlen[hh])*dS
    else:
        for i in range(len(X_index)):
            for j in range(len(Z_index)):
                d0 = np.sqrt((sphZ[hh] - sphZ_bin_mid[Z_index[j]])**2 + (sphX[hh] - sphX_bin_mid[X_index[i]])**2) ##########line distance
                d[i][j] =  d0
                SPHrho_column0[X_index[i]][Z_index[j]] += SPHmass_o[hh]/Smoothlen[hh]**2*F_xy(d[i][j]/Smoothlen[hh])
                total_mass += SPHmass_o[hh]/Smoothlen[hh]**2*F_xy(d[i][j]/Smoothlen[hh])*dS
    for i in range(len(X_index)):
        for j in range(len(Z_index)):
            SPHrho_column[X_index[i]][Z_index[j]] +=  SPHmass_o[hh]/Smoothlen[hh]**2*F_xy(d[i][j]/Smoothlen[hh])*SPHmass_o[hh]/total_mass
            total_mass_new +=  SPHmass_o[hh]/Smoothlen[hh]**2*F_xy(d[i][j]/Smoothlen[hh])*SPHmass_o[hh]/total_mass*dS
            #print(SPHrho_column[X_index[i]][Z_index[j]]*dS)
            #print(total_mass_new)
    tot_mm += total_mass_new
    ###########################3D kernel function######
            # 如果平滑长度超过了BOX的边界，则以BOX的边界为限制
    Smoothlen[hh] = max(Smoothlen[hh], bin_size * np.sqrt(3) / 2)
        #平滑长度至少为3D网格对角线的一半
         # 找到临近的网格，注意周期边界条件
    bin_idx = sphpos[hh] // bin_size  # 粒子所在网格的索引
    bin_idx_min = np.floor((sphpos[hh] - Smoothlen[hh]) / bin_size)
    bin_idx_max = np.ceil((sphpos[hh] + Smoothlen[hh]) / bin_size)
    for ix in range(int(bin_idx_min[0]), int(bin_idx_max[0])):
        for iy in range(int(bin_idx_min[1]), int(bin_idx_max[1])):
            for iz in range(int(bin_idx_min[2]), int(bin_idx_max[2])):
            # 计算网格中心到粒子的距离，并根据平滑长度计算kernel函数值
                bin_center = (np.array([ix, iy, iz]) + 0.5) * bin_size
                dist = np.linalg.norm(bin_center - sphpos[hh])
                weight =  M6(dist*MpcTocm,0, Smoothlen[hh]*MpcTocm,3)
                # 处理周期边界条件
                bin_center[0] = bin_center[0] % Boxsize
                bin_center[1] = bin_center[1] % Boxsize
                bin_center[2] = bin_center[2] % Boxsize
                # 将粒子的质量、H离子数密度、He离子数密度分配到网格中
                mass_weight = SPHmass_cgs[hh]*SPHmass_cgs[hh]/SPHrho[hh] * weight
                Ne_weight = SPHmass_cgs[hh]/SPHrho[hh]*Ne[hh] * weight
                NHI_weight = SPHmass_cgs[hh]/SPHrho[hh]*NHI[hh] * weight
                NHII_weight = SPHmass_cgs[hh]/SPHrho[hh]*NHII[hh] * weight
                NHeI_weight = SPHmass_cgs[hh]/SPHrho[hh]*NHeI[hh] * weight
                NHeII_weight = SPHmass_cgs[hh]/SPHrho[hh]*NHeII[hh] * weight
                NHeIII_weight = SPHmass_cgs[hh]/SPHrho[hh]*NHeIII[hh] * weight
                rho_weight = SPHmass_cgs[hh] * weight
                ne_weight = SPHmass_cgs[hh]/SPHrho[hh]*ne[hh] * weight
                nHI_weight = SPHmass_cgs[hh]/SPHrho[hh]*nHI[hh] * weight
                nHII_weight = SPHmass_cgs[hh]/SPHrho[hh]*nHII[hh] * weight
                nHeI_weight = SPHmass_cgs[hh]/SPHrho[hh]*nHeI[hh] * weight
                nHeII_weight = SPHmass_cgs[hh]/SPHrho[hh]*nHeII[hh] * weight
                nHeIII_weight = SPHmass_cgs[hh]/SPHrho[hh]*nHeIII[hh] * weight
                #print(rho_weight )
                # 处理周期边界条件
                if ix < 0:
                    ix1 = ix + Bin_num 
                elif ix >= Bin_num:
                    ix1 = ix - Bin_num
                else:
                    ix1 = ix
                if iy < 0:
                    iy1 = iy + Bin_num 
                elif iy >= Bin_num:
                    iy1 = iy - Bin_num 
                else:
                    iy1 = iy 
                if iz < 0:
                    iz1 = iz + Bin_num 
                elif iz >= Bin_num:
                    iz1 = iz - Bin_num 
                else:
                    iz1 = iz
                # 更新网格中的质量、H离子数密度、He离子数密度
                mass_sum0[ix1, iy1, iz1] += mass_weight
                NHI_sum0[ix1, iy1, iz1] += NHI_weight
                NHII_sum0[ix1, iy1, iz1] += NHII_weight
                NHeI_sum0[ix1, iy1, iz1] += NHeI_weight
                NHeII_sum0[ix1, iy1, iz1] += NHeII_weight
                NHeIII_sum0[ix1, iy1, iz1] += NHeIII_weight
                rho_sum0[ix1, iy1, iz1] += rho_weight
                nHI_sum0[ix1, iy1, iz1] += nHI_weight
                nHII_sum0[ix1, iy1, iz1] += nHII_weight
                nHeI_sum0[ix1, iy1, iz1] += nHeI_weight
                nHeII_sum0[ix1, iy1, iz1] += nHeII_weight
                nHeIII_sum0[ix1, iy1, iz1] += nHeIII_weight
                ###############质量#################
                total_mass_3D += mass_weight
                total_Ne += Ne_weight
                total_NHI += NHI_weight
                total_NHII += NHII_weight
                total_NHeI += NHeI_weight
                total_NHeII += NHeII_weight
                total_NHeIII += NHeIII_weight
                #############密度##############
                total_mass2_3D += rho_weight*dV
                total_Ne2 += ne_weight*dV
                total_NHI2 += nHI_weight*dV
                total_NHII2 += nHII_weight*dV
                total_NHeI2 += nHeI_weight*dV
                total_NHeII2 += nHeII_weight*dV
                total_NHeIII2 += nHeIII_weight*dV
    for ix in range(int(bin_idx_min[0]), int(bin_idx_max[0])):
        for iy in range(int(bin_idx_min[1]), int(bin_idx_max[1])):
            for iz in range(int(bin_idx_min[2]), int(bin_idx_max[2])):
            # 计算网格中心到粒子的距离，并根据平滑长度计算kernel函数值
                bin_center = (np.array([ix, iy, iz]) + 0.5) * bin_size
                dist = np.linalg.norm(bin_center - sphpos[hh])
                weight =  M6(dist*MpcTocm,0, Smoothlen[hh]*MpcTocm,3)
                # 处理周期边界条件
                bin_center[0] = bin_center[0] % Boxsize
                bin_center[1] = bin_center[1] % Boxsize
                bin_center[2] = bin_center[2] % Boxsize
                # 将粒子的质量、H离子数密度、He离子数密度分配到网格中
                mass_weight = SPHmass_cgs[hh]*SPHmass_cgs[hh]/SPHrho[hh] * weight
                NHI_weight = SPHmass_cgs[hh]/SPHrho[hh]*NHI[hh] * weight
                NHII_weight = SPHmass_cgs[hh]/SPHrho[hh]*NHII[hh] * weight
                NHeI_weight = SPHmass_cgs[hh]/SPHrho[hh]*NHeI[hh] * weight
                NHeII_weight = SPHmass_cgs[hh]/SPHrho[hh]*NHeII[hh] * weight
                NHeIII_weight = SPHmass_cgs[hh]/SPHrho[hh]*NHeIII[hh] * weight
                rho_weight = SPHmass_cgs[hh] * weight
                nHI_weight = SPHmass_cgs[hh]/SPHrho[hh]*nHI[hh] * weight
                nHII_weight = SPHmass_cgs[hh]/SPHrho[hh]*nHII[hh] * weight
                nHeI_weight = SPHmass_cgs[hh]/SPHrho[hh]*nHeI[hh] * weight
                nHeII_weight = SPHmass_cgs[hh]/SPHrho[hh]*nHeII[hh] * weight
                nHeIII_weight = SPHmass_cgs[hh]/SPHrho[hh]*nHeIII[hh] * weight
                # 处理周期边界条件
                if ix < 0:
                    ix1 = ix + Bin_num 
                elif ix >= Bin_num:
                    ix1 = ix - Bin_num
                else:
                    ix1 = ix
                if iy < 0:
                    iy1 = iy + Bin_num 
                elif iy >= Bin_num:
                    iy1 = iy - Bin_num 
                else:
                    iy1 = iy 
                if iz < 0:
                    iz1 = iz + Bin_num 
                elif iz >= Bin_num:
                    iz1 = iz - Bin_num 
                else:
                    iz1 = iz
                # 根据质量守恒和粒子数守恒更新网格中的质量、H离子数密度、He离子数密度
                if total_mass_3D == 0:
                    print('tot =', total_mass_3D)
                    break
                if total_mass_3D_new == float('inf'):
                    print('total_m=', total_mass_3D)
                    print(hh)
                    print(ix1,iy1,iz1)
                    print(bin_idx_min,   bin_idx_max)
                    print('NH = ', NHI[hh])
                    print('total_mass_3D = ', total_mass_3D)
                    print('total_m_new=', total_mass_3D_new)
                    break
                mass_sum[ix1, iy1, iz1] += SPHmass_cgs[hh]*mass_weight/total_mass_3D
                Ne_sum[ix1, iy1, iz1] += Ne[hh]*Ne_weight/total_Ne
                NHI_sum[ix1, iy1, iz1] += NHI[hh]*NHI_weight/total_NHI
                NHII_sum[ix1, iy1, iz1] +=  NHII[hh]*NHII_weight/total_NHII
                NHeI_sum[ix1, iy1, iz1] += NHeI[hh]*NHeI_weight/total_NHeI
                NHeII_sum[ix1, iy1, iz1] += NHeII[hh]*NHeII_weight/total_NHeII
                NHeIII_sum[ix1, iy1, iz1] += NHeIII[hh]*NHeIII_weight/total_NHeIII
                rho_sum[ix1, iy1, iz1] += SPHmass_cgs[hh]*rho_weight/total_mass2_3D
                ne_sum[ix1, iy1, iz1] += ne[hh]*ne_weight/total_Ne2
                nHI_sum[ix1, iy1, iz1] += NHI[hh]*nHI_weight/total_NHI2
                nHII_sum[ix1, iy1, iz1] += NHII[hh]*nHII_weight/total_NHII2
                nHeI_sum[ix1, iy1, iz1] += NHeI[hh]*nHeI_weight/total_NHeI2
                nHeII_sum[ix1, iy1, iz1] += NHeII[hh]*nHeII_weight/total_NHeII2
                nHeIII_sum[ix1, iy1, iz1] += NHeIII[hh]*nHeIII_weight/total_NHeIII2
                total_mass_3D_new += SPHmass_cgs[hh]*mass_weight/total_mass_3D
                total_NHI_new += NHI[hh]*NHI_weight/total_NHI
                total_NHII_new += NHII[hh]*NHII_weight/total_NHII
                total_NHeI_new += NHeI[hh]*NHeI_weight/total_NHeI
                total_NHeII_new +=   NHeII[hh]*NHeII_weight/total_NHeII
                total_NHeIII_new +=  NHeIII[hh]*NHeIII_weight/total_NHeIII
                total_mass2_3D_new += SPHmass_cgs[hh]*rho_weight/total_mass2_3D*dV
                total_NHI2_new += NHI[hh]*NHI_weight/total_NHI
                total_NHII2_new += NHII[hh]*NHII_weight/total_NHII
                total_NHeI2_new += NHeI[hh]*NHeI_weight/total_NHeI
                total_NHeII2_new +=   NHeII[hh]*NHeII_weight/total_NHeII
                total_NHeIII2_new +=  NHeIII[hh]*NHeIII_weight/total_NHeIII
    if total_mass_3D_new == float('inf'):
        print('total_m=', total_mass_3D)
        print(hh)
        print(ix,iy,iz)
        print(bin_idx_min,   bin_idx_max)
        print('NH = ', NHI[hh])
        print('total_m_new=', total_mass_3D_new)
        break
    #print('total_m=', total_mass_3D)
    #print('total_m_new=', total_mass_3D_new)
    #print('total_m2=', total_mass2_3D)
    #print("total_m2_new = ", total_mass2_3D_new)
    #print("mass_sum= ", mass_sum[ix, iy, iz])
    if hh%10000 == 0:
        print("it's  the %d step"  %hh )
        end_time = time.time()  # 记录程序结束时间
        run_time = end_time - start_time  # 计算程序运行时间，单位为秒
        print(f"程序已运行时间为：{run_time:.2f}秒")
        print(f"程序剩余时间约为：{((NumPart_Total[0]/(hh+1e-30)*run_time-run_time)/3600):.2f}小时")
        print("total_mass = ", total_mass_3D)
        sys.stdout.flush()
        #print("px = %f" %p_x)
        #print("pz = %f"  %p_z)
        #print("Smoothlength is %f" %Smoothlen[h])
        #print("X_index = ", X_index)
        #print("Z_index = ", Z_index)
        #print('Z_position', sphZ[h])
        #print('X_position', sphX[h])
        if Smoothlen[hh]/diag >= 0:
            print('distance = ', d)
            #print('total_distance = %f' %d_tot)
            print('total_mass_1 and initial mass =',total_mass_new, SPHmass_o[hh])
            #for ix in range(int(bin_idx_min[0]), int(bin_idx_max[0])):
                #for iy in range(int(bin_idx_min[1]), int(bin_idx_max[1])):
                    #for iz in range(int(bin_idx_min[2]), int(bin_idx_max[2])):
                        #print("mass, rho*V=", mass_sum[ix, iy, iz], rho_sum[ix,iy,iz]*dV)
            #print(ix, iy, iz)
            print("total_mass_3D_C,total_mass_NC and initial mass  = ", total_mass_3D_new, total_mass_3D, SPHmass_o[hh]*SPHmasstocgs)
            print("total_mass2_3D_C,total_mass2_NC and initial mass  = ", total_mass2_3D_new, total_mass2_3D, SPHmass_o[hh]*SPHmasstocgs)
            print("mass_ix1,iy1,iz1 = ", mass_sum[ix1, iy1, iz1], rho_sum[ix1, iy1, iz1]*dV)


        

print("tot_m = %f" %tot_mm)
X, Z = np.meshgrid(sphX_bins, sphZ_bins)
#SPHrho_column = SPHrho_column.T
hh = plt.pcolormesh(X, Z, SPHrho_column, cmap='rainbow')
plt.xlabel(r"X [Mpc]")
plt.ylabel(r"Y [Mpc]")
cb = plt.colorbar(hh)
cb.set_label('Column Density [M$\odot$ cm$^{-2}$ ]')  #设置colorbar的标签字体
plt.savefig("SPHrho-Cordinates_20_1001_50bin.png", bbox_inches="tight", dpi = 500)
plt.close()

log_SPHrho_S = np.log10(SPHrho_column/dS)
hh = plt.pcolormesh(X, Z, log_SPHrho_S, cmap='rainbow')
plt.xlabel(r"X [Mpc]")
plt.ylabel(r"Y [Mpc]")
cb = plt.colorbar(hh)
cb.set_label('Column Density [log (M$\odot$ Mpc$^{-2}$) ]')  #设置colorbar的标签字体
plt.savefig("SPHrho-Cordinates_log_20_1001_50bin.png", bbox_inches="tight", dpi = 500)
#plt.show()
plt.close()

filename1 = "column_denstiy1_50bin.hdf5"
if os.path.exists(filename1):
    os.remove(filename1)
f = h5py.File(filename1, "w")
# 在文件中创建数据集、组等等
f.create_dataset('X', (len(X),len(X)), data = X)
f.create_dataset('Z', (len(Z),len(Z)), data = Z)
f.create_dataset('SPHrho_S', (len(SPHrho_column), len(SPHrho_column)), data = SPHrho_column)
f.close()


bin_mass = np.zeros((Bin_num, Bin_num, Bin_num))


bin_size = Boxsize / Bin_num
# 计算每个bin的颜色
colors = np.log10(mass_sum.flatten() + 1)  # 使用对数缩放
colors2 = np.log10(rho_sum.flatten() + 1)  # 使用对数缩放

# 绘制三维图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x_vals, y_vals, z_vals = np.meshgrid(
    np.linspace(0, Boxsize, Bin_num), np.linspace(0, Boxsize, Bin_num), np.linspace(0, Boxsize, Bin_num))
#ax.scatter(sphX, sphY, sphZ, c=SPHmass_o, cmap='plasma', alpha=0.1)  # 绘制粒子
ax.scatter(x_vals.flatten(), y_vals.flatten(), z_vals.flatten(), c=colors, cmap='plasma', s=10)  # 绘制每个bin
plt.savefig('./mass_50bin.png')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x_vals, y_vals, z_vals = np.meshgrid(
    np.linspace(0, Boxsize, Bin_num), np.linspace(0, Boxsize, Bin_num), np.linspace(0, Boxsize, Bin_num))
#ax.scatter(sphX, sphY, sphZ, c=SPHmass_o, cmap='plasma', alpha=0.1)  # 绘制粒子
ax.scatter(x_vals.flatten(), y_vals.flatten(), z_vals.flatten(), c=colors2, cmap='plasma', s=10)  # 绘制每个bin
plt.savefig('./rho_50bin.png')

filename2 = "3D_BOX_50bin1.hdf5"
if os.path.exists(filename2):
    os.remove(filename2)
f = h5py.File(filename2, "w")
f.create_dataset('X', (len(x_vals),len(x_vals),len(x_vals)), data = x_vals)
f.create_dataset('Y', (len(y_vals),len(y_vals),len(y_vals)), data = y_vals)
f.create_dataset('Z', (len(z_vals),len(z_vals),len(z_vals)), data = z_vals)
f.create_dataset('SPHMasses', (len(mass_sum), len(mass_sum), len(mass_sum)), data = mass_sum)
f.create_dataset('SPHrho', (len(mass_sum), len(mass_sum), len(mass_sum)), data = rho_sum)
f.create_dataset('Ne', (len(mass_sum), len(mass_sum), len(mass_sum)), data = Ne_sum)
f.create_dataset('NHI', (len(mass_sum), len(mass_sum), len(mass_sum)), data = NHI_sum)
f.create_dataset('nHI', (len(mass_sum), len(mass_sum), len(mass_sum)), data = nHI_sum)
f.create_dataset('NHII', (len(mass_sum), len(mass_sum), len(mass_sum)), data = NHII_sum)
f.create_dataset('nHII', (len(mass_sum), len(mass_sum), len(mass_sum)), data = nHII_sum)
f.create_dataset('NHeI', (len(mass_sum), len(mass_sum), len(mass_sum)), data = NHeI_sum)
f.create_dataset('ne', (len(mass_sum), len(mass_sum), len(mass_sum)), data = ne_sum)
f.create_dataset('nHeI', (len(mass_sum), len(mass_sum), len(mass_sum)), data = nHeI_sum)
f.create_dataset('NHeII', (len(mass_sum), len(mass_sum), len(mass_sum)), data = NHeII_sum)
f.create_dataset('nHeII', (len(mass_sum), len(mass_sum), len(mass_sum)), data = nHeII_sum)
f.create_dataset('NHeIII', (len(mass_sum), len(mass_sum), len(mass_sum)), data = NHeIII_sum)
f.create_dataset('nHeIII', (len(mass_sum), len(mass_sum), len(mass_sum)), data = nHeIII_sum)
#############################without normalization###############
f.create_dataset('SPHMasses0', (len(mass_sum), len(mass_sum), len(mass_sum)), data = mass_sum0)
f.create_dataset('SPHrho0', (len(mass_sum), len(mass_sum), len(mass_sum)), data = rho_sum0)
f.create_dataset('NHI0', (len(mass_sum), len(mass_sum), len(mass_sum)), data = NHI_sum0)
f.create_dataset('nHI0', (len(mass_sum), len(mass_sum), len(mass_sum)), data = nHI_sum0)
f.create_dataset('NHII0', (len(mass_sum), len(mass_sum), len(mass_sum)), data = NHII_sum0)
f.create_dataset('nHII0', (len(mass_sum), len(mass_sum), len(mass_sum)), data = nHII_sum0)
f.create_dataset('NHeI0', (len(mass_sum), len(mass_sum), len(mass_sum)), data = NHeI_sum0)
f.create_dataset('nHeI0', (len(mass_sum), len(mass_sum), len(mass_sum)), data = nHeI_sum0)
f.create_dataset('NHeII0', (len(mass_sum), len(mass_sum), len(mass_sum)), data = NHeII_sum0)
f.create_dataset('nHeII0', (len(mass_sum), len(mass_sum), len(mass_sum)), data = nHeII_sum0)
f.create_dataset('NHeIII0', (len(mass_sum), len(mass_sum), len(mass_sum)), data = NHeIII_sum0)
f.create_dataset('nHeIII0', (len(mass_sum), len(mass_sum), len(mass_sum)), data = nHeIII_sum0)
f.close()
print("Finished!")