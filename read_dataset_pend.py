import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

from matplotlib import pylab as plt
from scipy.special import ellipj, ellipk

import torch
import scipy.io 

#******************************************************************************
# Read in data
#******************************************************************************
def data_from_name(name, noise, theta, ntr, nval, nivs, nivs2, npreds):   
    if name == 'pendulum':
        return pendulum(noise, theta, ntr, nval, nivs, nivs2, npreds)
    if name == 'wavy_beam':
        return wavy_beam(noise, ntr, nval, nivs, nivs2, npreds)
    if name == 'fluid_flow':
        return fluid_flow(noise, ntr, nval, nivs, nivs2, npreds)        
    else:
        raise ValueError('dataset {} not recognized'.format(name))


def rescale(Xsmall, Xsmall_test):
    #******************************************************************************
    # Rescale data
    #******************************************************************************
    Xmin = Xsmall.min()
    Xmax = Xsmall.max()
    
    Xsmall = ((Xsmall - Xmin) / (Xmax - Xmin)) 
    Xsmall_test = ((Xsmall_test - Xmin) / (Xmax - Xmin)) 

    return Xsmall, Xsmall_test



def pendulum(noise_db, theta, ntr,  nval, nivs, nivs2, npreds):
    
    np.random.seed(1)

    def sol(t,theta0):
        S = np.sin(0.5*(theta0) )
        K_S = ellipk(S**2)
        omega_0 = np.sqrt(9.81)
        sn,cn,dn,ph = ellipj( K_S - omega_0*t, S**2 )
        theta = 2.0*np.arcsin( S*sn )
        d_sn_du = cn*dn
        d_sn_dt = -omega_0 * d_sn_du
        d_theta_dt = 2.0*S*d_sn_dt / np.sqrt(1.0-(S*sn)**2)
        return np.stack([theta, d_theta_dt],axis=1)
    
    
    anal_ts = np.arange(0, 2200*0.1, 0.1)
    X_org0 = sol(anal_ts, theta)
    
    X = X_org0.T
    Xclean = X.copy()
    X += np.random.standard_normal(X.shape) * 0
    X_org = X.T

 
    # Rotate to high-dimensional space
    Q = np.random.standard_normal((64,2))
    Q,_ = np.linalg.qr(Q)
    
    X = X.T.dot(Q.T) # rotate
    Xclean = Xclean.T.dot(Q.T)


################## 
    if(noise_db != 0):
        # Calculate signal power
        sig_power = np.mean(Xclean**2)

        # Convert SNR from dB scale to linear scale
        snr_linear = 10**(noise_db / 10)

        # Calculate noise power based on desired SNR
        noise_power = sig_power / snr_linear

        # Generate noise with the calculated power
        noise = np.random.normal(scale=np.sqrt(noise_power), size=X.shape)

        # Add noise to the signal
        X = X + noise
 ##################
 
    minv = np.min(X[0:ntr+nval])
    ptpv = np.ptp(X[0:ntr+nval])
   
    # scale 
    X = 2 * (X - np.min(X)) / np.ptp(X) - 1
    Xclean = 2 * (Xclean - np.min(Xclean)) / np.ptp(Xclean) - 1

    
    # split into training, validation and test set 
    # 0:nivs-1 (initial values for validation) | nivs:nivs+ntr-1 (training samples) | nivs+ntr:nivs+ntr+nval-1 (validation samples) | nivs+ntr+nval:end (testing samples, first nivs are used as initial values)
    X_train = X[nivs:nivs+ntr]   
    X_val = X[0:nivs+ntr+nval] 
    X_test = X[nivs+ntr+nval:nivs+ntr+nval+nivs2+npreds]  # For statistical analysis, first nivs are used as initial values

    X_train_clean = Xclean[nivs:nivs+ntr]   
    X_val_clean = Xclean[0:nivs+ntr+nval] 
    X_test_clean = Xclean[nivs+ntr+nval:nivs+ntr+nval+nivs2+npreds]  # For statistical analysis, first nivs are used as initial values      

    print('X shape :',X.shape)    
    
    m = X_train.shape[1]
    n = 1
    #******************************************************************************
    # Return train and test set
    #******************************************************************************
    return X_train, X_val, X_test, X_train_clean, X_val_clean, X_test_clean, m, n, Q, X_org, minv, ptpv



def wavy_beam(noise_db, theta, ntr,  nval, nivs, nivs2, npreds): # noise: % of ptp value
    
    # load the data from .mat file
    efld_mat=scipy.io.loadmat('data_KAE_wavy.mat')
    efld_org=np.array(efld_mat['data_KAE_wavy'])
    print('Raw data size  ')
    print(efld_org.shape)   
    
    
    X = efld_org 
    Xclean = efld_org

##################  
    if(noise_db != 0):
        # Calculate signal power
        sig_power = np.mean(Xclean**2)

        # Convert SNR from dB scale to linear scale
        snr_linear = 10**(noise_db / 10)

        # Calculate noise power based on desired SNR
        noise_power = sig_power / snr_linear

        # Generate noise with the calculated power
        noise = np.random.normal(scale=np.sqrt(noise_power), size=X.shape)

        # Add noise to the signal
        X = X + noise
 ##################

    minv = np.min(X[0:ntr+nval])
    ptpv = np.ptp(X[0:ntr+nval])
   
    # scale 
    X = 2 * (X - np.min(X)) / np.ptp(X) - 1
    Xclean = 2 * (Xclean - np.min(Xclean)) / np.ptp(Xclean) - 1

    
    # split into training, validation and test set 
    # 0:nivs-1 (initial values for validation) | nivs:nivs+ntr-1 (training samples) | nivs+ntr:nivs+ntr+nval-1 (validation samples) | nivs+ntr+nval:end (testing samples, first nivs are used as initial values)
    X_train = X[nivs:nivs+ntr]   
    X_val = X[0:nivs+ntr+nval] 
    X_test = X[nivs+ntr+nval:nivs+ntr+nval+nivs2+npreds]  # For statistical analysis, first nivs are used as initial values

    X_train_clean = Xclean[nivs:nivs+ntr]   
    X_val_clean = Xclean[0:nivs+ntr+nval] 
    X_test_clean = Xclean[nivs+ntr+nval:nivs+ntr+nval+nivs2+npreds]  # For statistical analysis, first nivs are used as initial values      

    print('X shape :',X.shape)         
    
    m = X_train.shape[1]
    n = 1
    #******************************************************************************
    # Return train and test set
    #******************************************************************************
    return X_train, X_test, X_train_clean, X_test_clean, m, n, minv, ptpv

def fluid_flow(noise_db, theta, ntr,  nval, nivs, nivs2, npreds): # noise: % of ptp value
    
    # load the data from .mat file
    flow_mat=scipy.io.loadmat('data_KAE_flow.mat')
    flow_org=np.array(flow_mat['data_KAE_flow'])
    print('Raw data size  ')
    print(flow_org.shape)   
    
    
    X = flow_org 
    Xclean = flow_org
 
################## 
    if(noise_db != 0): 
        # Calculate signal power
        sig_power = np.mean(Xclean**2)

        # Convert SNR from dB scale to linear scale
        snr_linear = 10**(noise_db / 10)

        # Calculate noise power based on desired SNR
        noise_power = sig_power / snr_linear

        # Generate noise with the calculated power
        noise = np.random.normal(scale=np.sqrt(noise_power), size=X.shape)

        # Add noise to the signal
        X = X + noise
 ##################
     

    minv = np.min(X[0:ntr+nval])
    ptpv = np.ptp(X[0:ntr+nval])
   
    # scale 
    X = 2 * (X - np.min(X)) / np.ptp(X) - 1
    Xclean = 2 * (Xclean - np.min(Xclean)) / np.ptp(Xclean) - 1

    
    # split into training, validation and test set 
    # 0:nivs-1 (initial values for validation) | nivs:nivs+ntr-1 (training samples) | nivs+ntr:nivs+ntr+nval-1 (validation samples) | nivs+ntr+nval:end (testing samples, first nivs are used as initial values)
    X_train = X[nivs:nivs+ntr]   
    X_val = X[0:nivs+ntr+nval] 
    X_test = X[nivs+ntr+nval:nivs+ntr+nval+nivs2+npreds]  # For statistical analysis, first nivs are used as initial values

    X_train_clean = Xclean[nivs:nivs+ntr]   
    X_val_clean = Xclean[0:nivs+ntr+nval] 
    X_test_clean = Xclean[nivs+ntr+nval:nivs+ntr+nval+nivs2+npreds]  # For statistical analysis, first nivs are used as initial values      

    print('X shape :',X.shape)             
    
    m = X_train.shape[1]
    n = 1
 
    #******************************************************************************
    # Return train and test set
    #******************************************************************************
    return X_train, X_val, X_test, X_train_clean, X_val_clean, X_test_clean, m, n, minv, ptpv      