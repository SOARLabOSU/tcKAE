import argparse
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, Dataset
print(torch.__version__)

import torch.nn.init as init

from read_dataset_pend import data_from_name
from model_tcl import *
from tools import *
from train_tcl import *
import cProfile

import os
import scipy.io  # open source library for basic math functions
#==============================================================================
# Training settings
#==============================================================================
parser = argparse.ArgumentParser(description='PyTorch Example')
#
parser.add_argument('--model', type=str, default='koopmanAE', metavar='N', help='model')
#
parser.add_argument('--alpha', type=int, default='4',  help='model width')
#
parser.add_argument('--dataset', type=str, default='pendulum', metavar='N', help='dataset')
#
parser.add_argument('--theta', type=float, default=2.4,  metavar='N', help='angular displacement')
#
parser.add_argument('--noise', type=float, default=0.0,  metavar='N', help='noise level')
#
parser.add_argument('--lr', type=float, default=1e-2, metavar='N', help='learning rate (default: 0.01)')
#
parser.add_argument('--wd', type=float, default=1e0, metavar='N', help='weight_decay (default: 1e-5)')
#
parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
#
parser.add_argument('--batch', type=int, default=40, metavar='N', help='batch size (default: 10000)')
#
parser.add_argument('--folder', type=str, default='test',  help='specify directory to print results to')
#
parser.add_argument('--gamma_fwd', type=float, default='4',  help='balance between reconstruction and temporal loss')
#
parser.add_argument('--gamma_bwd', type=float, default='2',  help='tune backward loss')
#
parser.add_argument('--gamma_con', type=float, default='1e-4',  help='tune consistent loss')
#
parser.add_argument('--gamma_tc', type=float, default='4',  help='scaling factor for temporal consistency loss ')
#
parser.add_argument('--gamma_lin', type=float, default='0',  help='scaling factor for linear consistency loss ')
#
parser.add_argument('--gamma_inf', type=float, default='0',  help='scaling factor for inf norm loss ')
#
parser.add_argument('--steps', type=int, default='16',  help='steps for learning forward dynamics')
#
parser.add_argument('--steps_back', type=int, default='16',  help='steps for learning backwards dynamics')
#
parser.add_argument('--steps_tc', type=int, default='8',  help='steps upto which we want to enforce the temporal consistency')
#
parser.add_argument('--steps_chk', type=int, default='0',  help='steps upto which we want to forecast and check')
#
parser.add_argument('--bottleneck', type=int, default='16',  help='size of bottleneck layer')
#
parser.add_argument('--lr_update', type=int, nargs='+', default=[30, 200, 400, 700], help='decrease learning rate at these epochs')
#
parser.add_argument('--lr_decay', type=float, default='0.2',  help='TCL penalty lambda hyperparameter')
#
parser.add_argument('--backward', type=int, default=1, help='flag for deciding to train with backward dynamics')
#
parser.add_argument('--init_scale', type=float, default=0.99, help='init scaling')
#
parser.add_argument('--gradclip', type=float, default=0.05, help='gradient clipping')
#
parser.add_argument('--pred_steps', type=int, default='1000',  help='prediction steps')
#
parser.add_argument('--seed_idx', type=int, default='1',  help='seed index')
#
parser.add_argument('--epoch_trans', type=int, default='50',  help='tc loss becomes constant after epoch_trans, linearly increases for epoch<epoch_trans')
#
parser.add_argument('--clr', type=int, default='0',  help='clr = 1, continuous variation of scaling factor for physics term')
#
parser.add_argument('--ntr', type=int, default='32',  help='size of training dataset')
#
parser.add_argument('--nval', type=int, default='8',  help='size of validation dataset')
#
parser.add_argument('--nivs', type=int, default='8',  help='initial values')
#
parser.add_argument('--nivs2', type=int, default='32',  help='initial values for testing')
#
parser.add_argument('--tc_opt', type=int, default='1',  help='decides whether to enforce temporal consistency in latent or original state-space')

args = parser.parse_args()

# Generate 10 different seed values (randomly selected from 0 to 1000)
torch.manual_seed(0)
sd_arr = torch.randint(0,1000,(20,))
print('seed array: ', sd_arr)
sd = sd_arr[args.seed_idx]
print('seed value:', sd)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed(sd)
torch.manual_seed(sd)
np.random.seed(sd)
set_seed(sd)
device = get_device()



#******************************************************************************
# Create folder to save results
#******************************************************************************
#if not os.path.isdir(args.folder):
if not os.path.isdir(args.folder):
    os.mkdir(args.folder)

#******************************************************************************
# load data
#******************************************************************************
Xtrain, Xval, Xtest, Xtrain_clean, Xval_clean, Xtest_clean, m, n, Q, X_org, minv, ptpv = data_from_name(args.dataset, noise=args.noise, theta=args.theta, ntr=args.ntr, nval = args.nval, nivs = args.nivs, nivs2 = args.nivs2, npreds = args.pred_steps) # Q: np_array[64,2], X_org: np_array[2200,2]
# Number of timesteps in validation set
print("m value: ",m)

print('Xtrain shape :',Xtrain.shape)
print('Xval shape :',Xval.shape)
print('Xtest shape :',Xtest.shape)

print('**** Save noisy Xtest data ****')
save_var1 = {'Xtest' : np.asarray(Xtest)} 
scipy.io.savemat(args.folder +'/Xtest.mat', dict(save_var1), appendmat=True, format='5', long_field_names=False, do_compression=False, oned_as='row')

print('**** Save noisy Xval data ****')
save_var2 = {'Xval' : np.asarray(Xval)} 
scipy.io.savemat(args.folder +'/Xval.mat', dict(save_var2), appendmat=True, format='5', long_field_names=False, do_compression=False, oned_as='row')

print('**** Save clean Xtest data ****')
save_var2 = {'Xtest_clean' : np.asarray(Xtest_clean)} 
scipy.io.savemat(args.folder +'/Xtest_clean.mat', dict(save_var2), appendmat=True, format='5', long_field_names=False, do_compression=False, oned_as='row')

print('**** Save Q matrix data ****')
save_var2 = {'Q' : np.asarray(Q)} 
scipy.io.savemat(args.folder +'/Q.mat', dict(save_var2), appendmat=True, format='5', long_field_names=False, do_compression=False, oned_as='row')


#******************************************************************************
# Reshape data for pytorch into 4D tensor Samples x Channels x Width x Hight
#******************************************************************************

Xtrain = add_channels(Xtrain)
Xtest = add_channels(Xtest)
Xval = add_channels(Xval)

# transfer to tensor
Xtrain = torch.from_numpy(Xtrain).float().contiguous()
Xtest = torch.from_numpy(Xtest).float().contiguous()
Xval = torch.from_numpy(Xval).float().contiguous()

print("Xtrain shape: ", Xtrain.shape)

#******************************************************************************
# Reshape data for pytorch into 4D tensor Samples x Channels x Width x Hight
#******************************************************************************
Xtrain_clean = add_channels(Xtrain_clean)
Xtest_clean = add_channels(Xtest_clean)
Xval_clean = add_channels(Xval_clean)

# transfer to tensor
Xtrain_clean = torch.from_numpy(Xtrain_clean).float().contiguous()
Xtest_clean = torch.from_numpy(Xtest_clean).float().contiguous()
Xval_clean = torch.from_numpy(Xval_clean).float().contiguous()

Xval = Xval.to(device)
Xval_clean = Xval_clean.to(device)

#******************************************************************************
# Create Dataloader objects
#******************************************************************************

trainDat = []
start = 0
for i in np.arange(1, -1, -1):  # original: for i in np.arange(args.steps,-1, -1):
    if i == 0:
        trainDat.append(Xtrain[start:].float())
    else:
        trainDat.append(Xtrain[start:-i].float())
    start += 1

train_data = torch.utils.data.TensorDataset(*trainDat) # list: [steps+1] [ntr-steps, 1, N_tot , 1]
print("trainDat type:",type(trainDat))
print("trainDat list length:",len(trainDat))
print("trainDat[0] size:",trainDat[0].size())
print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

# just for checking
# print('DRIVER min:',minv)
# print('DRIVER ptp:',ptpv)
# print("DRIVER trainDat[0]:",trainDat[0])
# dum0 = ((trainDat[0] + 1)*ptpv)/2+minv   # scale back to original  
# print("DRIVER dum0:",dum0)
# #dum0 = trainDat[0].to(device)          
# dum1 = torch.squeeze(dum0).to(device)
# dum2 = torch.mm(torch.transpose(Qt,0,1),torch.transpose(dum1,0,1)) # shape: [2, batch] 
# print('CHECKING dum2 size :', dum2.size())
# Ek = (Itia*(dum2[1,:]**2))/2;  # kinetic energy, tensor of size [n_batch]
# Ep = (mass*g)*(1-torch.cos(dum2[0,:]));  # potential energy, tensor of size [n_batch]             
# E_tot = Ek + Ep; #total energy is expected to remain constant 
# print('CHECKING E total: ', E_tot)

del(trainDat)

train_loader = DataLoader(dataset = train_data,
                              batch_size = args.batch,
                              shuffle = False)




#==============================================================================
# Model
#==============================================================================
model = koopmanAE(m, n, args.bottleneck, args.steps, args.steps_back, args.steps_tc, args.alpha, args.init_scale)
print('koopmanAE')
#model = torch.nn.DataParallel(model)
model = model.to(device)


#==============================================================================
# Model summary
#==============================================================================
print('**** Setup ****')
print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
print('Total params: %.2fk' % (sum(p.numel() for p in model.parameters())/1000.0))
print('************')
print(model)


start_time = time.perf_counter() # track training time
#==============================================================================
# Start training
#==============================================================================
weights, grads_before_clipping, grads_after_clipping, error_val, err_mean_val, snapshots_pred_val, snapshots_truth_val, tc_loss_hist, fwd_loss_hist, lin_loss_hist, inf_loss_hist, bwd_loss_hist, iden_loss_hist, con_loss_hist, loss_hist, valid_loss_hist, model, optimizer, epoch_hist = train_tcl(model, train_loader, Xval, minv, ptpv, epoch_trans=args.epoch_trans, gamma_tc=args.gamma_tc,
                                                                                                                                                                                                                                                                                                      gamma_lin = args.gamma_lin, gamma_inf = args.gamma_inf, lr=args.lr, weight_decay=args.wd, gamma_fwd=args.gamma_fwd, num_epochs = args.epochs,
                                                                                                                                                                                                                                                                                                      learning_rate_change=args.lr_decay, epoch_update=args.lr_update,
                                                                                                                                                                                                                                                                                                      gamma_bwd = args.gamma_bwd, gamma_con = args.gamma_con, backward=args.backward, steps=args.steps, steps_back=args.steps_back, steps_tc = args.steps_tc,
                                                                                                                                                                                                                                                                                                      gradclip=args.gradclip, clr=args.clr, ntr = args.ntr, steps_chk = args.steps_chk, nbatch_org = args.batch, pred_steps = args.pred_steps, tc_opt = args.tc_opt, nivs = args.nivs, nval = args.nval)

end_time = time.perf_counter()
elapsed_time = end_time - start_time

# print(f"*************************************************************")        
# print(f"id(weights) as output of train function: {id(weights)}")   
# print(f"*************************************************************") 
                    
torch.save(model.state_dict(), args.folder + '/model'+'.pkl')

save_folder = args.folder
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
    
'''
torch.save({
    'grads_before_clipping': grads_before_clipping,
    'grads_after_clipping': grads_after_clipping,
    'weights': weights
}, os.path.join(save_folder, 'training_metrics.pth'))
'''

print('**** Training time ****')
# Prepare the data to be saved in MATLAB format
data_to_save = {'elapsed_time': elapsed_time}
# Specify the file path
file_path = os.path.join(args.folder, 'elapsed_time.mat')
# Save the data
scipy.io.savemat(file_path, data_to_save)

print('**** Loss history ****')
print(torch.FloatTensor(loss_hist).data.cpu().numpy())
save_loss = {'loss' : torch.FloatTensor(loss_hist).data.cpu().numpy()} 
scipy.io.savemat(args.folder +'/loss.mat', dict(save_loss), appendmat=True, format='5', long_field_names=False, do_compression=False, oned_as='row')

print('**** Validation Loss history ****')
print(torch.FloatTensor(valid_loss_hist).data.cpu().numpy())
save_valid_loss = {'valid_loss' : torch.FloatTensor(valid_loss_hist).data.cpu().numpy()} 
scipy.io.savemat(args.folder +'/valid_loss.mat', dict(save_valid_loss), appendmat=True, format='5', long_field_names=False, do_compression=False, oned_as='row')

print('**** Temporal Consistency Loss history ****')
print(torch.FloatTensor(tc_loss_hist).data.cpu().numpy())
save_tc_loss = {'tc_loss' : torch.FloatTensor(tc_loss_hist).data.cpu().numpy()} 
scipy.io.savemat(args.folder +'/tc_loss.mat', dict(save_tc_loss), appendmat=True, format='5', long_field_names=False, do_compression=False, oned_as='row')

print('**** Forward Loss history ****')
print(torch.FloatTensor(fwd_loss_hist).data.cpu().numpy())
save_fwd_loss = {'fwd_loss' : torch.FloatTensor(fwd_loss_hist).data.cpu().numpy()} 
scipy.io.savemat(args.folder +'/fwd_loss.mat', dict(save_fwd_loss), appendmat=True, format='5', long_field_names=False, do_compression=False, oned_as='row')

print('**** Linear Loss history ****')
print(torch.FloatTensor(lin_loss_hist).data.cpu().numpy())
save_lin_loss = {'lin_loss' : torch.FloatTensor(lin_loss_hist).data.cpu().numpy()} 
scipy.io.savemat(args.folder +'/lin_loss.mat', dict(save_lin_loss), appendmat=True, format='5', long_field_names=False, do_compression=False, oned_as='row')

print('**** Infinity Loss history ****')
print(torch.FloatTensor(inf_loss_hist).data.cpu().numpy())
save_inf_loss = {'inf_loss' : torch.FloatTensor(inf_loss_hist).data.cpu().numpy()} 
scipy.io.savemat(args.folder +'/inf_loss.mat', dict(save_inf_loss), appendmat=True, format='5', long_field_names=False, do_compression=False, oned_as='row')

print('**** Backward Loss history ****')
print(torch.FloatTensor(bwd_loss_hist).data.cpu().numpy())
save_bwd_loss = {'bwd_loss' : torch.FloatTensor(bwd_loss_hist).data.cpu().numpy()} 
scipy.io.savemat(args.folder +'/bwd_loss.mat', dict(save_bwd_loss), appendmat=True, format='5', long_field_names=False, do_compression=False, oned_as='row')

print('**** Identity Loss history ****')
print(torch.FloatTensor(iden_loss_hist).data.cpu().numpy())
save_iden_loss = {'iden_loss' : torch.FloatTensor(iden_loss_hist).data.cpu().numpy()} 
scipy.io.savemat(args.folder +'/iden_loss.mat', dict(save_iden_loss), appendmat=True, format='5', long_field_names=False, do_compression=False, oned_as='row')

print('**** Consistency Loss history ****')
print(torch.FloatTensor(con_loss_hist).data.cpu().numpy())
save_con_loss = {'con_loss' : torch.FloatTensor(con_loss_hist).data.cpu().numpy()} 
scipy.io.savemat(args.folder +'/con_loss.mat', dict(save_con_loss), appendmat=True, format='5', long_field_names=False, do_compression=False, oned_as='row')                   
            
            
            
#******************************************************************************
# Prediction (Test Error)
#******************************************************************************
Xinput,_ = Xtest[:-1], Xtest[1:]
#_, Xtarget = Xtest_clean[:-1], Xtest_clean[1:]
_, Xtarget = Xtest_clean[:-1], Xtest_clean[1:]

print('**** Save Xinput from driver ****')
save_var1 = {'Xinput' : np.asarray(Xinput)} 
scipy.io.savemat(args.folder +'/Xinput.mat', dict(save_var1), appendmat=True, format='5', long_field_names=False, do_compression=False, oned_as='row')
print('**** Save Xtarget from driver ****')
save_var1 = {'Xtarget' : np.asarray(Xtarget)} 
scipy.io.savemat(args.folder +'/Xtarget.mat', dict(save_var1), appendmat=True, format='5', long_field_names=False, do_compression=False, oned_as='row')


snapshots_pred = []
snapshots_truth = []

print("Xinput size :",Xinput.size()) # tensor [test_samples,1,N_1,1]
# print("===============================================================================")
# print("===============================================================================")
# print("Last element which is actually extrapolated:  Xinput[0] :",Xinput[1])
# print("===============================================================================")
# print("===============================================================================")            
            
# Test Error calculation code here        
error_test = []
snap_pred = np.zeros((m, n))
snap_truth = np.zeros((m, n))

for i in range(args.nivs2):
            error_temp = []
            init = Xinput[i].float().to(device)
            if i == 0:
                init0 = init
            
            z = model.encoder(init) # embedd data in latent space

            for j in range(args.pred_steps):
                if isinstance(z, tuple):
                    z = model.dynamics(*z) # evolve system in time
                else:
                    z = model.dynamics(z)
                if isinstance(z, tuple):
                    x_pred = model.decoder(z[0])
                else:
                    x_pred = model.decoder(z) # map back to high-dimensional space
                target_temp = Xtarget[i+j].data.cpu().numpy().reshape(m,n)
                x_pred_temp = x_pred.data.cpu().numpy().reshape(m,n)
                #scale back before calculating error
                target_temp = ((target_temp + 1) * ptpv)/2 + minv
                x_pred_temp = ((x_pred_temp + 1) * ptpv)/2 + minv
                
                error_temp.append(np.linalg.norm(x_pred_temp - target_temp) / np.linalg.norm(target_temp)) #error_temp is a 1000 element list having floating point values of error
                
                if i == 0:
                    snapshots_pred.append(x_pred_temp)
                    snapshots_truth.append(target_temp)
                if j == args.pred_steps-1 - i: # save the average value at final prediction point
                    snap_pred += x_pred_temp
                    snap_truth += target_temp
                
            #print('error_temp length:',len(error_temp))
            error_test.append(np.asarray(error_temp)) #error is a nivs element list where each element is pred_steps by 1 array 
            

# Ensure snap_pred and snap_truth are numpy arrays or lists.
snap_pred = np.asarray(snap_pred) / args.nivs
snap_truth = np.asarray(snap_truth) / args.nivs
# Prepare the dictionary to save.
save_preds0 = {'pred_snap': snap_pred, 'truth_snap': snap_truth}
# Save the dictionary as a .mat file.
scipy.io.savemat(args.folder + '/snaps.mat', save_preds0, appendmat=True, format='5', long_field_names=False, do_compression=False, oned_as='row')


error_test = np.asarray(error_test) #error is a list of elements where each element is pred_steps by 1 array
#print('error shape: ',error.shape)


print('**** Save error_test from driver ****')
save_var1 = {'error_test' : np.asarray(error_test)} 
scipy.io.savemat(args.folder +'/error_test.mat', dict(save_var1), appendmat=True, format='5', long_field_names=False, do_compression=False, oned_as='row')

    
print(" error_val shape: ", error_val.shape)
print(" error_test shape: ", error_test.shape) 


print('**** Save error_val from training ****')
save_var2 = {'error_val' : np.asarray(error_val.numpy())} 
scipy.io.savemat(args.folder +'/error_val.mat', dict(save_var2), appendmat=True, format='5', long_field_names=False, do_compression=False, oned_as='row')


    
y1_test=np.quantile(error_test, .05, axis=0)
y2_test=np.quantile(error_test, .95, axis=0)
y1_val=np.quantile(error_val, .05, axis=0)
y2_val=np.quantile(error_val, .95, axis=0)

#print('y1 type: ',type(y1))
#print('y1 shape: ',y1.shape)

np.save(args.folder +'/000_pred.npy', error_test)

err1 = error_test.mean(axis=0)
upp_lim1 = np.transpose(y1_test)
low_lim1 = np.transpose(y2_test) 


err_mean_val = np.array(err_mean_val)  # From training
err_mean_test = np.mean(error_test.mean(axis=0))
err_min_test = np.mean(y1_test)
err_max_test = np.mean(y2_test)
err_min_val = np.mean(y1_val)
err_max_val = np.mean(y2_val)
print('Average error of first pred: ', err1[args.nval])
print('Average error of last pred: ', err1[-1])
print('Average error overarll pred: ', err_mean_test)

print('**** Save error_mean_val from training ****')
save_var2 = {'err_mean_val' : np.asarray(err_mean_val)} 
scipy.io.savemat(args.folder +'/err_mean_val.mat', dict(save_var2), appendmat=True, format='5', long_field_names=False, do_compression=False, oned_as='row')

err_stats_test = np.array([err_min_test,err_mean_test,err_max_test, err_max_test-err_min_test])
err_stats_val = np.array([err_min_val,err_mean_val,err_max_val, err_max_val-err_min_val])
#print('err_stats0 type:',type(err_stats0))
    
snapshots_pred_val = [pred.cpu().numpy() for pred in snapshots_pred_val]
snapshots_truth_val = [truth.cpu().numpy() for truth in snapshots_truth_val]

import scipy
save_preds = {'pred' : np.asarray(snapshots_pred), 'truth': np.asarray(snapshots_truth), 'init': np.asarray(init0.float().to(device).data.cpu().numpy().reshape(m,n))} 
scipy.io.savemat(args.folder +'/snapshots_pred.mat', dict(save_preds), appendmat=True, format='5', long_field_names=False, do_compression=False, oned_as='row')

save_preds_val = {'pred_val' : np.asarray(snapshots_pred_val), 'truth_val': np.asarray(snapshots_truth_val), 'init': np.asarray(init0.float().to(device).data.cpu().numpy().reshape(m,n))} 
scipy.io.savemat(args.folder +'/snapshots_pred_val.mat', dict(save_preds_val), appendmat=True, format='5', long_field_names=False, do_compression=False, oned_as='row')

save_err_stats_val = {'err_stats_val_arr' : np.asarray(err_stats_val)} 
scipy.io.savemat(args.folder +'/err_stats_val_arr.mat', dict(save_err_stats_val), appendmat=True, format='5', long_field_names=False, do_compression=False, oned_as='row')

save_err_stats_test = {'err_stats_test_arr' : np.asarray(err_stats_test)} 
scipy.io.savemat(args.folder +'/err_stats_test_arr.mat', dict(save_err_stats_test), appendmat=True, format='5', long_field_names=False, do_compression=False, oned_as='row')

save_err = {'err_arr' : np.asarray(err1)} 
scipy.io.savemat(args.folder +'/err_arr.mat', dict(save_err), appendmat=True, format='5', long_field_names=False, do_compression=False, oned_as='row')

save_upplim = {'upp_lim_arr' : np.asarray(upp_lim1)} 
scipy.io.savemat(args.folder +'/upp_lim_arr.mat', dict(save_upplim), appendmat=True, format='5', long_field_names=False, do_compression=False, oned_as='row')

save_lowlim = {'low_lim_arr' : np.asarray(low_lim1)} 
scipy.io.savemat(args.folder +'/low_lim_arr.mat', dict(save_lowlim), appendmat=True, format='5', long_field_names=False, do_compression=False, oned_as='row')



#plt.close('all')
#******************************************************************************
# Eigenvalues
#******************************************************************************
model.eval()

#if hasattr(model.dynamics, 'dynamics'):
A =  model.dynamics.dynamics.weight.cpu().data.numpy()
#A =  model.module.test.data.cpu().data.numpy()
w, v = np.linalg.eig(A)
#print(np.abs(w))

save_eigvals = {'eigvals' : np.asarray(w), 'eigvals_abs': np.asarray(np.abs(w))} 
scipy.io.savemat(args.folder +'/koopman_eigvals.mat', dict(save_eigvals), appendmat=True, format='5', long_field_names=False, do_compression=False, oned_as='row')

epoch_start = 1
epoch_end = args.epochs-1
#print(f"*************************************************************")        
#print(f"id(weights) just before printing in driver: {id(weights)}")   
#print(f"*************************************************************") 
#print("weights at epoch 0 (inside driver after training):")
for name, weight in weights[0].items():
    np.set_printoptions(threshold=10, edgeitems=2, linewidth=75, suppress=True)
    # Directly print the shape of the numpy array
    #print(f"epoch: {0}: {name}: {weight}")
    np.set_printoptions(edgeitems=3) 
print("weights at epoch args.epochs-1 (inside driver after training):")
for name, weight in weights[args.epochs-1].items():
    np.set_printoptions(threshold=10, edgeitems=2, linewidth=75, suppress=True)
    # Directly print the shape of the numpy array
    #print(f"epoch: {args.epochs-1}: {name}: {weight}")
    np.set_printoptions(edgeitems=3)  
    
    
'''

def plot_grads_for_epochs(grads_before_clipping, grads_after_clipping, epoch_arr, save_folder, layer_names):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for epoch in epoch_arr:
        epoch_index = epoch - 1
        # Calculate the number of rows needed for up to 16 subplots with 4 per row
        rows = min((len(layer_names) + 3) // 4, 4)
        fig, axs = plt.subplots(rows, 4, figsize=(20, 5 * rows))
        axs = axs.flatten()

        for i, layer in enumerate(layer_names):
            if i >= 16:  # Limit to a maximum of 16 subplots
                break
            grads_before = grads_before_clipping[epoch_index].get(layer, 0)
            grads_after = grads_after_clipping[epoch_index].get(layer, 0)

            axs[i].plot(['Before', 'After'], [grads_before, grads_after], marker='o', linestyle='-')
            axs[i].set_title(layer)
            axs[i].set_ylabel('Gradient Norm')

        # Hide any unused axes
        for j in range(i + 1, 16):
            axs[j].set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, f'grad_norms_epoch_{epoch}.jpg'))
        plt.close(fig)



# Define layer_names based on your model's layers
layer_names = [name for name, param in model.named_parameters() if param.requires_grad]


epoch_arr = [1, args.epochs-1]  # Define the epochs you want to plot
save_folder = args.folder

plot_grads_for_epochs(grads_before_clipping, grads_after_clipping, epoch_arr, save_folder, layer_names)



def plot_weight_histograms_for_epoch(weights, epoch, save_folder):
    # Ensure the save folder exists
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    # Set numpy print options for concise output
    np.set_printoptions(threshold=10, edgeitems=2, linewidth=75, suppress=True)
    
    # Prepare the plot grid
    num_layers = len(weights[epoch])
    cols = 4  # Max number of columns
    rows = min((num_layers + cols - 1) // cols, 4)  # Calculate rows needed, max 4 rows
    fig, axs = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    
    if num_layers <= cols:
        axs = [axs]  # Make it iterable if only one row
    
    # Flatten the axes array for easy indexing
    axs = axs.flatten()
    
    # Iterate through each layer's weights for the specified epoch
    for i, (name, weight) in enumerate(weights[epoch].items()):
        if i >= 16:  # Limit to a maximum of 16 subplots
            break
        # Plot histogram on the ith subplot
        axs[i].hist(weight.flatten(), bins=50, alpha=0.7, density=True)
        axs[i].set_title(f'Epoch {epoch+1}: {name}')
    
    # Hide unused subplots
    for j in range(i+1, len(axs)):
        axs[j].axis('off')
    
    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f'weight_distributions_epoch1_{epoch}.jpg'))
    plt.close(fig)
    
    # Reset numpy print options to default
    np.set_printoptions(edgeitems=3)


save_folder = args.folder # Update this to your desired path
epoch = 0  # For the first epoch (assuming 0-based indexing)
plot_weight_histograms_for_epoch(weights, epoch, save_folder)
epoch = args.epochs-1  # For the first epoch (assuming 0-based indexing)
plot_weight_histograms_for_epoch(weights, epoch, save_folder)

end_time2 = time.perf_counter()
elapsed_time2 = end_time2 - start_time

print("Total training time in seconds: ", elapsed_time)
# Data to be saved
elapsed_time2_str = str(elapsed_time2)  # Convert to string if it's not already
file_path = os.path.join(args.folder, 'elapsed_time2.txt')
with open(file_path, 'w') as file:
    file.write(elapsed_time2_str)


elapsed_time1_str = str(elapsed_time)  # Convert to string if it's not already
file_path = os.path.join(args.folder, 'elapsed_time.txt')
with open(file_path, 'w') as file:
    file.write(elapsed_time1_str) 
    


fig = plt.figure(figsize=(6.1, 6.1), facecolor="white",  edgecolor='k', dpi=150)
plt.scatter(w.real, w.imag, c = '#dd1c77', marker = 'o', s=15*6, zorder=2, label='Eigenvalues')

maxeig = 1.4
plt.xlim([-maxeig, maxeig])
plt.ylim([-maxeig, maxeig])
plt.locator_params(axis='x',nbins=4)
plt.locator_params(axis='y',nbins=4)

#plt.xlabel('Real', fontsize=22)
#plt.ylabel('Imaginary', fontsize=22)
plt.tick_params(axis='y', labelsize=22)
plt.tick_params(axis='x', labelsize=22)
plt.axhline(y=0,color='#636363',ls='-', lw=3, zorder=1 )
plt.axvline(x=0,color='#636363',ls='-', lw=3, zorder=1 )

#plt.legend(loc="upper left", fontsize=16)
t = np.linspace(0,np.pi*2,100)
plt.plot(np.cos(t), np.sin(t), ls='-', lw=3, c = '#636363', zorder=1 )
plt.tight_layout()
plt.show()
plt.savefig(args.folder +'/000eigs' +'.png')
plt.savefig(args.folder +'/000eigs' +'.eps')
plt.close()

plt.close('all')

'''
