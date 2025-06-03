import torch
from torch import nn
import numpy as np

from tools import *



def train_tcl(model, train_loader, Xval, minv, ptpv, epoch_trans, gamma_tc, gamma_lin, gamma_inf, lr, weight_decay, 
          gamma_fwd, num_epochs, learning_rate_change, epoch_update, 
          gamma_bwd, gamma_con, backward, steps, steps_back, steps_tc, gradclip, clr, ntr, steps_chk, nbatch_org, pred_steps, tc_opt, nivs, nval):

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    device = get_device()
             
            
    def lr_scheduler(optimizer, epoch, lr_decay_rate=0.8, decayEpoch=[]):
                    """Decay learning rate by a factor of lr_decay_rate every lr_decay_epoch epochs"""
                    if epoch in decayEpoch:
                        for param_group in optimizer.param_groups:
                            param_group['lr'] *= lr_decay_rate
                        return optimizer
                    else:
                        return optimizer
                        
                     
        

    criterion = nn.MSELoss().to(device)
    
    gamma_tcr = gamma_tc/(epoch_trans-1)   # Linear increase rate for phy_sct
    
    epoch_hist = []
    loss_hist = []
    tc_loss_hist = []   
    fwd_loss_hist = []
    lin_loss_hist = []
    inf_loss_hist = []
    bwd_loss_hist = []
    iden_loss_hist = []
    con_loss_hist =[]
    epoch_loss = []
    valid_loss_hist = []

    # for logging the weghts and gradients
    grads_before_clipping = []
    grads_after_clipping = []
    weights = []

    with torch.autograd.set_detect_anomaly(True):

        for epoch in range(num_epochs):
            #print(epoch)
            loss_av = 0
            loss_tc_av = 0      
            loss_fwd_av = 0
            loss_lin_av = 0
            loss_inf_av = 0
            loss_bwd_av = 0
            loss_iden_av = 0
            loss_con_av = 0
            epoch_grads_before_sum = {name: [] for name, param in model.named_parameters() if param.requires_grad}
            epoch_grads_after_sum = {name: [] for name, param in model.named_parameters() if param.requires_grad}

            

            # ===================================== different learning options ====================================

            if(clr == 1): # continuous learning, varying gamma_tc continuously
                if(epoch<epoch_trans):
                    gamma_tc = epoch*gamma_tcr
                else:
                    gamma_tc = (epoch_trans-1)*gamma_tcr                                 
            # ======================================================================================================   
            
            max_batch_idx = ntr // nbatch_org
                
            for batch_idx, data_list in enumerate(train_loader):
                
                #print('batch_idx: ', batch_idx)

                
                if batch_idx == max_batch_idx:
                    max_batch_flag = 1
                else:
                    max_batch_flag = 0
                
                
                model.train()
                out, out_back, out_lat = model(data_list[0].to(device), mode='forward') #"data_list" and "out" has shape: [steps+1][batch,1,N_tot,1]
                
                
                               
                #print('data_list length: ', len(data_list))
                #print('data_list[0] size: ', data_list[0].size())
                
                #print('out_lat[0] size: ', out_lat[0].size(), 'out[0] size: ', out[0].size())
                #loss_fwd = 0.0
                
                nbatch = data_list[0].size()[0]
                
                steps_max_fwd = min(nbatch, steps)
                
                for k in range(steps_max_fwd): #note that stps
                    if k == 0:
                        #loss_fwd = criterion(out[k], data_list[k+1].to(device))
                        #print('length of data_list : ', len(data_list), ' length of out : ',len(out))
                        loss_fwd = criterion(out[0], data_list[0+1].to(device))
                    else:
                        #loss_fwd += criterion(out[k], data_list[k+1].to(device))
                        loss_fwd += criterion(out[k][:-k,:,:], data_list[1][k:,:,:].to(device))

                
                loss_identity = criterion(out[-1], data_list[0].to(device)) * steps

                # get the size of the batch and divide by 2
                # print('nbatch: ',nbatch)


                 
                if gamma_lin != 0 and gamma_inf != 0: # 
                # This is for enforcing linearity in latent space, for DAE (Lusch paper)
                    _, _, out_dae = model(data_list[0].to(device), mode='DAE_linear')
                    loss_linear_arr = torch.zeros(nbatch-1).to(device) 
                    for k in range(1,nbatch):
                        #print('k:',k)
                        #print('out_lat length: ', len(out_lat))
                        # print('\n')
                        loss_linear_arr[k-1] += criterion(out_dae[-1][k:nbatch,:,:], out_dae[k-1][0:nbatch-k,:,:])
                    loss_linear = loss_linear_arr.sum()
                    
                # This is for enforcing inf norm, for DAE (Lusch paper)
                    # Calculating the absolute difference between the tensors
                    diff1 = torch.abs(out[-1] - data_list[0].to(device))
                    diff2 = torch.abs(out[0] - data_list[1].to(device))                
                    # Finding the maximum value in the diff tensor, which will give the L∞ norm of the difference tensor.
                    loss_inf = (diff1.max() + diff2.max()) * steps
                    #print("loss_inf:", loss_inf)
                    
                else:
                    loss_linear = 0
                    loss_inf = 0

                    
                        
                # temporal consistency loss (calculated only if gamma_tc != 0)
                if gamma_tc != 0:
                    steps_max = min(nbatch, steps_tc)
                    loss_tc_arr = torch.zeros(steps_max-1).to(device)
         
                    for kc in range(1,steps_max): 
                        for k in range(steps_tc-kc):
                             #print('kc:',kc)
                             #print('k:',k)
                            # print('out length: ', len(out))
                            # print('\n')
                            
                             if tc_opt == 1:
                                # In latent space 
                                loss_tc_arr[kc-1] += criterion(out_lat[k][kc:nbatch,:,:], out_lat[k+kc][0:nbatch-kc,:,:])
                                #loss_tc_arr[kc-1] += (steps_tc-kc-k)*criterion(out_lat[k][kc:nbatch,:,:], out_lat[k+kc][0:nbatch-kc,:,:])
                             else:
                                # In original state-space
                                loss_tc_arr[kc-1] += criterion(out[k][kc:nbatch,:,:], out[k+kc][0:nbatch-kc,:,:])

                        loss_tc_arr[kc-1] = loss_tc_arr[kc-1] * (steps/(steps_tc-kc+1))  # normalize similar to other losses
                        loss_sum_norm = loss_tc_arr.sum()/(steps_max-1)
                    
                        #print("loss_tc_arr", loss_tc_arr)
                        #print("loss_sum_norm", loss_sum_norm)
                    if(nbatch < nbatch_org and steps_chk >= nbatch): # indicating last batch
                        steps_chk = nbatch-1

                    # print('steps_max: ',steps_max)
                    # print('nbatch: ',nbatch)
                    # print('nbatch_org: ',nbatch_org)   
                    # print('steps_chk: ',steps_chk)
                    # print('loss_tc_arr size: ',loss_tc_arr.size())
                    # print('\n')
                    if(steps_chk == 0):
                        loss_tc = loss_sum_norm
                        #print("loss_tc: ", loss_tc)
                    else:
                        loss_tc = loss_tc_arr[steps_chk-1]  #steps_chk < steps_max
                        #print("loss_tc: ", loss_tc)

                else:
                    loss_tc = 0.0
                    
                    
                    
                # Backward and consistency loss (calculated only if backward = 1)   
                loss_bwd = 0.0
                loss_consist = 0.0                       

                if backward == 1:
                    out, out_back, out_back_lat = model(data_list[-1].to(device), mode='backward') # last element of the list as input
       
                    steps_back = steps
                    steps_max_bwd = min(nbatch, steps_back)
                    for k in range(steps_max_bwd):
                        
                        if k == 0:
                            #loss_bwd = criterion(out_back[k], data_list[::-1][k+1].to(device))
                            #print('length of data_list[::-1] : ', len(data_list[::-1]), ' length of out_back : ',len(out_back), ' length of out : ',len(out))
                            loss_bwd = criterion(out_back[0], data_list[::-1][0+1].to(device))
                        else:
                            #loss_bwd += criterion(out_back[k], data_list[::-1][k+1].to(device))
                            loss_bwd += criterion(out_back[k][k:,:,:], data_list[::-1][1][:-k,:,:].to(device))
                            
                                   
                    A = model.dynamics.dynamics.weight
                    B = model.backdynamics.dynamics.weight

                    K = A.shape[-1]

                    for k in range(1,K+1):
                        As1 = A[:,:k]
                        Bs1 = B[:k,:]
                        As2 = A[:k,:]
                        Bs2 = B[:,:k]

                        Ik = torch.eye(k).float().to(device)

                        if k == 1:
                            loss_consist = (torch.sum((torch.mm(Bs1, As1) - Ik)**2) + \
                                             torch.sum((torch.mm(As2, Bs2) - Ik)**2) ) / (2.0*k)
                        else:
                            loss_consist += (torch.sum((torch.mm(Bs1, As1) - Ik)**2) + \
                                             torch.sum((torch.mm(As2, Bs2)-  Ik)**2) ) / (2.0*k)
        
      
                
                # if (epoch) % 20 == 0:
                    # #======== calculating the numerator involving loss_phy ==============
                    # optimizer.zero_grad()        
                    # loss_tc.backward(retain_graph = True)      # L_r in the paper  
                    # # Calculate gradient of each parameter at each layer
                    # #print("temporal consistency loss gradients_____________________________________________")            
                    # grads_tc = []
                    # param_idx = 0
                    # for param in model.parameters():
                        # param_idx = param_idx + 1
                        # #print("param_idx: ", param_idx)
                        # if(param.grad is not None):
                            # grads_tc.append(param.grad.view(-1))            
                    # grads_tc = torch.cat(grads_tc)
                    # # print("grads_tc type:", type(grads_tc))
                    # # print("grads_tc size:", grads_tc.size())
                    # # print("grads_tc:", grads_tc)             
                    # #print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")    
     
                
                if(clr == 1):
                    loss =  loss_identity + gamma_fwd * loss_fwd +  gamma_bwd * loss_bwd + gamma_con * loss_consist + gamma_tc * loss_tc + gamma_lin * loss_linear + gamma_inf * loss_inf
                    #print("loss: ", loss)
                else:
                    if gamma_tc == 0 and backward == 0: #DAE
                        loss = loss_identity + gamma_fwd * loss_fwd + gamma_lin * loss_linear + gamma_inf * loss_inf
                    elif gamma_tc == 0 and backward == 1: #cKAE
                        loss = loss_identity + gamma_fwd * loss_fwd +  gamma_bwd * loss_bwd + gamma_con * loss_consist 
                    elif gamma_tc != 0 and backward == 1: #tcKAE with backward  
                        if(epoch<epoch_trans):
                            print("Before epoch transition, epoch = ", epoch, " eptrans = ", epoch_trans)
                            loss = loss_identity + gamma_fwd * loss_fwd +  gamma_bwd * loss_bwd + gamma_con * loss_consist                            
                        else:
                            print("After epoch transition, epoch = ", epoch, " eptrans = ", epoch_trans)
                            loss = loss_identity + gamma_fwd * loss_fwd +  gamma_bwd * loss_bwd + gamma_con * loss_consist + gamma_tc * loss_tc
                    elif gamma_tc != 0 and backward == 0: #tcKAE with forward  
                        if(epoch<epoch_trans):
                            loss = loss_identity + gamma_fwd * loss_fwd + gamma_lin * loss_linear + gamma_inf * loss_inf                           
                        else:
                            loss = loss_identity + gamma_fwd * loss_fwd + + gamma_lin * loss_linear + gamma_inf * loss_inf + gamma_tc * loss_tc
                            
                         #loss = loss_fwd + gamma_fwd * loss_identity +  gamma_bwd * loss_bwd + gamma_con * loss_consist + gamma_lin * loss_linear + gamma_inf * loss_inf               
                    #print("loss: ", loss)
                
                loss_av = loss_av + loss
                #print("loss_av: ")
                loss_tc_av = loss_tc_av + loss_tc         
                loss_fwd_av = loss_fwd_av + loss_fwd
                loss_lin_av = loss_lin_av + loss_linear
                loss_inf_av = loss_inf_av + loss_inf
                loss_bwd_av = loss_bwd_av + loss_bwd
                loss_iden_av = loss_iden_av + loss_identity
                loss_con_av = loss_con_av + loss_consist
                                                    


                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                
                # Save gradients before clipping and weights
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        grad_norm_before = param.grad.norm().item() if param.grad is not None else 0
                        epoch_grads_before_sum[name].append(grad_norm_before)     
                        
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradclip) # gradient clip
                
                # Save gradients after clipping and store all weights
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        grad_norm_after = param.grad.norm().item() if param.grad is not None else 0
                        epoch_grads_after_sum[name].append(grad_norm_after)  # Store each batch's grad norm after clipping
                        


                        
                optimizer.step()           


            # Inside your epoch loop, after the batch loop has completed
            epoch_weights_after_last_batch = {name: param.data.cpu().numpy().copy() for name, param in model.named_parameters() if param.requires_grad}

            # Optionally, you can then append this dictionary to a list if you want to track the weights over multiple epochs
            weights.append(epoch_weights_after_last_batch)

            
            # After processing all batches in an epoch
            epoch_grads_before_avg = {name: np.mean(grads) for name, grads in epoch_grads_before_sum.items()}
            epoch_grads_after_avg = {name: np.mean(grads) for name, grads in epoch_grads_after_sum.items()}

  
                
            # Store the averaged gradient data
            grads_before_clipping.append(epoch_grads_before_avg)
            grads_after_clipping.append(epoch_grads_after_avg)
            

          
            
            

                
            # schedule learning rate decay    
            lr_scheduler(optimizer, epoch, lr_decay_rate=learning_rate_change, decayEpoch=epoch_update)
            loss_av = loss_av/(batch_idx+1.0)
            loss_tc_av = loss_tc_av/(batch_idx+1.0)      
            loss_fwd_av = loss_fwd_av/(batch_idx+1.0)
            loss_lin_av = loss_lin_av/(batch_idx+1.0)
            loss_inf_av = loss_inf_av/(batch_idx+1.0)
            loss_bwd_av = loss_bwd_av/(batch_idx+1.0)
            loss_iden_av = loss_iden_av/(batch_idx+1.0)
            loss_con_av = loss_con_av/(batch_idx+1.0)
            
            loss_hist.append(loss_av.data)
            if gamma_tc != 0:
                tc_loss_hist.append(loss_tc_av.data)      
            fwd_loss_hist.append(loss_fwd_av.data)
            if gamma_lin!=0 and gamma_inf != 0:
                lin_loss_hist.append(loss_lin_av.data)
                inf_loss_hist.append(loss_inf_av.data) 
            #bwd_loss_hist.append(loss_bwd_av.data)
            if backward != 0:
                bwd_loss_hist.append(loss_bwd_av)
                con_loss_hist.append(loss_con_av)             
            iden_loss_hist.append(loss_iden_av.data) 
            #con_loss_hist.append(loss_con_av.data)               
            epoch_loss.append(epoch)          
            #phy_sct_hist.append(phy_sctt)  
            #phy_scd_hist.append(phy_scdd)  
      
            

            # Model evaluation at the end of each epoch 
            model.eval()  # Set the model to evaluation mode
            with torch.no_grad():  # Deactivate gradient computation
            
            
                # Validation Error Calculation Code Here
                
                Xinput_val = Xval[0:nivs]
                Xtarget_val = Xval[nivs+ntr:nivs+ntr+nval]
                
                #print("Xinput size: ",Xinput.size())
                #print("Xtarget size: ",Xtarget.size())


                snapshots_pred_val = []
                snapshots_truth_val = []

                #print("Xinput size :",Xinput.size()) # tensor [test_samples,1,N_1,1]

                # For validation, we will use the reconstruction error in next nivs time samples
                error = []
                #print("nivs = ",nivs, " ntr = ",ntr, " nval = ",nval)
                for i in range(nivs):
                    error_temp_val = []
                    init = Xinput_val[i].float().to(device)
                    if i == 0:
                        init0 = init
                    
                    z = model.encoder(init) # embedd data in latent space
                    #pred_steps = Xtarget.size(0)
                    m = Xtarget_val.size(2)
                    n = 1
                    
                    snap_pred_val = np.zeros((m, n))
                    snap_truth_val = np.zeros((m, n))
                    
                    for j in range(nivs+ntr+nval+3):
                        if isinstance(z, tuple):
                            z = model.dynamics(*z) # evolve system in time
                        else:
                            z = model.dynamics(z)
                        if isinstance(z, tuple):
                            x_pred = model.decoder(z[0])
                        else:
                            x_pred = model.decoder(z) # map back to high-dimensional space
                        #if i+j>420:
                        # print("Epoch: ",epoch)
                        # print("nval: ",nval)
                        # print("Xtarget size: ",Xtarget.size())
                        # print("i :",i)
                        # print("j :",j)
                        # print("i+j: ",i+j)
                        
                        #if(j >= nivs+ntr-1-i and j <= nivs+ntr-1-i+nval-1):
                        if(j >= nivs+ntr-1-i and j <= nivs+ntr-1-i+nval-1):
                            i_tar = j - (nivs+ntr-1-i)
                            
                            #print("i = ",i, " j = ",j, " i_tar = ",i_tar, " max{j} = ", nivs+ntr+nval-i-1)
                            target_temp_val = Xtarget_val[i_tar].reshape(m, n)
                            x_pred_temp_val = x_pred.reshape(m, n)
                            
                            # scale back before calculating error
                            target_temp_val = ((target_temp_val + 1) * ptpv) / 2 + minv
                            x_pred_temp_val = ((x_pred_temp_val + 1) * ptpv) / 2 + minv

                            #print(x_pred_temp.device)  # should print cuda:0 or similar
                            #print(target_temp.device)  # should print cuda:0 or similar
                            error_temp_val.append((x_pred_temp_val - target_temp_val).norm() / target_temp_val.norm())  # Here we are using PyTorch norm

                            if i == 0:
                                snapshots_pred_val.append(x_pred_temp_val)
                                snapshots_truth_val.append(target_temp_val) 
                            if j == pred_steps-1 - i: # save the average value at final prediction point
                                snap_pred_val += x_pred_temp_val
                                snap_truth_val += target_temp_val
                    #print('error_temp length:',len(error_temp))
                    error.append(torch.stack(error_temp_val)) #error is a nivs element list where each element is nval by 1 array 

                error_tensor_val = torch.stack(error)
                #print('error_tensor size:',error_tensor.size())
                # error_val = torch.zeros((nivs, nval), device=device)

                # for i in range(nivs):
                    # error_val[i, :] = error_tensor[i, nivs-i-1:nivs-i-1+nval]
                error_val = error_tensor_val
                err_mean_val = error_val.mean().item()  # Calculating mean using PyTorch
                
            
            
            valid_loss_hist.append(err_mean_val)

            
            model.train()  # Set the model back to training mode

            if (epoch) % 20 == 0:
                    print('********** Epoche %s **********' %(epoch+1))                
                    print("loss identity: ", loss_iden_av.item())
                    if backward == 1:
                        print("loss backward: ", loss_bwd_av.item())
                        print("loss consistent: ", loss_con_av.item())
                    print("loss forward: ", loss_fwd_av.item())
                    if gamma_lin != 0 and gamma_inf != 0:
                        print("loss linear: ", loss_lin_av.item())
                        print("loss infinity: ", loss_inf_av.item())
                    if gamma_tc != 0:
                        print("loss pred consistency: ", loss_tc_av.item())               
                    print("loss sum: ", loss_av.item())
                    print("loss validation: ", err_mean_val)
                    epoch_hist.append(epoch+1) 

                    if hasattr(model.dynamics, 'dynamics'):
                        w, _ = np.linalg.eig(model.dynamics.dynamics.weight.data.cpu().numpy())
                        print(np.abs(w))







        if backward == 1:
            loss_consist = loss_consist.item()
                    
           
        return weights, grads_before_clipping, grads_after_clipping, error_val, err_mean_val, snapshots_pred_val, snapshots_truth_val, tc_loss_hist, fwd_loss_hist, lin_loss_hist, inf_loss_hist, bwd_loss_hist, iden_loss_hist, con_loss_hist, loss_hist, valid_loss_hist, model, optimizer, [epoch_hist, loss_fwd.item(), loss_consist]


