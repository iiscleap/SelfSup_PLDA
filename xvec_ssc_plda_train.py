#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 15:51:43 2019

@author: prachi singh 
@email: prachisingh@iisc.ac.in 

This code is for DNN training 
Explained in paper:
P. Singh, S. Ganapathy, Self-Supervised Metric Learning With Graph Clustering For Speaker Diarization


Check main shell scripts:  , to run for different parameters
"""

import os
import sys
import numpy as np
import random
import pickle
import subprocess
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
from models_train_ssc_plda import weight_initialization,Deep_Ahc_model
import torch.utils.data as dloader
from arguments import read_arguments as params
from pdb import set_trace as bp
sys.path.insert(0,'services/')
import kaldi_io
import services.agglomerative_dihard as ahc
import services.pic_dihard_ami as pic


# read arguments
opt = params()
# #select device
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpuid

loss_lamda = opt.alpha
dataset=opt.dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
print(device)

# Model defined here
def normalize(system):
     # to make zero mean and unit variance
        my_mean = np.mean(system)
        my_std = np.std(system)
        system = system-my_mean
        system /= my_std
        return system


def mostFrequent(arr, n):

    # Insert all elements in Hash.
    Hash = dict()
    for i in range(n):
        if arr[i] in Hash.keys():
            Hash[arr[i]] += 1
        else:
            Hash[arr[i]] = 1

    # find the max frequency
    max_count = 0
    res = -1
    for i in Hash:
        if (max_count < Hash[i]):
            res = i
            max_count = Hash[i]

    return res


class Deep_AHC:
    def __init__(self,pldamodel,fname,reco2utt,xvecdimension,n_prime,writer=None):
        self.reco2utt = reco2utt
        self.xvecD = xvecdimension
        # self.model = model
        # self.optimizer = optimizer
        self.n_prime = n_prime
        self.fname = fname
        self.final =0
        self.forcing_label = 0
        self.results_dict={}
        self.pldamodel = pldamodel
        self.data = None
        self.lamda = 0.0
        self.K = 40
        self.z = 0.5
        
        
    def write_results_dict(self, output_file):
        """Writes the results in label file"""
        f = self.fname
        output_label = open(output_file+'/'+f+'.labels','w')

        hypothesis = self.results_dict[f]
        meeting_name = f
        reco = self.reco2utt.split()[0]
        utts = self.reco2utt.rstrip().split()[1:]
        if reco == meeting_name:
            for j,utt in enumerate(utts):
                towrite = utt +' '+str(hypothesis[j])+'\n'
                output_label.writelines(towrite)
        output_label.close()

        rttm_channel=0
        segmentsfile = opt.segments+'/'+f+'.segments'
        python = opt.which_python
      
        cmd = '{} {}/diarization/make_rttm.py --rttm-channel 0 {} {}/{}.labels {}/{}.rttm' .format(python,opt.kaldi_recipe_path,segmentsfile,output_file,f,output_file,f)        
        os.system(cmd)
    

    def compute_score(self,rttm_gndfile,rttm_newfile,outpath,overlap):
      fold_local='services/'
      scorecode='score.py -r '
     
      # print('--------------------------------------------------')
      if not overlap:

          cmd=opt.which_python +' '+ fold_local + 'dscore-master/' + scorecode + rttm_gndfile + ' --ignore_overlaps --collar 0.25 -s ' + rttm_newfile + ' > ' + outpath + '.txt'
          os.system(cmd)
      else:
          cmd=opt.which_python + ' '+ fold_local + 'dscore-master/' + scorecode + rttm_gndfile + ' -s ' + rttm_newfile + ' > ' + outpath + '.txt'
          os.system(cmd)
      # print('----------------------------------------------------')
      # subprocess.check_call(cmd,stderr=subprocess.STDOUT)
      # print('scoring ',rttm_gndfile)
      bashCommand="cat {}.txt | grep OVERALL |awk '{{print $4}}'".format(outpath)
      output=subprocess.check_output(bashCommand,shell=True)
      return float(output.decode('utf-8').rstrip())
      # output = subprocess.check_output(['bash','-c', bashCommand])
  
    def normalized_bce_loss(self,output,target):
        output_sig = self.sigmoid(output)
        # norm loss 
        # num = -( target * torch.log(output_sig) + (1-target)*torch.log(1-output_sig))
        # target_1 =  target * torch.log(1-output_sig)
        # target_0 = (1-target) * torch.log(output_sig)

        # loss with weight
        weight = torch.Tensor([0.6],device=device)
        num = -(weight * target * torch.log(output_sig) + (1-weight)*(1-target)*torch.log(1-output_sig))
        target_1 = weight * target * torch.log(1-output_sig)
        target_0 = (1-weight)*(1-target) * torch.log(output_sig)

        den = num - (target_1+target_0)
        
        loss  = torch.mean(num/den)
        # loss = torch.mean(num)

        return loss

    def compute_bce_scores_loss(self,scores,labelfull,clean_ind=[],soft=0):
        N_clean =  scores.shape[0]
        if len(clean_ind) !=0:
            N_clean = len(clean_ind)
            labelfull = labelfull[clean_ind]
            scores = scores[np.ix_(clean_ind,clean_ind)]
        uni_gnd = np.unique(labelfull)
        mask_clean = torch.zeros((N_clean,N_clean))
        
        for ind in uni_gnd:
            index = np.where(labelfull==ind)[0]
            mask_clean[np.ix_(index,index)] = 1 
        # bp()
        # scores_clean = scores_clean * mask_clean
        mask_clean[np.tril_indices(N_clean)] = -1
        full_target = mask_clean[np.triu_indices(N_clean,k=1)]
        full_target_soft = full_target
        targetind = np.where(full_target==1)[0]
        nontargetind = np.where(full_target==0)[0]
        full_target_soft[targetind] = 1.0
        full_target_soft[nontargetind] = 0.0
        random.shuffle(targetind)
        random.shuffle(nontargetind)
        y = scores[np.triu_indices(N_clean,k=1)]
        print('total pairs:{} target pairs:{} nontarget pairs: {}'.format(len(full_target),len(targetind),len(nontargetind)))
        
        N_batches =1
        N_pairs = len(full_target)
        batchsize = 400000
        if N_pairs > batchsize:
            N_batches = int(N_pairs/batchsize)
        minibatches = np.arange(N_pairs)
        np.random.shuffle(minibatches)
        N_target_org = len(targetind)
        N_nontarget_org = len(nontargetind)
        if N_batches > 1:
            N_target = len(targetind)- (len(targetind) % N_batches)
            mini_target = targetind[:N_target].reshape(N_batches,-1)
            N_nontarget = len(nontargetind)- (len(nontargetind) % N_batches)
            mini_nontarget = nontargetind[:N_nontarget].reshape(N_batches,-1)
            minibatches = np.hstack((mini_target,mini_nontarget))
            np.random.shuffle(minibatches.T)
            # minibatches = minibatches[:batchsize*N_batches]
            print('N_target:{} minitarget:{} N_nontarget:{} mininontargets:{}'.format(N_target,mini_target.shape[1],N_nontarget,mini_nontarget.shape[1]))
            # pos_weight = torch.FloatTensor ([mini_nontarget.shape[1] / mini_target.shape[1] ], device=device)
        else:
            N_target = len(targetind)
            N_nontarget = len(nontargetind)
            minibatches = minibatches.reshape(N_batches,-1)
        
        print('minibatches shape: ',minibatches.shape)
        # pos_weight = torch.FloatTensor ([N_nontarget / N_target], device=device)
        pos_weight = torch.FloatTensor ([1], device=device)
        if soft:
            return y,full_target_soft,minibatches,pos_weight
        else:
            return y,full_target,minibatches,pos_weight
    
    
    def compute_loss(self,A,minibatch,lamda):
        loss = 0.0
        weight = 1

        for m in minibatch:
            loss += -weight*A[m[0],m[1]]+lamda*(A[m[0],m[2]]+A[m[1],m[2]])+ 3.0
            # loss += -weight*A[m[0],m[1]]+lamda*(A[m[0],m[2]])+ 2.0
        # print('sum loss : ',loss)
        return loss/len(minibatch)
    
    def compute_cluster(self,labels):
        unifull = np.unique(labels)
        ind = []
        for i,val in enumerate(unifull):
            ind.append((np.where(labels==val)[0]).tolist())
        return ind

    def dataloader_from_list(self):
        reco2utt = self.reco2utt
        D = self.xvecD

        channel = 1

        reco2utt=reco2utt.rstrip()
        f = self.fname
        utts = reco2utt.split()[1:]
        filelength = len(utts)
        if os.path.isfile(opt.xvecpath+f+'.npy'):
            system = np.load(opt.xvecpath+f+'.npy')

        else:
            print('xvectors in numpy format required!')
            return 0
            # arkscppath=opt.xvecpath+'xvector.scp'
            # xvec_dict= { key:mat for key,mat in kaldi_io.read_vec_flt_scp(arkscppath) }
            # system = np.empty((len(utts),D))
            # for j,key in enumerate(utts):
            #     system[j] = xvec_dict[key]
            # if not os.path.isdir(opt.xvecpath):
            #     os.makedirs(opt.xvecpath)
            # np.save(opt.xvecpath+f+'.npy',system)

        x1_array=system[np.newaxis]
        data_tensor = torch.from_numpy(x1_array).float()
        self.data = data_tensor
        return filelength
    
   
    def train_with_selective_binary_entropy(self,model_init,pretrain=0):
        """
        Train the SSC using N* clusters using PIC

        Parameters
        ----------
       Initial model

        Returns
        -------
        None.

        """
       
        alpha = loss_lamda # final stage
        count = 0
        f = self.fname
        data = self.data
        
        print('---------------------------------------------------------')
        print('\nfilename:',f)
        count +=1
        inpdata =  data.float().to(device)
        nframe = data.size()[1]
        # n_prime = self.n_prime
        if opt.threshold is None:
            n_prime = self.n_prime
        else:
            n_prime = 2 # atleast 2 speakers in training
        
        print('starting cluster: ',nframe)

        phi_range = [0.3,0.4,0.5,0.6,0.7]
        phi_count = 2
        
        # threshold = phi_range[phi_count]
        threshold = None
        
        max_spks = n_prime
        period0len = n_prime

        current_lr = opt.lr
        labelfull_feed=np.arange(nframe)
        clusterlen_feed=[1]*len(labelfull_feed)
        target = 1
        affinity_init,plda_init,PCA_transform = model_init.compute_plda_affinity_matrix(self.pldamodel,inpdata,target=target) # original filewise PCA transform
        
        pca_dim = PCA_transform.shape[0]
        net = Deep_Ahc_model(self.pldamodel,plda_init,dimension=self.xvecD,red_dimension=pca_dim,device=device)
        model = net.to(device)
        
        # Optimizer
        optimizer = optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=opt.lr)
        
        
        model.eval()
        # initialize filewise PCA using svd                  
        if not pretrain:
            # PCA_transform = model_init.compute_filewise_PCAtransform(self.pldamodel,inpdata) # original filewise PCA transform
            model.init_weights(PCA_transform)
            output_model = model(inpdata)
            # output_model = model_init.compute_plda_affinity_matrix(self.pldamodel,inpdata)
            output_model1 = output_model.detach().cpu().numpy()[0]
            output_model = output_model1.copy()
            pldafold = 'plda_pca_baseline/{}_scores/plda_scores/'.format(dataset)
           
            pldafile = '{}/{}.npy'.format(pldafold,f)
           
            if not os.path.isfile(pldafile):
                np.save(pldafile,output_model) 
            
        else:
            model = model_init
            output_model = model(inpdata)
            output_model1 = output_model.detach().cpu().numpy()[0]
            output_model = output_model1.copy()
        labelfull_feed=np.arange(output_model.shape[0])
        clusterlen_feed=[1]*len(labelfull_feed)    
        
        if opt.threshold is None:
            print('DER before training with n_clusters:',n_prime)
        else:
            print('DER before training with threshold:',opt.threshold)
            
        _,_=self.validate_path_integral(output_model1, count,n_prime,clusterlen_feed,labelfull_feed,2)
        
        count +=1
        
        clusterlen_old = clusterlen_feed.copy()
        labelfull_old = labelfull_feed.copy()
        n_clusters = max(period0len,n_prime)
        # n_clusters = 2

        distance_matrix = 1/(1+np.exp(-output_model))#(output_model+1)/2
        # distance_matrix = output_model.copy()
        final_k = min(self.K, nframe - 1)
        if 'dihard' in opt.dataset:
            mypic =pic.PIC_dihard_threshold(n_clusters,clusterlen_old,labelfull_old,distance_matrix.copy(),threshold,K=final_k,z=self.z) 
        else:
            mypic =pic.PIC_ami_threshold(n_clusters,clusterlen_old,labelfull_old,distance_matrix.copy(),threshold,K=final_k,z=self.z) 

        if threshold is not None:
            labelfull,clusterlen,W = mypic.gacCluster_org() 
        else:
            labelfull,clusterlen,W = mypic.gacCluster_oracle_org()
        n_clusters = len(clusterlen)
        cluster = self.compute_cluster(labelfull)
        phi_count = min(0,phi_count - 1)
        n_clusters = len(clusterlen)
        
        print('\n training at {} clusters \n'.format(n_clusters))
        
        
        
        if n_prime==1:
            print('not enough confident labels, skip training')
            matrixfold = "%s/plda_scores/" % (opt.outf)
            savedict = {}
            savedict['output'] = output_model
            if not os.path.isdir(matrixfold):
                os.makedirs(matrixfold)
            # save in pickle
            matrixfile = matrixfold + '/'+f+'.pkl'
            with open(matrixfile,'wb') as sf:
                  pickle.dump(savedict,sf)
            return
        # with random weights
        model.eval()
        output = model(inpdata)
       
        per_loss = opt.eta
      
        soft =0
        y,target,minibatches,pos_weight = self.compute_bce_scores_loss(output[0],labelfull,clean_ind,soft=soft)
        self.sigmoid = nn.Sigmoid()
        if soft:
            self.compute_bce = nn.KLDivLoss(reduction='batchmean')
            self.lsigmoid = nn.LogSigmoid()
            y = self.lsigmoid(y)
        else:
            self.compute_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            

        avg_loss = self.compute_bce(y,target)
        
        print('loss_initial:',avg_loss)
       
        decay = 0.1
        diff_loss = 0
        for epoch in range(opt.epochs):
            decay = 0.1
            
            for m,minibatch in enumerate(minibatches):
                # tot_avg_loss = 0

                model.train()
                model.zero_grad()
                optimizer.zero_grad()
                
                out_train = model(inpdata)
                if len(clean_ind) > 0:
                    y = out_train[0][np.ix_(clean_ind,clean_ind)][np.triu_indices(nframe_clean,k=1)]
                else: 
                    y = out_train[0][np.triu_indices(nframe,k=1)]
                if soft:
                    y = self.lsigmoid(y)
                y_m = y[minibatch]
                target_m = target[minibatch]
                
                triplet_avg_loss = self.compute_bce(y_m,target_m)
            
                triplet_avg_loss.backward()
                optimizer.step()
                
            model.eval()
            if epoch ==0:
                prev_loss = avg_loss
                
            out_val = model(inpdata)
            
            if len(clean_ind) > 0:
                    y = out_val[0][np.ix_(clean_ind,clean_ind)][np.triu_indices(nframe_clean,k=1)]
            else: 
                    y = out_val[0][np.triu_indices(nframe,k=1)]
            if soft:
                y = self.lsigmoid(y)
            
            tot_avg_loss = self.compute_bce(y,target)
            
            diff_loss = prev_loss - tot_avg_loss
            prev_loss = tot_avg_loss
           
            print("\n[epoch %d]  current_lr: %.5f  tot_avg_loss: %.5f " % (epoch+1,current_lr,tot_avg_loss))
            if tot_avg_loss < per_loss*avg_loss or (diff_loss< 1e-3 and diff_loss> 0) or current_lr < 1e-6:
                break


        
        self.final = 1
        model.eval()
        output1 = model(inpdata)
        output1 = output1.cpu().detach().numpy()
        output = output1[0]
        if opt.threshold is None:
            print('System DER with n_clusters:',n_prime)
        else:
            print('System DER with threshold:',opt.threshold)
        valcluster,val_label=self.validate_path_integral(output, count,n_prime,clusterlen_feed,labelfull_feed,1)
        
        count +=1

        print('Saving learnt parameters')
        matrixfold = "%s/plda_scores/" % (opt.outf)
        savedict = {}
        savedict['output'] = output
        if not os.path.isdir(matrixfold):
            os.makedirs(matrixfold)
        # save in pickle
        matrixfile = matrixfold + '/'+f+'.pkl'
        with open(matrixfile,'wb') as sf:
              pickle.dump(savedict,sf)

        if not os.path.isdir(opt.outf+'/models/'):
            os.makedirs(opt.outf+'/models/')
        torch.save(model.state_dict(),opt.outf+'/models/'+f+'.pth')
    
    def train_with_ahc_binary_entropy(self,model_init,pretrain=0,target_energy=target_energy):
        """
        Train the SSC using N* clusters using PIC

        Parameters
        ----------
       Initial model

        Returns
        -------
        None.

        """
       
        alpha = loss_lamda # final stage
        count = 0
        f = self.fname
        data = self.data
        
        print('---------------------------------------------------------')
        print('\nfilename:',f)
        count +=1
        inpdata =  data.float().to(device)
        nframe = data.size()[1]
        # n_prime = self.n_prime
        if opt.threshold is None:
            n_prime = self.n_prime
        else:
            n_prime = 2 # atleast 2 speakers in training
        
        print('starting cluster: ',nframe)

        phi_range = [-0.5, 0.0,0.1,0.2,]
        phi_count = 0
        
        threshold = phi_range[phi_count]
        # threshold = None
        
        max_spks = n_prime
        period0len = n_prime

        current_lr = opt.lr
        labelfull_feed=np.arange(nframe)
        clusterlen_feed=[1]*len(labelfull_feed)
        # target = 1
        affinity_init,plda_init,PCA_transform = model_init.compute_plda_affinity_matrix(self.pldamodel,inpdata,target=target_energy) # original filewise PCA transform
        
        pca_dim = PCA_transform.shape[0]
        net = Deep_Ahc_model(self.pldamodel,plda_init,dimension=self.xvecD,red_dimension=pca_dim,device=device)
        model = net.to(device)
        
        # Optimizer
        optimizer = optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=opt.lr)
        
        # self.compute_bce = nn.BCEWithLogitsLoss()
        
        model.eval()
        # initialize filewise PCA using svd                  
        if not pretrain:
            # PCA_transform = model_init.compute_filewise_PCAtransform(self.pldamodel,inpdata) # original filewise PCA transform
            model.init_weights(PCA_transform)
            output_model = model(inpdata)
            # output_model = model_init.compute_plda_affinity_matrix(self.pldamodel,inpdata)
            output_model1 = output_model.detach().cpu().numpy()[0]
            output_model = output_model1.copy()

            pldafold = 'plda_pca_baseline/{}_scores/plda_scores/'.format(dataset)
           
            pldafile = '{}/{}.npy'.format(pldafold,f)
           
            if not os.path.isfile(pldafile):
                np.save(pldafile,output_model) 
        else:
            model = model_init
            output_model = model(inpdata)
            output_model1 = output_model.detach().cpu().numpy()[0]
            output_model = output_model1.copy()
        labelfull_feed=np.arange(output_model.shape[0])
        clusterlen_feed=[1]*len(labelfull_feed)    
        
        if opt.threshold is None:
            print('DER before training with n_clusters:',n_prime)
        else:
            print('DER before training with threshold:',opt.threshold)
            
        _,_=self.validate(output_model1, count,n_prime,clusterlen_feed,labelfull_feed,2)
        
        count +=1
        
        clusterlen_old = clusterlen_feed.copy()
        labelfull_old = labelfull_feed.copy()
        n_clusters = max(period0len,n_prime)
        # n_clusters = 2
        distance_matrix = output_model.copy()
        clus =ahc.clustering(n_clusters,clusterlen_old,labelfull_old,dist=threshold)
        labelfull,clusterlen = clus.Ahc_full(distance_matrix)
        
        n_clusters = len(clusterlen)
        cluster = self.compute_cluster(labelfull)
        phi_count = min(0,phi_count - 1)
        n_clusters = len(clusterlen)
        
        print('training at {} clusters'.format(n_clusters))
        
        if n_prime==1:
            print('not enough confident labels, skip training')
            matrixfold = "%s/plda_scores/" % (opt.outf)
            savedict = {}
            savedict['output'] = output_model
            if not os.path.isdir(matrixfold):
                os.makedirs(matrixfold)
            # save in pickle
            matrixfile = matrixfold + '/'+f+'.pkl'
            with open(matrixfile,'wb') as sf:
                  pickle.dump(savedict,sf)
            return
        # with random weights
        model.eval()
        output = model(inpdata)
       
        per_loss = opt.eta
                  
        y,target,minibatches,pos_weight = self.compute_bce_scores_loss(output[0],labelfull)
        
        self.sigmoid = nn.Sigmoid()
        self.compute_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        avg_loss = self.compute_bce(y,target)
      
        print('loss_initial:',avg_loss)
       
        decay = 0.1
        diff_loss = 0
        for epoch in range(opt.epochs):
            decay = 0.1
           
            for m,minibatch in enumerate(minibatches):
                # tot_avg_loss = 0

                model.train()
                model.zero_grad()
                optimizer.zero_grad()
                
                out_train = model(inpdata)
                y = out_train[0][np.triu_indices(nframe,k=1)]
                y_m = y[minibatch]
                target_m = target[minibatch]
                
                triplet_avg_loss = self.compute_bce(y_m,target_m)
               
                triplet_avg_loss.backward()
                optimizer.step()
               
            model.eval()
            if epoch ==0:
                prev_loss = avg_loss
                
            out_val = model(inpdata)
            y = out_val[0][np.triu_indices(nframe,k=1)]            
            tot_avg_loss = self.compute_bce(y,target)
            diff_loss = prev_loss - tot_avg_loss
            prev_loss = tot_avg_loss
           
            print("\n[epoch %d]  current_lr: %.5f  tot_avg_loss: %.5f " % (epoch+1,current_lr,tot_avg_loss))
            if tot_avg_loss < per_loss*avg_loss or (diff_loss< 1e-3 and diff_loss> 0) or current_lr < 1e-6:
                break


       
        self.final = 1
        model.eval()
        output1 = model(inpdata)
        output1 = output1.cpu().detach().numpy()
        output = output1[0]
        if opt.threshold is None:
            print('System DER with n_clusters:',n_prime)
        else:
            print('System DER with threshold:',opt.threshold)
        valcluster,val_label=self.validate(output, count,n_prime,clusterlen_feed,labelfull_feed,1)
        
        count +=1

        print('Saving learnt parameters')
        matrixfold = "%s/plda_scores/" % (opt.outf)
        savedict = {}
        savedict['output'] = output
        if not os.path.isdir(matrixfold):
            os.makedirs(matrixfold)
        # save in pickle
        matrixfile = matrixfold + '/'+f+'.pkl'
        with open(matrixfile,'wb') as sf:
              pickle.dump(savedict,sf)

        if not os.path.isdir(opt.outf+'/models/'):
            os.makedirs(opt.outf+'/models/')
        torch.save(model.state_dict(),opt.outf+'/models/'+f+'.pth')
    
    def validate_path_integral(self,output_new, period,n_clusters,clusterlen,labelfull,flag):
            f = self.fname
            overlap =1
            N = len(labelfull)
            clusterlen_org = clusterlen.copy()
            nframe = output_new.shape[0]
           
            distance_matrix = 1/(1+np.exp(-output_new))
    
            if opt.threshold != None:
                 n_clusters = 1
            final_k = min(self.K, nframe - 1) 
            if 'dihard' in opt.dataset:
                mypic =pic.PIC_dihard_threshold(n_clusters,clusterlen_org,labelfull,distance_matrix.copy(),opt.threshold,K=final_k,z=self.z) 
            else:
                mypic =pic.PIC_ami_threshold(n_clusters,clusterlen_org,labelfull,distance_matrix.copy(),opt.threshold,K=final_k,z=self.z) 

            if N < 200 or flag ==2:
                if opt.threshold == None:
                    labelfull,clusterlen = mypic.gacCluster_oracle_org()
                else:
                    labelfull,clusterlen = mypic.gacCluster_org()
            else:
                if opt.threshold == None:
                    labelfull,clusterlen= mypic.gacCluster_oracle()
                else:
                    labelfull,clusterlen = mypic.gacCluster()
            
            n_clusters=len(clusterlen)
            print('clusterlen:',clusterlen, 'n_clusters:',n_clusters)
            self.results_dict[f]=labelfull
            if self.final:
                if self.forcing_label:
                    out_file=opt.outf+'/'+'final_rttms_forced_labels/'
                else:
                    out_file=opt.outf+'/'+'final_rttms/'
            else:
                out_file=opt.outf+'/'+'rttms/'
            if not os.path.isdir(out_file):
                os.makedirs(out_file)
            outpath=out_file +'/'+f
            rttm_newfile=out_file+'/'+f+'.rttm'
            
            rttm_gndfile = opt.rttm_ground_path+'/'+f+'.rttm'
            self.write_results_dict(out_file)
            # bp()
            der=self.compute_score(rttm_gndfile,rttm_newfile,outpath,0)
            if overlap:
                der = self.compute_score(rttm_gndfile,rttm_newfile,outpath,overlap)
            print("\n%s [period %d] DER: %.2f" % (self.fname,period, der))
            cluster = self.compute_cluster(labelfull)

            return cluster,labelfull
        
    def validate(self,output_new, period,n_clusters,clusterlen,labelfull,flag):
            # lamda = 0
            # bp()
            f = self.fname
            overlap =1
            clusterlen_org = clusterlen.copy()
            if opt.threshold == None or self.final==0:
                myahc =ahc.clustering(n_clusters, clusterlen_org,labelfull,dist=None)
            else:
                myahc =ahc.clustering(None, clusterlen_org,labelfull,dist=float(opt.threshold))
            labelfull,clusterlen = myahc.Ahc_full(output_new)
            print('clusterlen:',clusterlen)
            self.results_dict[f]=labelfull
            if self.final:
                if self.forcing_label:
                    out_file=opt.outf+'/'+'final_rttms_forced_labels/'
                else:
                    out_file=opt.outf+'/'+'final_rttms/'
            else:
                out_file=opt.outf+'/'+'rttms/'
            if not os.path.isdir(out_file):
                os.makedirs(out_file)
            outpath=out_file +'/'+f
            rttm_newfile=out_file+'/'+f+'.rttm'
            rttm_gndfile = opt.rttm_ground_path+'/'+f+'.rttm'
            # rttm_gndfile = 'rttm_ground/'+f+'.rttm'
            self.write_results_dict(out_file)
            # bp()
            der=self.compute_score(rttm_gndfile,rttm_newfile,outpath,0)
            if overlap:
                der = self.compute_score(rttm_gndfile,rttm_newfile,outpath,overlap)
            
            print("\n%s [period %d] DER: %.2f" % (self.fname,period, der))
           
            cluster = self.compute_cluster(labelfull)

            return cluster,labelfull

def main():
    seed=555
    

    if "dihard" in opt.dataset:
        target_energy = 1
        xvecD=512
        pca_dim=30
    elif "AMI" in opt.band:
        target_energy = 0
        xvecD=512
        pca_dim=30
    else:
        print("only dihard and AMI are parameters can be set")
        return
    pair_list = open(opt.reco2utt_list).readlines()
    filelen =len(pair_list)
    if opt.reco2num_spk !=None:
        reco2num = open(opt.reco2num_spk).readlines()
    else:
        reco2num = "None"
        
    kaldimodel = pickle.load(open(opt.kaldimodel,'rb')) # PCA Transform and mean of heldout set
    ind = list(np.arange(filelen))
    random.shuffle(ind)    

    for i in range(filelen):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        print('pca_dim:',pca_dim)
        net_init = weight_initialization(kaldimodel,dimension=xvecD,pca_dimension=pca_dim,device=device)
        model_init = net_init.to(device)
              
        # training
        reco2utt = pair_list[i]
        fname = reco2utt.split()[0]
        
        matrixfold = "%s/plda_scores/" % (opt.outf)
        matrixfile = matrixfold + '/'+fname+'.pkl'
        if os.path.isfile(matrixfile):
            continue
        if reco2num != "None":
            n_prime = int(reco2num[i].split()[1])
            if n_prime == 0:
                cosinefold = 'plda_pca_baseline/{}_scores/plda_scores/'.format(dataset)
                cosinefile = '{}/{}.npy'.format(cosinefold,fname)
                if os.path.isfile(cosinefile):
                    output = np.load(cosinefile)
                    savedict = {}
                    savedict['output'] = output
                    if not os.path.isdir(matrixfold):
                        os.makedirs(matrixfold)
                    # save in pickle
                    with open(matrixfile,'wb') as sf:
                          pickle.dump(savedict,sf)
                    continue
        else:
            n_prime = 2 # needs atleast 2 clusters
        
        
        print('output_folder:',opt.outf)
        ahc_obj = Deep_AHC(kaldimodel,fname,reco2utt,xvecD,n_prime)
        filelength = ahc_obj.dataloader_from_list()
        if opt.clustering == 'pic':             
            ahc_obj.train_with_selective_binary_entropy(model_init,pretrain=0,target_energy=target_energy)
        else:
            ahc_obj.train_with_ahc_binary_entropy(model_init,pretrain=0,target_energy=target_energy)

        print('output_folder:',opt.outf)


if __name__ == "__main__":
    main()