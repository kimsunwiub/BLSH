import numpy as np
import mir_eval
import pickle
import torch

# Misc. helper functions
def count_mins(X):
    tot_mins = 0
    for i in range(len(X)):
        tot_mins += len(X[i])
    tot_mins /= (16000 * 60)
    print ("{} mins".format(int(np.round(tot_mins))))
    
def save_pkl(X, name):
    with open(name, 'wb') as handle: 
        pickle.dump({'temp': X}, handle)
    return

def load_pkl(name):
    with open(name, 'rb') as handle: 
        X = pickle.load(handle)['temp']
    return X

def SDR(s,sr):
    eps=1e-20
    ml=np.minimum(len(s), len(sr))
    s=s[:ml]
    sr=sr[:ml]
    return ml, 10*np.log10(np.sum(s**2)/(np.sum((s-sr)**2)+eps)+eps)

def get_mir_scores(s, n, x, sr):
    ml = np.int(np.minimum(len(s), len(sr)))
    source = np.array(s[:ml])[:,None].T
    noise = np.array(n[:ml])[:,None].T
    sourceR = np.array(sr[:ml])[:,None].T
    noiseR = np.array(x[:ml]-sr[:ml])[:,None].T
    sdr,sir,sar,_=mir_eval.separation.bss_eval_sources(
            np.concatenate((source, noise),0),
            np.concatenate((sourceR, noiseR),0), 
            compute_permutation=False)   
    return sdr[0],sir[0],sar[0]

def pt_to_np(X):
    return X.detach().cpu().numpy()

# Helper functions for training weak learners
class signBNN(torch.autograd.Function):
    def forward(self, x):
        self.save_for_backward(x)
        return x.sign()
    def backward(self, grad_y):
        x, =self.saved_tensors
        grad_input = grad_y.mul(1-torch.tanh(x)**2)
        return grad_input

def bssm_tanh(input_data, p_m):
    temp = torch.mm(input_data, p_m)
    output = torch.tanh(temp)
    bssm = torch.mm(output, output.t())
    bssm = (bssm+1)/2
    return bssm

def bssm_sign(input_data, p_m):
    temp = torch.mm(input_data, p_m)
    signbnnh1=signBNN()
    output = signbnnh1(temp)
    bssm = torch.mm(output, output.t())
    bssm = (bssm+1)/2
    return bssm

def xent_fn(p, q):
    ret_xent = -(
                (p+1e-20)*torch.log(q+1e-20) \
                + (1-p+1e-20) * torch.log(1-q+1e-20)
            )
    return ret_xent

def train_pm_xent(wi, p_m, input_data, ssm, optimizer, num_iter):
    for i in range(num_iter):
        bssm = bssm_tanh(input_data, p_m)
        loss = (xent_fn(bssm,ssm)*wi).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return p_m, bssm