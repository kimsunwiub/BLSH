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
    """
    s: Source signal
    n: Noise signal
    x: s + n
    sr: Reconstructed signal (or some signal to evaluate against)
    """
    ml = np.int(np.minimum(len(s), len(sr)))
    source = np.array(s[:ml])[:,None].T
    noise = np.array(n[:ml])[:,None].T
    sourceR = np.array(sr[:ml])[:,None].T
    noiseR = np.array(x[:ml]-sr[:ml])[:,None].T
    sdr,sir,sar,_=mir_eval.separation.bss_eval_sources(
            np.concatenate((source, noise),0),
            np.concatenate((sourceR, noiseR),0), 
            compute_permutation=False)   
    # Take the first element from list for source's performance
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
    # For debugging purposes
    temp = torch.mm(input_data, p_m)
    output = torch.tanh(temp)
    bssm = torch.mm(output, output.t())
    bssm = (bssm+1)/2
    return bssm

def bssm_sign(input_data, p_m): #TODO BIAS
    temp = torch.mm(input_data, p_m) # 1000 x 513. dot 513 x 1
    signbnnh1=signBNN()
    output = signbnnh1(temp) # 1000 x 1
    bssm = torch.mm(output, output.t()) # 1000 x 1000
    bssm = (bssm+1)/2
    return bssm

def xent_fn(p, q):
    ret_xent = -(
                (p+1e-20)*torch.log(q+1e-20) \
                + (1-p+1e-20) * torch.log(1-q+1e-20)
            )
    return ret_xent

def np_xent_fn(p, q):
    ret_xent = -(
                (p+1e-20)*np.log(q+1e-20) \
                + (1-p+1e-20) * np.log(1-q+1e-20)
            )
    return ret_xent

def validate(Xva, args, p_m):
    # Validation
    epoch_losses = []
    for i in range(0, len(Xva), args.segment_len):
        Xva_seg = torch.cuda.FloatTensor(Xva[i:i+args.segment_len])
        ssm_va = torch.mm(Xva_seg, Xva_seg.t())
        ssm_va /= ssm_va.max()
        Xva_seg_bias = torch.cat((Xva_seg, torch.ones((len(Xva_seg),1)).cuda()), 1)
        bssm = bssm_sign(Xva_seg_bias, p_m)
        loss_va = ((bssm-ssm_va)**2).mean()
        epoch_losses.append(float(loss_va))
    return np.mean(epoch_losses)

def get_stats(X):
    return X.min(), X.max(), X.mean(), X.std()

def get_beta(Xtr, wi, p_m, args):
    beta_m = []
    for i in range(0, len(Xtr), args.segment_len):
        Xtr_seg = torch.cuda.FloatTensor(Xtr[i:i+args.segment_len])
        wi_seg = torch.cuda.FloatTensor(wi[i:i+args.segment_len])

        # Create SSM with a kernel
        if args.kernel == "linear":
            ssm_tr = torch.mm(Xtr_seg, Xtr_seg.t())
        elif args.kernel == "rbf":
            ssm_tr = torch.exp(torch.mm(Xtr_seg, Xtr_seg.t()) / args.sigma2)

        Xtr_seg_bias = torch.cat((Xtr_seg, torch.ones((len(Xtr_seg),1)).cuda()), 1)
        
        if args.debug_option > 2:
            bssm = bssm_tanh(Xtr_seg_bias, p_m) 
        else:
            bssm = bssm_sign(Xtr_seg_bias, p_m)
            
        # Backprop with weighted sum of errors
        sqerr = (bssm-ssm_tr)**2
        e_t = (sqerr * wi_seg).sum()
        e_t = pt_to_np(e_t)
        beta_i = 0.5 * np.log((1-e_t)/e_t)
        beta_m.append(beta_i)
    return np.mean(beta_m)