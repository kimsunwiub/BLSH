import numpy as np
import mir_eval
import pickle

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
