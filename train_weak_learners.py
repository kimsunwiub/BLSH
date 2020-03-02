from argparse import ArgumentParser
import numpy as np
import pickle
import time
import os

import torch
import torch.nn as nn
from torch.autograd import Variable

from utils import pt_to_np, signBNN, bssm_tanh, bssm_sign, xent_fn, train_pm_xent

import warnings # TODO: Replace deprecated fn.
warnings.filterwarnings("ignore")

def parse_arguments():
    parser = ArgumentParser()

    parser.add_argument("--use_stft", action='store_true', 
                        help="Using STFT features")
    parser.add_argument("--use_mel", action='store_true',
                        help="Using Mel features")
    parser.add_argument("--use_mfcc", action='store_true',
                        help="Using MFCC features")
    parser.add_argument("--seed", type=int, default=42,
                        help = "Data: Seed for train and test speaker selection")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help = "GPU_ID")
    parser.add_argument("--load_model", type=str, default=None,
                        help="Load previous permutations and results")
    parser.add_argument("--segment_len", type=int, default=1000,
                        help = "Segment length: ")
    parser.add_argument("--num_iters_pm", type=int, default=300,
                        help = "Num iteration to train random projections")
    parser.add_argument("--num_proj", type=int, default=200,
                        help = "Number of projections")
    parser.add_argument("--ssmD", type=str, default='mse', 
                        help="Distance for SSM. Options: mse, xent")
    parser.add_argument("--kernel", type=str, default='cosine', 
                        help="Kernel for SSM. Options: cosine, rbf")
    parser.add_argument("--sigma2", type=float, default=0.0, 
                        help="Denominator value for RBF kernel")
    parser.add_argument("--save_every", type=int, default=10,
                        help = "Specify saving frequency")
    parser.add_argument("--save_wi", action='store_true',
                        help="Debugging option to save example weights")
    return parser.parse_args()

def main():
    args = parse_arguments()
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu_id)
    np.random.seed(args.seed)

    if args.use_stft:
        Xtr_load_nm = "Xtr_STFT.npy"
    elif args.use_mel:
        Xtr_load_nm = "Xtr_Mel.npy"
    elif args.use_mfcc:
        Xtr_load_nm = "Xtr_MFCC.npy"
        
    # Load training dictionary
    Xtr = np.load(Xtr_load_nm)
    segment_len = 1000
    Ntr, n_features = Xtr.shape
    
    ## --- Training params ---
    np.random.seed(42)
    Xtr_shuffled = Xtr[np.random.permutation(len(Xtr))]
    
    # Load previous model if given
    m_start = 0
    if args.load_model:
        projections = list(np.load("{}_projs.npy".format(args.load_model)))
        wip1 = np.load("{}_wip1.npy".format(args.load_model))
#         betas = list(np.load("{}_betas.npy".format(args.load_model)))
        betas = []
        m_start = len(projections)
        model_nm = args.load_model
    else:
        wip1 = np.ones((Ntr,segment_len), dtype=np.float32)
        wip1_init_val = np.float32(1/(segment_len*segment_len))
        wip1 *= wip1_init_val
        print ("Observation weights\n\tShape: {}\n\tValues: {}".format(
            wip1.shape, wip1[0,0]))
        projections = []
        betas = []
        model_nm = "proj[n={}]_feat[{}]_ssmD[{}]_kern[{}|{}]".format(
            len(projections), Xtr_load_nm.split('.')[0], args.ssmD, 
            args.kernel, '_'.join(str(args.sigma2).split('.')))
    print ("Starting {}...".format(model_nm))
        
    # Training
    learningRate=1e-2
    for m in range(m_start, args.num_proj + m_start):
        # Init Training
        wi = wip1
        wip1 = np.zeros(wi.shape, dtype=np.float32)
        p_m = torch.Tensor(n_features, 1)
        p_m = Variable(torch.nn.init.xavier_normal_(p_m).cuda(), requires_grad=True)
        optimizer = torch.optim.Adam([p_m], betas=[0.95, 0.98], lr=learningRate)

        # Train projections and Adaboost
        toc = time.time()
        for i in range(0,len(Xtr_shuffled),segment_len):
            if i % (segment_len*10) == 0:
                print ("m={} Progress {:.2f}%".format(m, 100*i/len(Xtr_shuffled)))
            Xtr_seg = torch.cuda.FloatTensor(Xtr_shuffled[i:i+segment_len])
            if len(Xtr_seg) < segment_len:
                print (
                    "Xtr_seg length: {}. Break.".format(len(Xtr_seg)))
                break
            wi_seg = torch.cuda.FloatTensor(wi[i:i+segment_len])
            if len(wi_seg) < segment_len:
                print (
                    "wi_seg length: {}. Break.".format(len(wi_seg)))
                break
                
            # Create SSM with a kernel
            if args.kernel == "cosine":
                ssm = torch.mm(Xtr_seg, Xtr_seg.t())
            elif args.kernel == "rbf":
                ssm = torch.exp(torch.mm(Xtr_seg, Xtr_seg.t()) / args.sigma2)
            else:
                print ("Check --kernel option...")
                return -1
            ssm /= ssm.max()
            
            # Train the weak learner
            p_m, ssm_hat = train_pm_xent(
                wi_seg, p_m, Xtr_seg, ssm, optimizer, args.num_iters_pm)
            bssm = bssm_sign(Xtr_seg, p_m)
            
            # Compute distance between BSSM and SSM
            if args.ssmD == "mse":
                sse = (bssm-ssm)**2
            elif args.ssmD == "xent":
                sse = xent_fn(bssm,ssm)
            else:
                print ("Check --ssmD option...")
                return -1
            sse_div = sse/sse.max()
            
            # Updating weak learner weight and observation weights
            e_t = (sse_div*wi_seg).sum()
            if e_t == 0.0:
                print ("e_t reached 0")
                return -1
            beta_m_log = 0.5 * torch.log((1-e_t)/e_t)
            wip1[i:i+segment_len] = pt_to_np(
                wi_seg * torch.exp(beta_m_log * sse_div))
            
        wip1 /= wip1.sum()
        tic = time.time()
        print ("Time: Learning projection: {:.2f}".format(tic-toc))
        projections.append(p_m.detach().cpu().numpy())
        betas.append(beta_m_log.detach().cpu().numpy())

        # Saving results
        if (m+1) % args.save_every == 0:
            model_nm = "proj[n={}]_feat[{}]_ssmD[{}]_kern[{}|{}]".format(
                len(projections), Xtr_load_nm.split('.')[0], args.ssmD, 
                args.kernel, '_'.join(str(args.sigma2).split('.')))
            np.save("Ada_Results/{}_projs".format(model_nm), np.array(projections))
            np.save("Ada_Results/{}_betas".format(model_nm), np.array(betas))
            if args.save_wi:
                np.save("Ada_Results/{}_wip1".format(model_nm), wip1)
    
if __name__ == "__main__":
    main()
