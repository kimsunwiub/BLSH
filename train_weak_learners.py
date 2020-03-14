from argparse import ArgumentParser
import numpy as np
import pickle
import time
import os

import torch
import torch.nn as nn
from torch.autograd import Variable

from utils import pt_to_np, signBNN, bssm_tanh, bssm_sign, bssm_sign_nograd, xent_fn, validate, save_pkl, load_pkl, get_beta

import warnings # TODO: Replace deprecated fn.
warnings.filterwarnings("ignore")

def parse_arguments():
    parser = ArgumentParser()

    parser.add_argument("--use_stft", action='store_true', 
                        help="Using STFT features")
    parser.add_argument("--use_log_mel", action='store_true',
                        help="Using log Mel features")
    parser.add_argument("--use_mfcc", action='store_true',
                        help="Using MFCC features")
    parser.add_argument("--seed", type=int, default=42,
                        help = "Data: Seed for speaker selection")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help = "GPU_ID")
    parser.add_argument("--load_model", type=str, default=None,
                        help="Load previous permutations and results")
    parser.add_argument("--segment_len", type=int, default=1000,
                        help = "Segment length: ")
    parser.add_argument("--max_iter", type=float, default=100,
                        help = "Maximum number of iterations for learners")
    parser.add_argument("--lr", type=float, default=1e-4, 
                        help="Learning rate for training weak learners")
    parser.add_argument("--num_proj", type=int, default=100,
                        help = "Number of projections")
    parser.add_argument("--kernel", type=str, default='linear', 
                        help="Kernel for SSM. Options: linear, rbf")
    parser.add_argument("--sigma2", type=float, default=0.0, 
                        help="Denominator value for RBF kernel")
    parser.add_argument("--save_every", type=int, default=25,
                        help = "Specify saving frequency")
    parser.add_argument("--debug", action='store_true',
                        help="Debugging option (tweak lr, save wip1s)")
    parser.add_argument("--use_bias", action='store_true',
                        help="Add bias term to weak learners")
    parser.add_argument("--debug_option", type=int, default=0,
                        help = "1: Normal, 2: Tanh pm, 3: Tanh beta")
    parser.add_argument("--tanhscale", type=float, default=1.0, 
                        help="Tanh scaling for option 2 and 3")
    return parser.parse_args()

def main():
    args = parse_arguments()
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu_id)

    if args.use_stft:
        Xtr_load_nm = "Xtr_STFT.npy"
        Xva_load_nm = "Xva_STFT.npy"
    elif args.use_log_mel:
        Xtr_load_nm = "Xtr_log_Mel.npy"
        Xva_load_nm = "Xva_log_Mel.npy"
    elif args.use_mfcc:
        Xtr_load_nm = "Xtr_MFCC.npy"
        Xva_load_nm = "Xva_MFCC.npy"
    
    # Load training dictionary
    Xtr = np.load(Xtr_load_nm)
    
    # Training datset 
    truncate_len = len(Xtr) % args.segment_len
    np.random.seed(42)
    Xtr_shuffled = Xtr[np.random.permutation(len(Xtr))][:-truncate_len]
    # TODO: Continue shuffling per epoch, or remove it
    Ntr, n_features = Xtr_shuffled.shape
    
    # Validation dataset 
    Xva = np.load(Xva_load_nm)
    truncate_len = len(Xva) % args.segment_len
    Xva = Xva[:-truncate_len]
    
    tol = 1/(args.segment_len**2)
    
    # Load previous model if given
    if args.load_model:
        projections = list(np.load("{}_projs.npy".format(args.load_model)))
        proj_losses = load_pkl("{}_projlosses.pkl".format(args.load_model))
        wip1 = np.load("{}_wip1.npy".format(args.load_model))
        betas = list(np.load("{}_betas.npy".format(args.load_model)))
        m_start = len(projections)
        model_nm = args.load_model
    else:
        m_start = 0
        wip1 = np.ones((Ntr,args.segment_len), dtype=np.float32)
        wip1_init_val = np.float32(1/(args.segment_len*args.segment_len))
        wip1 *= wip1_init_val
        print ("Observation weights\n\tShape: {}\n\tValues: {}".format(
            wip1.shape, wip1[0,0]))
        projections = []
        proj_losses = []
        betas = []
        model_nm = "proj[n={}]_feat[{}]_kern[{}_{}]_lr[{:.0e}]_bias[{}]".format(
            len(projections), Xtr_load_nm.split('.')[0], args.kernel, 
            args.sigma2, args.lr, args.use_bias)
        if args.debug_option > 0:
            model_nm += "_debug[{}|{}]".format(args.debug_option, args.tanhscale)
    print ("Starting {}...".format(model_nm))
        
    
    for m in range(m_start, args.num_proj + m_start):
        # Init Training
        wi = wip1
        if args.use_bias:
            p_m = torch.Tensor(n_features+1, 1)
        else:
            p_m = torch.Tensor(n_features, 1)
        p_m = Variable(
            torch.nn.init.xavier_normal_(p_m).cuda(), requires_grad=True)
        optimizer = torch.optim.Adam([p_m], betas=[0.95, 0.98], lr=args.lr)

        toc = time.time()
        epoch = 0
        lsi = 0 # Losses start index
        tr_losses = []
        for epoch in range(args.max_iter):
            ep_losses = []
            # Training
            for i in range(0, Ntr, args.segment_len):
                Xtr_seg = torch.cuda.FloatTensor(Xtr[i:i+args.segment_len])
                wi_seg = torch.cuda.FloatTensor(wi[i:i+args.segment_len])

                # Create SSM 
                ssm_tr = torch.mm(Xtr_seg, Xtr_seg.t())*2 - 1

                # Train the weak learner  
                if args.use_bias:
                    Xtr_seg_bias = torch.cat((Xtr_seg, torch.ones((len(Xtr_seg),1)).cuda()), 1)
                else:
                    Xtr_seg_bias = Xtr_seg
                    
                if args.debug_option > 1:
                    bssm = bssm_tanh(Xtr_seg_bias, p_m, args.tanhscale) 
                else:
                    bssm = bssm_sign(Xtr_seg_bias, p_m)
                # Backprop with weighted sum of errors
                sqerr = (bssm-ssm_tr)**2
                e_t = (sqerr * wi_seg).sum()
                optimizer.zero_grad()
                e_t.backward()
                optimizer.step()
                e_t = float(e_t)
                ep_losses.append(e_t)
            if args.debug:
                if (epoch+1) % 20 == 0:
                    print("DEBUG: ", m, epoch, np.mean(ep_losses))
            tr_losses.append(np.mean(ep_losses))

        # Update Adaboost parameters at end of training        
        beta = get_beta(Xtr_shuffled, wi, p_m, args)
        wip1 = np.zeros(wi.shape, dtype=np.float32)
        for i in range(0, Ntr, args.segment_len):
            Xtr_seg = torch.cuda.FloatTensor(Xtr[i:i+args.segment_len])
            wi_seg = torch.cuda.FloatTensor(wi[i:i+args.segment_len])

            # Create SSM with a kernel
            ssm_tr = torch.mm(Xtr_seg, Xtr_seg.t())*2 - 1
                
            if args.use_bias:
                Xtr_seg_bias = torch.cat((Xtr_seg, torch.ones((len(Xtr_seg),1)).cuda()), 1)
            else:
                Xtr_seg_bias = Xtr_seg
            
            bssm = bssm_sign_nograd(Xtr_seg_bias, p_m)
            # Backprop with weighted sum of errors
            sqerr = (bssm-ssm_tr)**2 /2 - 1

            wip1[i:i+args.segment_len] = pt_to_np(wi_seg) * np.exp(
                -beta * pt_to_np(sqerr))
            wip1[i:i+args.segment_len] /= wip1[i:i+args.segment_len].sum()

        tic = time.time()
        print ("Time: Learning projection #{}: {:.2f} for {} iterations".format(
            m+1, tic-toc, epoch))
        print ("\tbeta: {:.3f}".format(beta))
        if args.debug:
            print ("DEBUG. tr_losses[::20] = ", tr_losses[::20])
        projections.append(p_m.detach().cpu().numpy())
        proj_losses.append(tr_losses)
        betas.append(beta)

        # Saving results
        if (m+1) % args.save_every == 0 or m < 5:
            model_nm = "proj[n={}]_feat[{}]_kern[{}_{}]_lr[{:.0e}]_bias[{}]".format(
                len(projections), Xtr_load_nm.split('.')[0], args.kernel, 
                args.sigma2, args.lr, args.use_bias)
            if args.debug_option > 0:
                model_nm += "_debug[{}]".format(args.debug_option)
            np.save("Ada_Results/{}_projs".format(model_nm), 
                    np.array(projections))
            save_pkl(proj_losses, 
                     "Ada_Results/{}_projlosses.pkl".format(model_nm))
            np.save("Ada_Results/{}_betas".format(model_nm), np.array(betas))
            np.save("Ada_Results/{}_wip1".format(model_nm), wip1)

    np.save("Ada_Results/{}_wip1".format(model_nm), wip1)

    
if __name__ == "__main__":
    main()

