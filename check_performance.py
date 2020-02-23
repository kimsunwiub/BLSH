from argparse import ArgumentParser
import numpy as np
import librosa
import pickle
import torch
import os

from utils import SDR, get_mir_scores, load_pkl

def parse_arguments():
    parser = ArgumentParser()
    
    parser.add_argument("-o", "--is_oracle", action='store_true',
                        help="kNN on original features")
    parser.add_argument("-p", "--is_proj", action='store_true',
                        help="kNN on projections")
    parser.add_argument("--use_stft", action='store_true',
                        help="Using STFT features")
    parser.add_argument("--use_mel", action='store_true',
                        help="Using Mel features")
    parser.add_argument("--use_mfcc", action='store_true',
                        help="Using MFCC features")
    parser.add_argument("--is_closed", action='store_true',
                        help="Open (Test) / Closed (Val)")
    parser.add_argument("--use_kernel", action='store_true',
                        help="RBF Kernel for comparison")
    parser.add_argument("--sigma2", type=float, default=1.0,
                        help = "Width of RBF kernel")
    
    parser.add_argument("--load_model", type=str, default=None,
                        help="Trained projections")    
    parser.add_argument("-n", "--n_proj", type=int, default=200,
                        help = "Number of projections")
    parser.add_argument("--use_perc", type=float, default=1.0,
                        help = "Random sample %% of training set")
    parser.add_argument("-k", "--K", type=int, default=10,
                        help = "Number of neighbors")
    parser.add_argument("--seed", type=int, default=42,
                        help = "Seed for random sampling from dictionary")
    parser.add_argument("--gpu_id", type=int, default=-1,
                        help = "GPU ID. -1 for ")
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    print (args.n_proj, args.seed, args.is_closed, args.load_model, args.gpu_id, args.use_perc, args.is_closed)
    if args.gpu_id != -1:
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu_id)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    if args.use_stft:
        Xtr_load_nm = "Xtr_STFT.npy"
        Xva_load_nm = "Xva_STFT.pkl"
        Xte_load_nm = "Xte_STFT.pkl"
    elif args.use_mel:
        Xtr_load_nm = "Xtr_Mel.npy"
        Xva_load_nm = "Xva_Mel.pkl"
        Xte_load_nm = "Xte_Mel.pkl"
    elif args.use_mfcc:
        Xtr_load_nm = "Xtr_MFCC.npy"
        Xva_load_nm = "Xva_MFCC.pkl"
        Xte_load_nm = "Xte_MFCC.pkl"
        
    # Load training dictionary
    Xtr = np.load(Xtr_load_nm)
    Ytr = np.load("IBM_STFT.npy")
    
    # Random sampling
    np.random.seed(args.seed)
    perm_idx = np.random.permutation(len(Xtr))[:int(len(Xtr) * args.use_perc)]
    Xtr = Xtr[perm_idx]
    Ytr = Ytr[perm_idx]
    
    # Load testing data
    if args.is_closed:
        print ("Loading closed set")
        
        Xva = load_pkl(Xva_load_nm)
        vaX = load_pkl("vaX_STFT.pkl")
        with open('vasnx_wavefiles.pkl', 'rb') as handle:
            val_waves_dict = pickle.load(handle)
            
    else:   
        print ("Loading open set")
            
        Xva = load_pkl(Xte_load_nm)
        vaX = load_pkl("teX_STFT.pkl")
        with open('tesnx_wavefiles.pkl', 'rb') as handle:
            val_waves_dict = pickle.load(handle)
            
    vas = val_waves_dict['s']
    van = val_waves_dict['n']
    vax = val_waves_dict['x']
    
    # Load projections and apply
    if args.is_proj:
        print ("Loading projections...")
        if args.load_model:
            projections = np.load(
                "{}_projs.npy".format(args.load_model))
            projections = projections.squeeze().T
            projections = projections[:,:args.n_proj]
        else:
            # LSH Random projection baseline
            np.random.seed(args.seed)
            # projections = np.random.rand(Xtr.shape[1], args.n_proj)
            projections = np.random.normal(loc=0.0, 
                            scale=1./args.n_proj, 
                            size=(Xtr.shape[1], args.n_proj))  
            
        if args.gpu_id != -1:
            Xtr = torch.cuda.FloatTensor(Xtr)
            projections = torch.cuda.FloatTensor(projections)
            applied_tr = torch.sign(Xtr.mm(projections))
        else:
            applied_tr = np.sign(np.dot(Xtr, projections))
        
    else:
        # Oracle kNN baseline
        if args.gpu_id != -1:
            applied_tr = torch.cuda.FloatTensor(Xtr)
        else:
            applied_tr = Xtr
        
    # Get scores
    N_va = len(Xva)
    SDRlist=np.zeros(N_va)
    ml=np.zeros(N_va)
    mirSDRlist=np.zeros(N_va)
    mirSIRlist=np.zeros(N_va)
    mirSARlist=np.zeros(N_va)
    
    for i in range(N_va):
        # Apply projections on validation data
        if args.is_proj:
            if args.gpu_id != -1:
                Xva_i = torch.cuda.FloatTensor(Xva[i])
                applied_vate = torch.sign(Xva_i.mm(projections))
            else:
                applied_vate = np.sign(np.dot(Xva[i], projections))
        else:
            if args.gpu_id != -1:
                applied_vate = torch.cuda.FloatTensor(Xva[i])
            else:
                applied_vate = Xva[i]
        
        # Compute cosine similarity scores
        if args.gpu_id != -1:
            scores = applied_tr.mm(applied_vate.t())
            scores = scores.detach().cpu().numpy()
        else:
            scores = np.dot(applied_tr, applied_vate.T)
        if args.use_kernel:
            scores = np.exp(scores/args.sigma2)
        if args.is_proj:
            scores = ((scores+args.n_proj)/2)
        
        # Apply average of K IBMs
        K_locs = np.argpartition(-scores, args.K, 0)[:args.K]
        Yhat = np.mean(Ytr[K_locs],0)
        applied = vaX[i] * Yhat
        recon = librosa.istft(applied.T, hop_length=256)

        ml[i], SDRlist[i] = SDR(vas[i], recon)
        msdr, msir, msar = get_mir_scores(
            vas[i], van[i], vax[i], recon)
        mirSDRlist[i] = msdr
        mirSIRlist[i] = msir
        mirSARlist[i] = msar

        # Print every 100 signals
        if i % 100 == 0:
            curr_tot_SDR = np.sum(ml*SDRlist/np.sum(ml))
            curr_tot_mSDR = np.sum(ml*mirSDRlist/np.sum(ml))
            curr_tot_mSIR = np.sum(ml*mirSIRlist/np.sum(ml))
            curr_tot_mSAR = np.sum(ml*mirSARlist/np.sum(ml))

            prog = i/len(vas)*100

            print ("{}: {:.2f} SDR {:.2f} mSDR {:.2f} mSIR {:.2f} mSAR {:.2f}"
                   .format(i, prog, curr_tot_SDR, curr_tot_mSDR, 
                           curr_tot_mSIR, curr_tot_mSAR))
    
    # Save results
    result_dict = {
        'sdr': SDRlist,
        'msdr': mirSDRlist,
        'msir': mirSIRlist,
        'msar': mirSARlist,
        'ml': ml
    }
    
    loaded = 0
    if args.load_model:
        loaded = 1
        
    test_nm = "results_[{}|{}]_proj[n={}]_feat[{}]_seed[{}]_perc[{}]".format(args.is_proj, 
                                                                     loaded, args.n_proj, 
                                                                    Xtr_load_nm.split('.')[0], args.seed,
                                                                            int(args.use_perc*100))
#     test_nm = "AdaRes_kTyp{}{}_K{}_feat{}{}{}_nproj{}_perc{}".format(
#         int(args.is_oracle), int(args.is_proj), args.K, 
#         int(args.use_stft), int(args.use_mel), int(args.use_mfcc),
#         args.n_proj, int(args.use_perc*100))
    
    if args.is_closed:
        test_nm += "CLOSED"
        print ("SAVING {}".format(test_nm))
    
    with open('{}.pkl'.format(test_nm), 'wb') as handle:
            pickle.dump(result_dict, handle)
    print ("Results saved")
    
if __name__ == "__main__":
    main()
