from argparse import ArgumentParser
import pickle
import os
from loader import *

def parse_arguments():
    parser = ArgumentParser()    
    parser.add_argument("--seed", type=int, default=42,
                        help = "Seed for train and test speaker selection")
    
    parser.add_argument("--make_wavefiles", action='store_true',
                        help = "Option to generate wavefiles")
    parser.add_argument("--use_mel", action='store_true',
                        help = "Option to use mel spectrogram")
    parser.add_argument("--use_mfcc", action='store_true',
                        help = "Option to use MFCC")
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu_id)
    np.random.seed(args.seed)
    
    if args.make_wavefiles:
        # Create and save wavefiles for specified number of train and test speakers
        trs, trn, trx, vas, van, vax, _ = setup_ada_training_data(args.seed)
        tes, ten, tex, _ = setup_ada_testing_data(args.seed)
        
        # Save files
        waves_dict = {
            'trs': trs, 'trx': trx, 'trn': trn,
            'vas': vas, 'vax': vax, 'van': van,
            'tes': tes, 'tex': tex, 'ten': ten
        }
        with open('wavefiles_adaboost.pkl', 'wb') as handle:
            pickle.dump(waves_dict, handle)
            
    else:
        # Check if wavefiles have been generated
        if not os.path.exists('wavefiles_adaboost.pkl'):
            print ("Need to generate wavefiles first.")
            return -1
        
        # STFT
        trS = [librosa.stft(x, n_fft=1024, hop_length=256).T for x in trs]
        trN = [librosa.stft(x, n_fft=1024, hop_length=256).T for x in trn]
        trX = [librosa.stft(x, n_fft=1024, hop_length=256).T for x in trx]

        vaX = [librosa.stft(x, n_fft=1024, hop_length=256).T for x in vax]

        teX = [librosa.stft(x, n_fft=1024, hop_length=256).T for x in tex]
        
        # Magnitude
        trS_mag = [np.abs(stft) for stft in trS]
        trN_mag = [np.abs(stft) for stft in trN]
        trX_mag = [np.abs(stft) for stft in trX]

        vaX_mag = [np.abs(stft) for stft in vaX]

        teX_mag = [np.abs(stft) for stft in teX]
        
        # IBM
        IBM = [(trS_mag[i] > trN_mag[i])*1 for i in range(len(trS_mag))]
        IBM = np.concatenate(IBM, 0)
        
        if args.use_mel:
            # Mel spectrogram
            pass
        elif args.use_mfcc:
            # Mel cepstrogram
            pass
        else:
            # STFT
            Xtr = np.concatenate(trX_mag, 0)
            Xva = np.concatenate(vaX_mag, 0)
            Xte = np.concatenate(teX_mag, 0)
            
        # Normalize
        Xtr = Xtr/np.linalg.norm(Xtr+1e-4, axis=1)[:,None]
        Xva = Xva/np.linalg.norm(Xva+1e-4, axis=1)[:,None]
        Xte = Xte/np.linalg.norm(Xte+1e-4, axis=1)[:,None]
        
        # Save files
        if args.use_mel: 
            suffix = "Mel"
        elif args.use_mfcc: 
            suffix = "MFCC"
        else: 
            suffix = "STFT"
        np.save("Xtr_{}".format(suffix), Xtr)
        np.save("Xva_{}".format(suffix), Xva)
        np.save("Xte_{}".format(suffix), Xte)
        np.save("Ytr", IBM)
        
if __name__ == "__main__":
    main()
