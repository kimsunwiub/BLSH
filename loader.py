import numpy as np
import librosa
import random
import os

from utils import count_mins

def setup_ada_training_data(seed):
    # 1. Load Duan noise set
    noise_frqs = load_Duan_noises('Data/Duan')
    noise_idx = np.arange(len(noise_frqs))

    # 2. Load training speakers
    trs_spkr_lists = []
    for i in range(1,8+1): 
        # Use all 8 dialects and 20 speakers
        trs_spkr_lists += get_random_dr_speakers(i, 20, seed)
    random.seed(seed)
    random.shuffle(trs_spkr_lists)
    print ("Train Speakers: {}".format(trs_spkr_lists))
    N_tr_spkrs = len(trs_spkr_lists)
    print ("Total ", N_tr_spkrs, " speakers")
    
    # 3. Mix with the noise
    trs, trn, trx, vas, van, vax = load_trainset(
        trs_spkr_lists, noise_idx, noise_frqs, seed)
    print ("Loaded {} training utterances".format(len(trs)))
    print ("Loaded {} validation utterances".format(len(vas)))
    count_mins(trs), count_mins(vas)
    
    # 4. Normalize
    max_amp = np.concatenate(trx).ravel().max()
    trs, trn, trx, vas, van, vax = normalize_frqs(max_amp, 
                                    [trs, trn, trx, vas, van, vax])
    
    return trs, trn, trx, vas, van, vax, max_amp

def setup_ada_testing_data(seed):
    # 1. Load DEMAND noise set
    noise_frqs = load_DEMAND_noises('Data/DEMAND')
    
    # 2. Load testing speakers
    tes_spkr_lists = []
    for dr_idx in range(1,8+1):
        dr_spkrs = ['dr{}/{}'.format(dr_idx, name) for name in
                    os.listdir('Data/test/dr{}'.format(dr_idx))
                    if 'Store' not in name]
        dr_num_spkrs = min(count_gender(dr_spkrs))
        dr_spkrs = get_random_dr_speakers_test(
            dr_idx, dr_num_spkrs, 42)
        tes_spkr_lists.append(dr_spkrs)
    tes_spkr_lists  = [item for sublist in tes_spkr_lists
                       for item in sublist]
    random.shuffle(tes_spkr_lists)
    print ("Test Speakers: {}".format(tes_spkr_lists))
    tes_spkr_lists = tes_spkr_lists[:-2]
    
    # 3. Mix with the noise
    tes, ten, tex = load_testset(
        tes_spkr_lists, noise_frqs, seed)
    print ("Loaded {} testing utterances".format(len(tes)))
    count_mins(tes)
    
    # 4. Normalize
    max_amp = np.concatenate(tex).ravel().max()
    tes, ten, tex = normalize_frqs(max_amp, 
                                    [tes, ten, tex])
    
    return tes, ten, tex, max_amp

def load_trainset(trs_spkr_lists, noise_idx_list, noise_frqs, seed):
    trs, trn, trx, vas, van, vax = [], [], [], [], [], []
    for trs_spkr in trs_spkr_lists:
        spkr_frqs = load_spkr('Data/train/{}'.format(trs_spkr))
        for j in range(len(spkr_frqs)):
            s = spkr_frqs[j]
            for idx in noise_idx_list:
                n, x = get_and_add_noise(
                    noise_frqs[idx], s)
                if j < 2:
                    vas.append(s)
                    van.append(n)
                    vax.append(x)
                else:
                    trs.append(s)
                    trn.append(n)
                    trx.append(x)
    return trs, trn, trx, vas, van, vax

def load_testset(tes_spkr_lists, noise_frqs, seed):
    tes, ten, tex = [], [], []
    for tes_spkr in tes_spkr_lists:
        spkr_frq = load_spkr('Data/test/' + tes_spkr)
        noise_idxs = np.random.permutation(len(noise_frqs))[:len(spkr_frq)]
        for j in range(len(spkr_frq)):
            s = spkr_frq[j]
            n = noise_frqs[noise_idxs[j]]
            while len(n) < len(s):
                n = np.tile(n,2)
            spkr_n, spkr_x = get_and_add_noise(n, s)
            tes.append(s)
            ten.append(spkr_n)
            tex.append(spkr_x)

    return tes, ten, tex

def load_spkr(spkr_dir):
    spkr_files = [x for x in os.listdir(spkr_dir) if 'wav' in x]
    spkr_frqs = [librosa.load('{}/{}'.format(spkr_dir, x), sr=16000)[0] for x in spkr_files]
    spkr_frqs = [frqs/frqs.std() for frqs in spkr_frqs]
    return spkr_frqs

def load_Duan_noises(noise_dir):
    noise_files = ['birds.wav', 'casino.wav', 'cicadas.wav', 'computerkeyboard.wav', 'eatingchips.wav', 'frogs.wav', 'jungle.wav', 'machineguns.wav', 'motorcycles.wav', 'ocean.wav']
    noise_frqs = [librosa.load('{}/{}'.format(noise_dir, x), sr=16000)[0] for x in noise_files]
    noise_frqs = [frqs/frqs.std() for frqs in noise_frqs]
    return noise_frqs

def load_DEMAND_noises(noise_dir):
    noise_files = os.listdir(noise_dir)
    noise_files.sort()
    noise_frqs = [librosa.load('{}/{}'.format(noise_dir, x), 
                               sr=16000)[0] for x in noise_files]
    noise_frqs = [frqs/frqs.std() for frqs in noise_frqs]
    print (noise_files, len(noise_frqs))
    return noise_frqs

def get_and_add_noise(noise, source):
    noise_start = np.random.randint(0,len(noise) - len(source))
    spkr_noise = noise[noise_start:noise_start + len(source)]
    return spkr_noise, spkr_noise + source

def load_more_spkr_with_noise(spkr_dir, noise, seed):
    spkr_frqs = load_spkr(spkr_dir)
    spkr_s = np.concatenate(spkr_frqs, 0)
    spkr_n, spkr_x = get_and_add_noise(noise, spkr_s, seed)
    return spkr_s, spkr_n, spkr_x

def get_random_dr_f_speakers(dr_idx, num_speakers, seed):
    all_spkrs = ['dr{}/{}'.format(dr_idx, name) for name in
                    os.listdir('Data/train/dr{}'.format(dr_idx))if 'Store' not in name]
    f_spkrs = [spkr for spkr in all_spkrs if spkr.split('/')[1][0] == 'f']
    np.random.seed(seed)
    perms = np.random.permutation(len(f_spkrs))[:num_speakers]
    return [f_spkrs[i] for i in perms]

def get_random_dr_m_speakers(dr_idx, num_speakers, seed):
    all_spkrs = ['dr{}/{}'.format(dr_idx, name) for name in
                    os.listdir('Data/train/dr{}'.format(dr_idx))if 'Store' not in name]
    m_spkrs = [spkr for spkr in all_spkrs if spkr.split('/')[1][0] == 'm']
    np.random.seed(seed)
    perms = np.random.permutation(len(m_spkrs))[:num_speakers]
    return [m_spkrs[i] for i in perms]

def get_random_dr_speakers(dr_idx, num_speakers, seed):
    num_f = num_m = num_speakers//2
    if num_speakers % 2 != 0:
        if dr_idx % 2 == 0:
            num_f += 1
        else:
            num_m += 1
    f_spkrs = get_random_dr_f_speakers(dr_idx, num_f, seed)
    m_spkrs = get_random_dr_m_speakers(dr_idx, num_m, seed)
    fm_spkrs = f_spkrs + m_spkrs
    return fm_spkrs

def count_gender(spkrs):
    female_count = 0
    male_count = 0
    for s in spkrs:
        gender = s.split('/')[1][0]
        if gender == 'f':
            female_count += 1
        elif gender == 'm':
            male_count += 1
        else:
            print (333)
            break
    return female_count, male_count

def get_random_dr_f_speakers_test(dr_idx, num_speakers, seed):
    all_spkrs = ['dr{}/{}'.format(dr_idx, name) for name in
                    os.listdir('Data/test/dr{}'.format(dr_idx))if 'Store' not in name]
    f_spkrs = [spkr for spkr in all_spkrs if spkr.split('/')[1][0] == 'f']
    np.random.seed(seed)
    perms = np.random.permutation(len(f_spkrs))[:num_speakers]
    return [f_spkrs[i] for i in perms]

def get_random_dr_m_speakers_test(dr_idx, num_speakers, seed):
    all_spkrs = ['dr{}/{}'.format(dr_idx, name) for name in
                    os.listdir('Data/test/dr{}'.format(dr_idx))if 'Store' not in name]
    m_spkrs = [spkr for spkr in all_spkrs if spkr.split('/')[1][0] == 'm']
    np.random.seed(seed)
    perms = np.random.permutation(len(m_spkrs))[:num_speakers]
    return [m_spkrs[i] for i in perms]

def get_random_dr_speakers_test(dr_idx, num_speakers, seed):
    num_f = num_m = num_speakers
    f_spkrs = get_random_dr_f_speakers_test(dr_idx, num_f, seed)
    m_spkrs = get_random_dr_m_speakers_test(dr_idx, num_m, seed)
    fm_spkrs = f_spkrs + m_spkrs
    return fm_spkrs

def normalize_frqs(max_amp, frqs):
    return [frq/max_amp for frq in frqs]

