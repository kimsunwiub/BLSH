# Boosted Locality Sensitive Hashing: Discriminative Binary Codes for Source Separation
This repository provides scripts required for training "Boosted Locality Sensitive Hash (BLSH)" functions and the kNN-based source separation algorithm. The BLSH method was introduced in [1].

## Description

#### loader.py
* Helper functions to load TIMIT speakers and Duan and DEMAND noise datasets.

#### generate_data.py
* A python script to generate training and testing wavefiles and spectral features.

#### check_performance.py
* A python script to check performance of oracle or projected kNN on various features.

#### train_weak_learners.py
* A python script to train weak learners on various features. 

#### utils.py
* Helper functions.

## Usage
Make sure that ```pip``` and ```python3``` are installed (The program was written using Python 3.6) and install the script's dependencies. Note: ```Librosa``` is used for audio reading and writing but can be replaced with other packages such as ```scipy.signal```. ```Matplotlib``` can be removed if not plotting the results. 

### Data Generation
* To generate the clean source, noises, and mixture wavefiles and the spectral features, 
```
python generate_data.py --make_wavefiles --option
```
where options are: None for STFT, --use_mel for mel spectrograms, --use_mfcc for MFCC

### Training weak learners
* To construct hash functions in the form of weak learners, 
```
python train_weak_learners.py --use_stft
```

* For other options, 
```
python train_weak_learners.py -h
```

### Testing performance
* Eg. to test the performance of kNN procedure using ground truth STFT dictionary on closed set, 
```
python check_performance.py -o --use_stft --is_closed
```

* For other options, 
```
python check_performance.py -h
```

### Datasets used in this repository
* TIMIT (https://catalog.ldc.upenn.edu/LDC93S1)
* Duan (http://www2.ece.rochester.edu/~zduan/data/noise/)
* DEMAND (https://zenodo.org/record/1227121#.Xbm4X797leg)

## References
1. S. Kim, H. Yang, and M. Kim. "Boosted Locality Sensitive Hashing: Discriminative Binary Codes for Source Separation" (https://arxiv.org/abs/2002.06239).

## Acknowledgements
This material is based upon work supported by the National Science Foundation under Award Number:1909509.
