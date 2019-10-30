## Boosted Locality Sensitive Hashing: Discriminative Binary Codes for Source Separation
By learning discriminative hash codes, our boosted locality sensitive hashing framework shows comparative performance to deep learning methods in terms of denoising performance and complexity. This repository contains code used in  **Boosted Locality Sensitive Hashing: Discriminative Binary Codes for Source Separation** (link to be posted).

### Repository structure

#### loader.py
* Helper functions to load TIMIT speakers and Duan and DEMAND noise datasets.

#### generate_data.py
* A python script to generate training and testing wavefiles and spectral features.

### Data Generation
* To generate the clean source, noises, and mixture wavefiles, 
```
python generate_data.py --make_wavefiles
```

* To generate the spectral features from the wavefiles,
```
python generate_data.py --option
```
where options are: None for STFT, --use_mel for mel spectrograms, --use_mfcc for MFCC

### Datasets used in this repository
* TIMIT (https://catalog.ldc.upenn.edu/LDC93S1)
* Duan (http://www2.ece.rochester.edu/~zduan/data/noise/)
* DEMAND (https://zenodo.org/record/1227121#.Xbm4X797leg)
