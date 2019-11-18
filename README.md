# Complex_NMF
This repository is for using NMF and complex NMF. Comments in the codes are written in Japanese.

# Description
Main contents are devided into two types.
1. NMF: just decomposing non negative spectrogram into basis and activation.
2. CNMF: decomposing complex spectrogram with using phase values.

# Usage
## NMF
First, please prepare the spectrogram whose type is ndarray. Then, the fitst argument is the path to the ndarray spectrogram.
```
$ python NMF.py "path_to_spectrogram.npy"
```
Return:
1. basis matrix
2. activation matrix
3. Errors based on the euclid_divergence

## Complex NMF
test
