# Complex_NMF
<div align="center">
<img src="images/CNMF.png" alt="" title="", width="800">
</div>

<br>

This repository is for using NMF and complex NMF. Comments in the codes are written in Japanese.

<br>

# Description
Main contents are devided into two types.
1. NMF: just decomposing non negative spectrogram into basis and activation.
2. CNMF: decomposing complex spectrogram with using phase values.

In this context, the basis matrix is considered to be fixed. Therefore, please give them initial basis matrix as a fixed matrix. Note that the activation is initialized by gaussian distribution.

<br>

# Usage
## Making initialized basis spevtrogram
In order to conduct source separation, basis supectrogram should be prepared. In this case, I made simple basis spectrogram from piano sound templates. These samplate is sampled from Musescore, one of the most popular notation software. In this repository named "basis_wav_piano" includes 5 octaves of single piano sound. Then using the code below, you can make fixed basis spectrograms which is made through NMF.

```
$ python python make_init_H.py [path_to_template_directory]
```
Arguments:  
1. Path to the directory which includes template wav files.


Return:
- Basis matrix (fixed_basis.npy)

<br>

In this algorithm, I adopt the parameters based on the harmonic feature of piano as follows. 130.81 is the frequency of C3. Also, the damping coefficient is set as 0.9, variance as 1.0 and the number of peak as 50.

<br>

<div align="center">
<img src="images/GMM_math.png" alt="" title="", width="400">
</div>
<div align="center">
<img src="images/parameters.png" alt="" title="", width="280">
</div>

<br>

<div align="center">
<img src="images/GMM.png" alt="" title="", width="400">
</div>

<br>

Those parameters determin the initial values of NMF to make basis spectrograms. Furthermore, you can prepare another template. Note that default settings are basically rely on the librosa's ones.

<br>

<div align="center">
<img src="images/fixed_basis_spectrogram.png" alt="" title="", width="400">
</div>


## NMF
First, please prepare the spectrogram whose type is ndarray. Also, prepare the initial values of basis matrix, which is fixed through the overall irerations. Note that the number of frequency bins must be the same number between the spectrogram and the initial values.

<br>


<br>

```
$ python NMF.py [path_to_spectrogram.npy] [iterations] [path_to_initial_values]
```

<br>

Arguments:  
1. Path to the spectrogram file made by ndarray.
2. The number of overall iterations.
3. Path to the initial values of basis matrix


Return:
- Basis matrix (basis_calc.npy)
- Activation matrix (activation_calc.npy)
- Errors based on the euclid_divergence (cost.npy)  
※Automatically saved ndarray as ".npy". In addition, automatically draw the learning curve.
<div align="center">
<img src="images/learning_curve_NMF.png" alt="" title="", width="400">
</div>

<br>

## Complex NMF
As the case of NMF, please prepare the spectrigram with ndarray. At the same time, make the initial values as fixed matrix. Then the activation matrix is initialized using the same way as NMF while phase matrix is initialized as follows. It is based on the original paper of complex NMF[2].

<br>

<div align="center">
<img src="images/init_phases.png" alt="" title="", width="300">
</div>  

<br>

```
$ python CNMF.py [path_to_spectrogram.npy] [iterations] [path_to_initial_values] -thr [threshold]
```

<br>

Arguments:  
1. Path to the spectrogram file made by ndarray.
2. The number of overall iterations.
3. Path to the initial values of basis matrix
4. Threshold of the error (optional)


Return:
- Basis matrix (basis_calc.npy)
- Activation matrix (activation_calc.npy)
- Exponential-phase matrix (phase_calc.npy)
- Reconstructed spectrigram (reconst.npy)
- Errors based on the euclid_divergence (cost.npy)  

※Automatically saved ndarray as ".npy". In addition, automatically draw the learning curve.
<div align="center">
<img src="images/learning_curve_CNMF.png" alt="" title="", width="400">
</div>

<br>

# References
[1] Kameoka, Hirokazu, et al. "Complex NMF: A new sparse representation for acoustic signals." 2009 IEEE International Conference on Acoustics, Speech and Signal Processing. IEEE, 2009.
