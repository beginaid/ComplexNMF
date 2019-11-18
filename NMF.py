import sys
import numpy as np
import librosa
import my_functions

args = sys.argv

spec_path = args[1]
n_iter = int(args[2])
init_H_path = args[3]

S = np.load(spec_path)
init_H = np.load(init_H_path)

H, U, cost = my_functions.NMF(Y=np.abs(S), n_iter=n_iter, init_H=init_H)
print(H, U, cost)
