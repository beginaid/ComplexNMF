import sys
import numpy as np
import librosa
import my_functions

args = sys.argv

spec_path = args[1]
S = np.load(spec_path)
H, U, cost = my_functions.NMF(np.abs(S))
print(H, U, cost)
