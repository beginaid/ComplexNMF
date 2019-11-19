import sys
import numpy as np
import librosa
import my_functions
import argparse

parser = argparse.ArgumentParser(description="This program is for complex-NMF")
parser.add_argument(
    "spec_path", help="Path to the spectrogram you want to decompose")
parser.add_argument("n_iter", help="The number of iterations")
parser.add_argument(
    "init_H_path", help="Path to the spectrogram you want to fix")
parser.add_argument("-thr", "--thres")

args = parser.parse_args()

spec_path = args.spec_path
n_iter = int(args.n_iter)
init_H_path = args.init_H_path

S = np.load(spec_path)[:, :100]
init_H = np.load(init_H_path)

error, F, H, U, P_exp, nm_iter, gap = my_functions.CNMF(
    Y=S, n_iter=n_iter, init_H=init_H)

np.save("cost.npy", error)
np.save("reconst.npy", F)
np.save("basis_calc.npy", H)
np.save("activation_calc.npy", U)
np.save("phase_calc.npy", P_exp)
