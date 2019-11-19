import my_functions
import sys
import numpy as np
import matplotlib.pyplot as plt
import librosa

args = args = sys.argv

log_power = my_functions.initial_values(args[1])
np.save("fixed_basis", log_power)
librosa.display.specshow(log_power, y_axis="log")
