from Methods import *
import os.path
from matplotlib import pyplot as plt
import time

ecg_file = open(os.path.join('..', 'ECG_Data', 'T21.ascii'), 'r')
signal_file = open(os.path.join('..', 'Signal', 'SignalPy.txt'), 'r')

ecg_cache = []
signal_cache = []
