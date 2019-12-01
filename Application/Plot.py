from Methods import *
import os.path


ecg = []
signal = []
signal2 = []
ecg_from_file(ecg,os.path.join('..','ECG_Data','T21.ascii'))
signal_from_file(signal, os.path.join('..','Signal','Signal.txt'))
#ecg_signal_from_file(ecg,signal,"../Signal/SignalCorrected.txt")
'''
signal_from_file(signal2,"Signal.txt")
signal_from_file(ecg2,"ECG.txt")
signal = signal_from_file(signal,"Sample1Corrected.txt")
ecg = ecg_from_file(ecg,"Sample 2.ascii")
signal = signal_from_file(signal, "Sample2Corrected.txt")
ecg = ecg_from_file(ecg,"Sample 3.ascii")
signal = signal_from_file(signal,"Sample3Corrected.txt")
ecg = ecg_from_file(ecg,"Sample 4.ascii")
signal = signal_from_file(signal, "Sample4Corrected.txt")
ecg = np.asarray(ecg)
ecg = np.append(ecg,-ecg)
signal = np.asarray(signal)
signal = np.append(signal,signal)
ecg = ecg.reshape(ecg.shape[0],1)
signal = signal.reshape(signal.shape[0],1) 
#plt.plot(range(len(signal2)),signal2)
#plt.plot(range(len(ecg2)),ecg2)
'''
plt.plot(range(len(signal[0:1000000])),signal[0:1000000])
plt.plot(range(len(ecg[0:1000000])),ecg[0:1000000])
plt.show()