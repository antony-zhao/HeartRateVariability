from matplotlib import pyplot as plt

def ecg_from_file(ecg, filename):
    f = open(filename, 'r')
    for x in f:
        ecg.append(float(x[x.index(',') + 1:x.index('\n')]))
    f.close()

def signal_from_file(signal, filename):
    f = open(filename,'r')
    for x in f:
        signal.append(float(x))
    f.close()
    return signal

def ecg_signal_from_file(ecg,signal,filename):
    f = open(filename, 'r')
    for x in f:
        ecg.append(float(x[0:x.index('\t')]))
        ecg.append(int(x[x.index('\t') + 1:x.index('\n')]))
    f.close()