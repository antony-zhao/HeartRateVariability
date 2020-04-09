import numpy as np
import os

file = open(os.path.join('..','Signal','SignalPy.txt'),'r')
out = open(os.path.join('..', 'Signal', 'HRV.txt'), 'w')

length = 0
first = True
averageLength = 100
num_signals_to_average = 8
signals = []

for line in file:
    if int(line[0:-1]) == 1 and (first or length/4 > 0.8 * averageLength and length/4 < 1.2 *averageLength):
        if first:
            first = False
        else:
            length += 1
            signals.append(length/4)
            while len(signals) > num_signals_to_average:
                signals.pop(0)
            averageLength = np.mean(signals)
        out.write(str(length/4) + "\t" + str(averageLength) + "\n")
        length = 0
    elif first:
        continue
    elif length/4 > 1.2 * averageLength:
        signals.append(averageLength)
        length = 0
    else:
        length += 1
