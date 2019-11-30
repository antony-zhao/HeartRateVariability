#include <string> 
#include <vector>
#include <limits>
#include <iostream>
#include <fstream>

#define AVERAGINGINTERVAL 100000
#define INTERVAL 1200
#define LEEWAY 0.1

using namespace std;

fstream inFileSignal, inFileECG, outFileSignalECG;
int SignalBuffer[INTERVAL];
double ECGBuffer[INTERVAL];

void InitFiles(string file_in, string file_out, string ecg_file) {
	inFileSignal.open(file_in, fstream::in);
	inFileECG.open(ecg_file, fstream::in);
	outFileSignalECG.open(file_out, fstream::out);
}

int AverageSignalsPerInterval() {
	int numSignals = 0;
	int placeholder;
	for (int i = 0; i < INTERVAL; i++) {
		inFileSignal >> placeholder;
		if (placeholder == 1) {
			numSignals++;
		}
	}
	return INTERVAL / numSignals;
}

int CleanSignals(string in, string out, string ecg) {
	InitFiles(in, out,ecg);
	int averageSignals = AverageSignalsPerInterval();
	inFileSignal.close();
	inFileSignal.open(in, fstream::in);
	while (!inFileSignal.eof()) {
		int numSignals = 0;
		for (int i = 0; i < INTERVAL; i++) {
			int placeholder = 0;
			double ecgplaceholder = 0;
			if (!inFileSignal.eof()) {
				inFileSignal >> placeholder;
				inFileSignal.ignore(100, '\n');
				inFileECG.ignore(100, ' ');
				inFileECG.ignore(100, ' ');
				inFileECG.ignore(100, ',');
				inFileECG >> ecgplaceholder;
				inFileECG.ignore(100, '\n');
			}
			SignalBuffer[i] = placeholder;
			ECGBuffer[i] = ecgplaceholder;
			if (placeholder == 1) {
				numSignals++;
			}
		}
		if (INTERVAL/numSignals > averageSignals* (1 + LEEWAY) || INTERVAL/numSignals < averageSignals * (1 - LEEWAY)) {
			continue;
		}
		else {
			for (int i = 0; i < INTERVAL; i++)
				outFileSignalECG << ECGBuffer[i] << '\t' << SignalBuffer[i] << endl;
		}
	}
}
