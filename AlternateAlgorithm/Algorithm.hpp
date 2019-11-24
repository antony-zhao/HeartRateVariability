#include <iostream>
#include <fstream>
#include <filesystem>

#define BUFFER 1000
#define SURROUND 2

using namespace std;

fstream inFile, outFile;
double ECG[BUFFER];
double signals[BUFFER];


void InitializeFiles(string in, string out) {
	inFile.open(in, fstream::in);
	outFile.open(out, fstream::out);
}

double ReadData() {
	//To be modified
	double placeholder;
	inFile >> placeholder;
	return placeholder;
}

double ReadFromASCIIFile() {
	char charPlaceholder;
	double doublePlaceholder;

	if (!inFile.eof()) {
		do {
			charPlaceholder = inFile.get();
			if (charPlaceholder == '#') {
				inFile.ignore(400, '\n');
			}
		} while (charPlaceholder == '#');

		inFile.ignore(100, ',');
		doublePlaceholder = ReadData();
		return doublePlaceholder;
	}
	else {
		return 0;
	}
}

void FillECG() {
	for (int i = 0; i < BUFFER; i++) {
		ECG[i] = ReadFromASCIIFile();
	}
}

/*
double BaseLine() {
	double sum = 0;
	int numSignals = BUFFER;
	for (int i = 0; i < BUFFER; i++) {
		sum += ECG[i];
	}
	return sum / numSignals;
}
*/

void MarkSignals(string in, string out) {
	double base;
	double sureness;

	InitializeFiles(in, out);
	for (int i = 0; i < SURROUND; i++) {
		outFile << 0 << endl;
	}
	while (!inFile.eof()) {
		FillECG();
		//base = BaseLine();
		for (int i = SURROUND; i < BUFFER - SURROUND; i++) {
			sureness = 0;
			for (int j = i - SURROUND; j < i + SURROUND; j++) {
				sureness += abs(ECG[i] - ECG[j]);
			}
			sureness /= 2 * SURROUND * ECG[i];
			outFile << sureness << endl;
		}
	}
}