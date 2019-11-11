#include "PTCorrections.hpp"

#define SAMPLEFORMAT "\t%*d/%*d/%*d %*d:%*d:%*lf %*c%*c,%lf\n"
#define FORMAT ""

using namespace std;

int main(void) {
	string inFile = "T21_transition example1_180s.ascii";
	string outFile = "Signal.txt";
	string outFile2 = "SignalCorrected.txt";
	init(outFile, outFile2);
	clearSignalArray();
	readData();
	calcAverageInterval();
	markSignals();
	signals;
	markedSignals;
}
