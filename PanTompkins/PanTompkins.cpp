#include "PanTompkinsCleanOnly.hpp"
#include "PanTompkinsAlgorithm.hpp"

#define SAMPLEFORMAT "\t%*d/%*d/%*d %*d:%*d:%*lf %*c%*c,%lf\n"
#define FORMAT ""

using namespace std;

int main(void) {
	string file1 = "../ECG_Data/T21.ascii";
	string file2 = "../Signal/Signal.txt";
	string file3 = "../Signal/SignalCorrected.txt";
	PanTompkins(file1,file2);
	CleanSignals(file2, file3, file1);
}