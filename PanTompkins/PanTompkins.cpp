#include "PTCorrections.hpp"
#include "panTompkinsAlgorithm.hpp"

#define SAMPLEFORMAT "\t%*d/%*d/%*d %*d:%*d:%*lf %*c%*c,%lf\n"
#define FORMAT ""

using namespace std;

int main(void) {
	string file1 = "../ECG_Data/T21.ascii";
	string file2 = "../Signal/Signal.txt";
	string file3 = "../Signal/SignalCorrected.txt";
	InitPT(file1, file2);
	PanTompkins();
	InitFiles(file2, file3);
	PTCorrections();
}