//#include "PanTompkinsCleanOnly.hpp"
#include "panTompkinsAlgorithm.hpp"
#include "PTCorrections.hpp"

#define SAMPLEFORMAT "\t%*d/%*d/%*d %*d:%*d:%*lf %*c%*c,%lf\n"
#define FORMAT ""

using namespace std;

int main(void) {
	string file1 = "../../ECG_Data/T21.ascii";
	string file2 = "../../Signal/Signal.txt";
	string file3 = "../../Signal/SignalCorrected.txt";
    //inFilePT.open(file1,fstream::in);
    //cout << (bool)inFilePT.is_open();

    PanTompkins(file1,file2);
}