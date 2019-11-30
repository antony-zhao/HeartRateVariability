//#include "PanTompkinsCleanOnly.hpp"
#include "PanTompkinsAlgorithm.hpp"
#include "PTCorrections.hpp"

#define SAMPLEFORMAT "\t%*d/%*d/%*d %*d:%*d:%*lf %*c%*c,%lf\n"
#define FORMAT ""

using namespace std;

int main(void) {
	string file1 = "../ECG_Data/T21.ascii";
	string file2 = "../Signal/Signal.txt";
	string file3 = "../Signal/SignalCorrected.txt";
	/*
	InitPT(file1,file2);
	if(!inFilePT.is_open() || !outFilePT.is_open()){
	    cout << "Did not open" << endl;
	    return 1;
	}
	else{
        cout << "Opened" << endl;
        inFilePT.close();
        outFilePT.close();
	}
	 */
	PanTompkins(file1,file2);
}