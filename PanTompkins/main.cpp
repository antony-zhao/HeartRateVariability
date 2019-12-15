//#include "PanTompkinsCleanOnly.hpp"
#include "panTompkinsAlgorithm.hpp"
#include "PTCorrections.hpp"

#define SAMPLEFORMAT "\t%*d/%*d/%*d %*d:%*d:%*lf %*c%*c,%lf\n"
#define FORMAT ""

using namespace std;

int main(void) {
	string file1 = "../../ECG_Data/T21.ascii";
	string file2 = "../../Signal/Signal.txt";
	string averaged = "../../Signal/Averaged.txt";
	string variance = "../../Signal/Variance.txt";
	string test = "../../Signal/Test.txt";
	string inverted = "../../Signal/Inverted.txt";

	FileInput f(file1, test);
    TestScalings(f);
    f.SetInFile(file1);
    f.SetOutFile(inverted);
    TestInverted(f);
}