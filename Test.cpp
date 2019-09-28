#include "PTCorrections.hpp"

#define SAMPLEFORMAT "\t%*d/%*d/%*d %*d:%*d:%*lf %*c%*c,%lf\n"
#define FORMAT ""

using namespace std;

int main(void){
    char outFile[] = "Signal.txt";
    char outFile2[] = "Test.txt";
	init(outFile,outFile2);
    //readData();
	//markSignals();
	//writeToFile();
	PTCorrections();
}
