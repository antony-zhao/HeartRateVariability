#include "PTCorrections.hpp"

#define SAMPLEFORMAT "\t%*d/%*d/%*d %*d:%*d:%*lf %*c%*c,%lf\n"
#define FORMAT ""

using namespace std;

int main(void){
    char format[] = "\t%*d/%*d/%*d %*d:%*d:%*lf %*c%*c,%lf\n";
    char inFile[] = "T21_transition example1_180s.ascii";
    char outFile[] = "Signal.txt";
    char outFile2[] = "SignalCorrected.txt";
    //init(inFile, outFile);
    //panTompkins(format);
    init(outFile, outFile2);
    clearSignalArray();
    readData();
    calcAverageInterval();
    markSignals();
    signals;
    markedSignals;
    //fout = fopen(outFile2,"w+");
    //fprintf(fout,"%d\n",1);
    /*
    init(inFile, outFile2);
    first = 1;
    while(!feof(finPT))
        output(input(SAMPLEFORMAT));
    fclose(finPT);
    fclose(foutPT);
    return 0;
    */ 
}
