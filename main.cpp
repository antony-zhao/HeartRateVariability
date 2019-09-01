#include <stdio.h>
#include <string.h>
#include <iostream>
#define SAMPLEFORMAT "\t%*d/%*d/%*d %*d:%*d:%*lf %*c%*c,%lf\n"
#define FORMAT ""
extern int scaleData(double);
extern void merge(int, int, int);
extern double aveBase();
extern void init(char[], char[]);
extern int inverted();
extern int input(char[]);
extern void linSearch(double[], double[]);
extern void panTompkins(char[]);
extern void output(int);
extern void output(double);
extern FILE* finPT;
extern FILE* foutPT;
extern int first; 

using namespace std;

int main(void){
    char inFile[] = "T21_transition example1_180s.ascii";
    char outFile[] = "Signal.txt";
    char outFile2[] = "ECG.txt";
    init(inFile, outFile);
    panTompkins(SAMPLEFORMAT);
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
