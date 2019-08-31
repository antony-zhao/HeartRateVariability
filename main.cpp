#include <stdio.h>
#include <string.h>
#define SAMPLEFORMAT "\t%*d/%*d/%*d %*d:%*d:%*lf %*s,%lf\n"
#define FORMAT ""
extern int scaleData(double);
extern void merge(int, int, int);
extern int aveBase();
extern void init(char[], char[]);
extern int inverted();
extern int input(char[]);
extern void linSearch(double[], double[]);
extern void panTompkins(char[]);
extern void output(int);

using namespace std;

int main(void){
    char format[] = SAMPLEFORMAT;
    char inFile[] = "Sample 1.ascii";
    char outFile[] = "Data.txt";
    init(inFile, outFile);
    panTompkins(format);
}
