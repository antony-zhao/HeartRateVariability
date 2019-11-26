#include <stdio.h>     
#include <stdbool.h>
#include <string.h>
#include <string> 
#include <vector>
#include <limits>
#include <iostream>
#include <fstream>

using namespace std;

#define BUFFER 12000
#define SKIP 1200

int signals[BUFFER];
int markedSignals[BUFFER];
int numPeaks;
int averageInterval = 400;
fstream inFile, outFile;

void ClearSignalArray();
void ReadData();
void MarkSignals();
void WriteToFile();
void CalcAverageInterval();
void PTCorrections();
void InitFiles(string, string);

void InitFiles(string file_in, string file_out) {
	inFile.open(file_in, fstream::in);
	outFile.open(file_out, fstream::out);
}


void ClearSignalArray(){
    //resets the arrays
    memset(signals, 0, sizeof(signals));
    memset(markedSignals,0,sizeof(markedSignals));
}


void ReadData(){
    //reads from the file into the signals array
    for(int i = 0; i < BUFFER; i++){
        //if not end of file, read what's next in the file into the array, otherwise just put 0
		if (!inFile.eof()) {
			inFile >> signals[i];
			inFile.ignore(100, '\n');
		}
        else
            signals[i] = 0;
    }
}

void MarkSignals(){
    //marks signals to be used as an anchor
    int anchor;
    bool found = false;
    //finds the anchor
    for(int i = 0; i < BUFFER; i++){
        if(signals[i] == 1){
            markedSignals[i] = 1;
            anchor = i;
            break;          
        }
    }
    while(anchor + (int)(averageInterval*0.1) < BUFFER){
    //while the anchor and the range of search does not reach the end of the buffer
        for(int i = 0; i < (int)(averageInterval * 0.1); i++){
            //limits the search to a range of +- averageInterval*0.2. If there is no signal found, then it just marks at the predicted location of
            //anchor + averageInterval. Otherwise mark the closest signal to anchor+averageInterval and set that as the new anchor
            if(signals[anchor + i + averageInterval] == 1){
                //checks in signals at the anchor + the averageInterval - i 
                //if there is a signal, add this into markedSignals, set anchor to this index, and break out of the for loop
                markedSignals[anchor + i + averageInterval] = 1;
                anchor += i + averageInterval;
                found = true;
                break;
            }
            else if(signals[anchor + averageInterval - i] == 1){
                //checks in signals at the anchor + the averageInterval - i
                //if there is a signal, add this into markedSignals, set anchor to this index, and break out of the for loop
                markedSignals[anchor + averageInterval - i] = 1;
                anchor += averageInterval - i;
                break;
                found = true;
            }
        }
        if(!found == true){
            //if there was no signal in this range, just put one where it would be predicted to be based on the average (for missing signals)
            markedSignals[anchor + averageInterval] = 1;
            anchor += averageInterval;
        }
        //resets false
        found = false;
    }
}

void WriteToFile(){
    //for now it just writes the markedSignals. Eventually it will write from signals whenever signals is corrected even further
    for(int i = 0; i < BUFFER; i++){
		outFile << markedSignals[i] << endl;
    }
}
/*
void writeToTerminal(){
    for(int i = 0; i < BUFFER; i ++){
        printf("%d\n", markedSignals[i]);
    }
}
*/
void CalcAverageInterval(){
    //calculates the average rr interval in the array
    //first index of signal
    int newAverageInterval;
    int first = -1;
    //last index of signal
    int last = -1;
    for(int i = 0; i < BUFFER; i++){
        if(signals[i] == 1){
            if(first == -1){
                first = i;
            }
            if(last < i){
                last = i;
            }
            numPeaks++;
        }
    }
    //total range that still has peaks/number of peaks in that time
    try{
        newAverageInterval = (last - first) / numPeaks;
        if(newAverageInterval > averageInterval*0.99 && newAverageInterval < averageInterval*1.01){
           averageInterval = newAverageInterval;
        }
    } catch(...){
        return;
    }
}

void PTCorrections(){
    //skips the initialization
    int placeholder;
    for(int i = 0; i < SKIP; i++){
		inFile.ignore(100, '\n');
		outFile << "0\n";
    }
    ReadData();
    MarkSignals();
    WriteToFile();
    while(!inFile.eof()){
        //calcAverageInterval();
        //clearSignalArray();
        ReadData();
        MarkSignals();
        WriteToFile();
    }
	inFile.close();
	outFile.close();
}