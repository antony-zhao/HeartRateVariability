#include "panTompkinsAlgorithm.hpp"

#define BUFFER 1200
#define SKIP 100

int signals[BUFFER];
int markedSignals[BUFFER];
int intervalSum;
int numPeaks;
int averageInterval = 300;

void clearSignalArray();
void readData();
void markSignals();
void writeToFile();
void calcAverageInterval();
void PTCorrections();

void clearSignalArray(){
    memset(signals, 0, sizeof(signals));
    memset(markedSignals,0,sizeof(markedSignals));
}

void readData(){
    for(int i = 0; i < BUFFER; i++){
        if(!feof(fin))
            fscanf(fin,"%d\n",&signals[i]);
        else
            signals[i] = 0;
    }
}

void markSignals(){
    int anchor;
    for(int i = 0; i < BUFFER; i++){
        if(signals[i] == 1){
            markedSignals[i] = 1;
            anchor = i;
            break;          
        }
    }
    while(anchor + (int)(averageInterval*0.2) < BUFFER){
        for(int i = 0; i < (int)(averageInterval * 0.2); i++){
            if(signals[anchor + i + averageInterval] == 1){
                markedSignals[anchor + i + averageInterval] = 1;
                anchor += i + averageInterval;
                break;
            }
            else if(signals[anchor + averageInterval - i] == 1){
                markedSignals[anchor + averageInterval - i] = 1;
                anchor += averageInterval - i;
                break;
            }
            else if(i == (int)(averageInterval * 0.2) - 1 ){
                markedSignals[anchor + averageInterval] = 1;
                anchor += averageInterval;
            }
        }
    }
}

void writeToFile(){
    for(int i = 0; i < BUFFER; i++){
        fprintf(fout,"%d\n",markedSignals[i]);
    }
}

void calcAverageInterval(){
    averageInterval = 0;
    for(int i = 0; i < BUFFER; i++){
        if(signals[i] == 1){
            averageInterval += intervalSum;
            intervalSum = 0;
            numPeaks++;
        }
        else{
            intervalSum++;
        }
    }
    averageInterval /= numPeaks;
}

void PTCorrections(){
    for(int i = 0; i < SKIP; i++)
        fscanf(fin,"%*d\n");
    while(!feof(fin)){
        calcAverageInterval();
        clearSignalArray();
        readData();
        markSignals();
        writeToFile();
    }
    fclose(fin);
    fclose(fout);
}