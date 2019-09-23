#include "panTompkinsAlgorithm.hpp"

#define BUFFER 1200
#define SKIP 600

int signals[BUFFER];
int markedSignals[BUFFER];
int numPeaks;
int averageInterval = 400;

void clearSignalArray();
void readData();
void markSignals();
void writeToFile();
void calcAverageInterval();
void PTCorrections();

void clearSignalArray(){
    //resets the arrays
    memset(signals, 0, sizeof(signals));
    memset(markedSignals,0,sizeof(markedSignals));
}

void readData(){
    //reads from the file into the signals array
    for(int i = 0; i < BUFFER; i++){
        if(!feof(fin))
            fscanf(fin,"%d\n",&signals[i]);
        else
            signals[i] = 0;
    }
    signals[1] = 1;
}

void markSignals(){
    //marks signals to be used as an anchor
    int anchor;
    bool found = false;
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
                found = true;
                break;
            }
            else if(signals[anchor + averageInterval - i] == 1){
                markedSignals[anchor + averageInterval - i] = 1;
                anchor += averageInterval - i;
                break;
                found = true;
            }
        }
        if(!found == true){
            markedSignals[anchor + averageInterval] = 1;
            anchor += averageInterval;
        }
        found = false;
    }
}

void writeToFile(){
    for(int i = 0; i < BUFFER; i++){
        fprintf(fout,"%d\n",markedSignals[i]);
    }
}
/*
void writeToTerminal(){
    for(int i = 0; i < BUFFER; i ++){
        printf("%d\n", markedSignals[i]);
    }
}
*/
void calcAverageInterval(){
    //calculates the average rr interval in the array
    averageInterval = 0;
    int first = -1;
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
    averageInterval = (last - first) / numPeaks;
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