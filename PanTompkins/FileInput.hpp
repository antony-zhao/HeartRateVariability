#include <iostream>
#include <fstream>

using namespace std;

class FileInput{
private:
    fstream inFile, outFile;
    string in, out;

public:
    FileInput(){}

    FileInput(string in, string out){
        this->in = in;
        this->out = out;
        inFile.open(in,fstream::in);
        outFile.open(out,fstream::out);
        if(!inFile.is_open() && !outFile.is_open()){
            throw "Neither file opened";
        }
        else if(!inFile.is_open()){
            throw "Did not open inFile";
        }
        else if(!outFile.is_open()){
            throw "Did not open outFile";
        }
        else{
            cout << "Opened" << endl;
        }
    }

    double InputFromSample(){
        if(!inFile.eof()){
            double value;
            inFile.ignore(200,',');
            inFile >> value;
            return value;
        }
        else{
            return 0;
        }
    }

    void WriteToTxt(int val){
        outFile << val << endl;
    }

    void WriteToTxt(double val){
        outFile << val << endl;
    }

    int FileLength()
    {
        int length = 0;
        while (!inFile.eof()) {
            length++;
            inFile.ignore(1000, '\n');
        }
        inFile.close();
        inFile.open(in, fstream::in);
        return length;
    }

    bool Eof(void){
        return inFile.eof();
    }

    void CloseFiles(){
        inFile.close();
        outFile.close();
    }

    double Input(){
        throw "Not implemented";
    }

    void SetInFile(string infile){
        inFile.open(infile, fstream::in);
        if(!inFile.is_open()){
            throw "Failed opening inFile";
        }
        this->in = infile;
    }

    void SetOutFile(string outfile){
        outFile.open(outfile, fstream::out);
        if(!outFile.is_open()){
            throw "Opening outfile failed";
        }
        this->out = outfile;
    }

    void Reopen(){
        inFile.open(in, fstream::in);
        outFile.open(out, fstream::out);
    }

    void SetPrecision(int precision){
        outFile << setprecision(precision);
    }
};