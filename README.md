# Heart Rate Variability

## About
The program uses a convolutional neural network, using Keras, to identify and mark
the R-waves, and produces relevant data on the heart rate variability on the sample.
There is also an interface, using Matplotlib widgets, 
to check if the marks are correct. Our data is from the Ponemah software and has a
specific format.

## How To Use

### File Structure
#### Application
Contains the python programs to run the program. 
#### ECG_Data
Contains the ECG data as well as the R-peak markings produced by model_prediction.py.
#### Signal
Contains the sheets for the timestamps of the peaks.
#### Training
Contains the data needed to train the program (numpy files as well as raw files of
signals and markings)


### config.py
Contains parameters for the program, adjusting the values in here will adjust the 
parameters in the other programs.

### ecg_marking_matcher.py
Reads from a file containing ECG data and a sheet containing the timestamps of markings,
and produces 2 files which can be used to create a dataset from.
#### Inputs
![Input ECG](Format Samples/t21-600s.png)\
![Input Sheet](Format Samples/t21-input.png)
#### Outputs
![Output raw ECG](Format Samples/ECG.png)\
![Output markings](Format Samples/markings-(training).png)

### dataset.py
Creates .npy files of the train and test datasets, using the outputs from 
ecg_marking_matcher.py. The samples are (samples, datapoints, stack).

### model.py
Trains a model using the datasets created from dataset.py. The model is a 1D convolutional
network, and the one sample is (datapoints, stack), with stack adding a temporal dimension
and allowing the model to incorporate data from previous signals.

### model_prediction.py
Runs the previously trained model on a file of ECG data. It will output multiple files of 
the markings in case the file is too large so that the plotting program
won't lag when displaying. These have the convention of inputfilename + number.txt, so
for example, Sample.ascii might be split into Sample001.txt, Sample002.txt, and Sample003.txt.
#### Input
![Input ECG](Format Samples/ascii.png)
#### Output
![Output markings](Format Samples/markings.png)

### plot.py
Plots markings produced by model_prediction.py, and allows for user corrections if desired.
#### Input
![Input](Format Samples/input-to-plot.png)
#### Plot
![Plot](Format Samples/plot.png)

### post_processing.py
Converts the markings from model_prediction.py into a sheet of relevant information on the heart rate variability
on the data. Allows for selecting of multiple files together and joins them afterwards.
#### Output
![Sheet](Format Samples/sheet.png)


## Contact Me
Email me at: tonyzhao.davis@gmail.com or ayzhao7761@berkeley.edu

