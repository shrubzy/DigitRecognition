# Description  
A script that trains a neural network to classify handwritten digits using the MNIST dataset.   

# Options  
By default the program expects the MNIST dataset to be in the same directory as python files.
This can be changed so that the dataset is downloaded in "neuralnet.py".

There is a flag in "neuralnet.py" that controls whether the images should be shown as a matplotlib plot.
This is set to false by default.

# Install  
Download the files in the repository.  

# Run
Running the "make" command creates the virtual environment and installs the required libraries.

The virtual environtment is activated using the command:
	source ./env/bin/activate

Once the virtual environment is activated the program can be run using:
	python3 classifier.py

Upon execution the program will train the neural network.
The user will then be prompted to enter a filepath to an image.
The program will then output the classification for the image.

The program can be terminated by typing "exit" when prompted for a filepath.

The virtual environment is deactivate by using the command "deactivate".
The virtual environment can be cleaned using "make clean".
