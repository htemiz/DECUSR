# DECUSR
Deep Concolutional Neural Network for Ultrasound Super Resolution

Please refer to the article by Hakan Temiz and Hasan Şakir Bilge:
	https://ieeexplore.ieee.org/abstract/document/9078131
  
The four trained models (.h5 files) of DECUSR for scales 2, 3, 4 and 8.

.py files contain the definition of DECUSR model. Decusr.py includes a generic, definition.
You can construct the DECUSR with the code and instructions in the file. 

Decusr_for_DeepSR.py provides the definition tailored to use DECUSR with DeepSR,
which eases and automates the super-resolution-specific processes (training, test, augmenting,
normalization, etc.), for super-resolution.

For more information on how to use DeepSR, please refer to:
	https://github.com/htemiz/DeepSR

	PyPi page:

	https://pypi.org/project/DeepSR/

For program manual please refer to:
	https://github.com/htemiz/DeepSR/blob/master/DeepSR/docs/DeepSR%20Manual.pdf

Just a basic instructions to run this model for training and test with DeepSR:
	
	python.exe -m DeepSR.DeepSR --modelfile <full path of this file > --train --test .... <other command arguments> ...

	
To install DeepSR:
	
	pip install DeepSR

PLese note that, software live and change like humans, therefore, the program may not work on your computing environment 
due to possible changes in the dependent software packages. You can run the DeepSR with exact versions of dependent packages
as instructued in the program manual.

New Releases to be issued soon...
