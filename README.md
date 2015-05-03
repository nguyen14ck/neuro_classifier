# CLASSIFIER PROGRAM
#

# FILES & PACKAGES

	
|── .
|  |── README.md                     
|  
|  |── .\mregit_senti

|  |  |── cloth_test_sample.txt         
|  |  |── cloth_train_sample1.txt       
|  |  |── cloth_train_sample2.txt       
|  |  |── dependency-reduced-pom.xml    
|  |  |── pom.xml                       
|  |  
|  |  |── .\mregit_senti\cloth_output_sample

|  |  |  |── _SUCCESS                      
|  |  |  |── part-m-00000                  
|  |  |  
|  |  |── .\mregit_senti\cloth_train_sample1_temp1

|  |  |  |── _SUCCESS                      
|  |  |  |── part-r-00000                  
|  |  |  
|  |  |── .\mregit_senti\cloth_train_sample1_temp2

|  |  |  |── _SUCCESS                      
|  |  |  |── part-r-00000                  
|  |  |  
|  |  |── .\mregit_senti\src
|  |  |  
|  |  |  |── .\mregit_senti\src\main
|  |  |  |  
|  |  |  |  |── .\mregit_senti\src\main\java
|  |  |  |  |  
|  |  |  |  |  |── .\mregit_senti\src\main\java\neuro
|  |  |  |  |  |  
|  |  |  |  |  |  |── .\mregit_senti\src\main\java\neuro\core

|  |  |  |  |  |  |  |── ActivationLayer.java          
|  |  |  |  |  |  |  |── ActivationNetwork.java        
|  |  |  |  |  |  |  |── ActivationNeuron.java         
|  |  |  |  |  |  |  |── BackPropagationLearning.java  
|  |  |  |  |  |  |  |── BipolarSigmoidFunction.java   
|  |  |  |  |  |  |  |── DoubleRange.java              
|  |  |  |  |  |  |  |── IActivationFunction.java      
|  |  |  |  |  |  |  |── IntRange.java                 
|  |  |  |  |  |  |  |── ISupervisedLearning.java      
|  |  |  |  |  |  |  |── Layer.java                    
|  |  |  |  |  |  |  |── Network.java                  
|  |  |  |  |  |  |  |── neural_network.java           
|  |  |  |  |  |  |  |── Neuron.java                   
|  |  |  |  |  |  |  |── SigmoidFunction.java          
|  |  |  |  |  |  |  
|  |  |  |  |  |  |── .\mregit_senti\src\main\java\neuro\mre

|  |  |  |  |  |  |  |── senti_classify.java           
|  |  |  |  |  |  |  
	
	
    PACKAGE neuro.core:
		Neuron - abstract class of neurons
		Layer - abstract class of collection of neurons
		Network - abstract class of network
		
		ActivationNeuron - implementation of Neuron abstract class
		ActivationLayer - implementation of Layer abstract class
		ActivationNetwork - implementation of Network abstract class
		
		IActivationFunction - activation function's interface
		Sigmoid - implementation IActivationFunction
		BipolarSigmoid - implementation of IActivationFunction
		
		IUnsupervisedLearning - interface for unsupervised learning algorithms 
		ISupervisedLearning - interface for supervised learning algorithms
		Back Propagation Learning - inplementation of supervisedLearning interface for  multi-layer neural network 
	
    PACKAGE neuro.mre:
		senti_classify - class contains:
			+ class Map_Training
			+ class Reduce_Training
			+ class Map_Testing
			+ class Reduce_Testing (not implemented, only use counter)
			+ class main (driver)
			
	
# COMPONENTS & LICENSE
    This program uses some following components:
	•	Neural network class (Andrew Kirillov, 2006) - GPL3 - http://www.codeproject.com/Articles/16447/Neural-Networks-on-C


# NEURAL NETWORK

This network has following configuration:
	+ network: 1 input layer (with 5 neurons), 1 hidden layer (5 neurons) 1 output layer (2 neurons)

	+ input vector: 5 dimensions (this vector is extracted from the previous linguistic program)
	  first 5 numbers are for input, 3 last numbers are ground truth (for training, and for testing purpose)
	  each column indicates:
	 (1):	positive scores
	 (2):	negative scores
	 (3):	number of positive sentences
	 (4):	number of negative sentences
	 (5):	scores of summary text

	If review/score >= 4, this review is positive. If review/score <= 2, it's negative.

	columns 6-8 are target values, prepared in different forms for different applications.

	 (6):	ground truth_ 1 if positive, 0 if negative
	 (7):	ground truth_ 0 if positive, 1 if negative
	 (8): 	ground truth_ 1 if positive, -1 if negative

Because classification process needs validation, dataset will be split into 3 equal parts. For each runs, 2 parts are used for input data, and the remaining part as testing data.


# RUNNING INSTRUCTION

This program takes 5 arguments to run:

	hadoop jar senti_classify.jar neuro.mre.senti_classify  <number of training epochs> <path of input file 1 for training> <path of input file 2 for training> <path of input file testing> <path of output folder> 

Training epochs should be set at least 2.
	
After entering 5 arguments, program prints out all arguments it's just read. And user can check again to ensure it reads properly.

The main class is senti_classify. Normally, hadoop will ask for main class name after jar file name.

	ex: hadoop jar senti_classify_small.jar neuro.mre.senti_classify 2 /user/cloudera/MapReduce/cloth_train_sample1.txt /user/cloudera/MapReduce/cloth_train_sample2.txt /user/cloudera/MapReduce/cloth_test_sample.txt /user/cloudera/MapReduce/cloth_output_sample

However in other times, because there only 1 executing class in this jar file, hadoop can ignore this argument, and program receives this name class as a new first argument. Please delete class name if you see so after you enter executing command.

	ex: hadoop jar senti_classify_small.jar 2 /user/cloudera/MapReduce/cloth_train_sample1.txt /user/cloudera/MapReduce/cloth_train_sample2.txt /user/cloudera/MapReduce/cloth_test_sample.txt /user/cloudera/MapReduce/cloth_output_sample
	
Program runs 3 mapreduce jobs (each job can has many tasks): 2 jobs for training (2 epochs, after each epoch, it will update average neurons' weights for next epoch), and 1 job for testing (takes neurons' weights from training steps)
	
Finally, it generates results in output folder. Intermediate data of computing process will be stored in two temporary folders beside output folder.


