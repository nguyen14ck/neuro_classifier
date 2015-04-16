# NEURAL NETWORK

This network has following configuration:
+ network: 1 input layer (with 5 neurons), 1 hidden layer (5 neurons) 1 output layer (2 neurons)

+ input vector: 5 dimensions (this vector is extracted from the previous linguistic program)
  first 5 numbers are for input, 3 last numbers are ground truth (for training, and for testing purpose)

will be extracted as following vector:
(1)				(2)				(3)				(4)				(5)				(6)		(7)		(8)
0.3014439716794325		-0.0869025267933975		0.07142857142857142		0.07142857142857142		0.7536099291985813		1		0		1		

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


#RUNNING INSTRUCTION

This program takes 5 arguments to run:

	hadoop jar senti_classify.jar neuro.mre.senti_classify  <number of training epochs> <path of input file 1 for training> <path of input file 2 for training> <path of input file training> <path of output folder> 

Training epochs should be set at least 2.
	
After entering 5 arguments, program prints out all arguments it's just read. And user can check again to ensure it reads properly.

The main class is senti_classify. Normally, hadoop will ask for main class name after jar file name.

	ex: hadoop jar senti_classify_small.jar neuro.mre.senti_classify 2 /user/cloudera/MapReduce/cloth_train_sample1.txt /user/cloudera/MapReduce/cloth_train_sample2.txt /user/cloudera/MapReduce/cloth_test_sample.txt /user/cloudera/MapReduce/cloth_output_sample

However in other times, because there only 1 executing class in this jar file, hadoop can ignore this argument, and program receives this name class as a new first argument. Please delete class name if you see so after you enter executing command.

	ex: hadoop jar senti_classify_small.jar 2 /user/cloudera/MapReduce/cloth_train_sample1.txt /user/cloudera/MapReduce/cloth_train_sample2.txt /user/cloudera/MapReduce/cloth_test_sample.txt /user/cloudera/MapReduce/cloth_output_sample
	
Program runs 3 mapreduce jobs (each job can has many tasks): 2 jobs for training (2 epochs, after each epoch, it will update average neurons' weights for next epoch), and 1 job for testing (takes neurons' weights from training steps)
	
Finally, it generates results in output folder. Intermediate data of computing process will be stored in two temporary folders beside output folder.


