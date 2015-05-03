package neuro.mre;

import neuro.core.*;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.StringTokenizer;

import java.net.URL;
import java.lang.*;

//import java.io.InputStreamReader;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.ReduceContext;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.Reducer.Context;
//import org.apache.hadoop.mapreduce.filecache.DistributedCache;
import org.apache.hadoop.mapreduce.lib.chain.ChainMapper;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

import com.sun.tools.internal.xjc.reader.xmlschema.bindinfo.BIConversion.Static;

import neuro.core.*;


/**
 * Neural Network in MapReduce!
 *
 */

public class senti_classify
{
	//Network parameters

	private static int iteration = 0;
//	private static int path_index = 0; //indicate which temporary path to store weights 
	
//	private static String str_path_1 = "";
//	private static String str_path_2 = "";
	
	private static Path inPath_temp_1;
	private static Path inPath_temp_2;
	private static Path inPath;
	
	private static int epochs;
	

	//counter & stat
	public static final String CONFUSION_MATRIX_COUNTER_GROUP = "ACCURACY";




	//
	//MAP_TRAINING
	//
	  public static class Map_Training extends Mapper<LongWritable, Text, Text, Text>
	  {
		  //Network parameters
			private static double		learningRate = 0.1;
			private static double		momentum = 0.0;
			private static  double		sigmoidAlphaValue = 2.0;
			private static double		learningErrorLimit = 0.1;
			private static int			sigmoidType = 0;

			
			private static int inputsCount = 5; //input for first layer
			private static int [] neuronsCount = {5,5,2}; //layersCount = neuronsCount.length; previous layer is input for next layer
			private static double [][][] weights = {{},{},{}}; //store weight for 2nd epock, and for testing
			private static double [][][] biases = {{},{},{}};	
		  
		  
			ActivationNetwork network;
			BackPropagationLearning trainner;
		    
			//map parameters
			private Text out_key = new Text();
		    private Text out_val = new Text();
		    
		    double error = learningErrorLimit;
		    ArrayList errorsList = new ArrayList( );
		    int count = 0;
		    Boolean cont = true;
		    public static final String RECORDS_COUNTER_NAME = "Records";
		   
		    //object parameters
		    int dim_max = inputsCount;
		    String input_string = null;
		    Path input_temp_1 = null;
		    Path input_temp_2 = null;
		    Path common_path = null;
		    Path common_folder = null;
		    FileSystem fst = null;
		    
		    String accuracy;
			int tp_pp = 0;
			int tp_pn = 0;
			int tn_pp = 0;
			int tn_pn = 0;
			double best_accuracy = 0;
			int best_ep = 0;
			
			String last_ep;
		    
			
			//***********************************************
			//READ NEURON NEURONS' WEIGHTS FROM PREVIOUS EPOCH
		    public void initialize(Context context) throws IOException, InterruptedException 
		    {
		    	Configuration conf = context.getConfiguration();
		    	
		    	//initialize arrays to store neural network's paramters
				for (int i=0; i < neuronsCount.length; i++)
				{
					if (neuronsCount[i] > dim_max)
						dim_max = neuronsCount[i];
				}
				weights = new double [neuronsCount.length][dim_max][dim_max];
		    	biases =  new double [neuronsCount.length][dim_max][dim_max]; 
		    	

				
		    	
		    	input_string = conf.get("temp_string");
		    	
		    	
//				input_string = ""; //for local mode & debug
				input_temp_1 = new Path(input_string + "_temp1/_SUCCESS");
				input_temp_2 = new Path(input_string + "_temp2/_SUCCESS");
				
				
				
				fst = FileSystem.get(conf);
				
			
				
				if(fst.exists(input_temp_1))
			    {
					common_path = new Path(input_string + "_temp1/part-r-00000");
					common_folder = new Path(input_string + "_temp1");
			    }
				else if (fst.exists(input_temp_2))
				{
					common_path = new Path(input_string + "_temp2/part-r-00000");
					common_folder = new Path(input_string + "_temp2");
				}
				else
				{
					network = new ActivationNetwork (new BipolarSigmoidFunction(sigmoidAlphaValue), inputsCount, neuronsCount); //use constructor for training (without pre-set weights)
					trainner = new BackPropagationLearning( network, learningRate, momentum);
				}
				
				if (common_path != null)
				{
//					
					
					try
		  			{                    
						
						//READ ALL FILES IN OUPUT FOLDER OF PREVIOUS EPOCH
						FileStatus[] status_list = fst.listStatus(common_folder);
						if(status_list != null)
						{
							for(FileStatus status : status_list)
							{
							    //add each file to the list of inputs for the map-reduce job
								if (status.isFile())
								{
									BufferedReader br=new BufferedReader(new InputStreamReader(fst.open(status.getPath())));	
													
	                    	                        
						            String line;
			                        line=br.readLine();
			                        int last_length = 0;
			                        int current_length = 0;
			                               
			                        while (line != null)
			                        {	
			                        	String [] str_line = line.split("\t");
			                        	String [] str_index = str_line[0].split("_");
			                        	String [] str_kval =  str_line[1].split("_");
			                        	current_length = str_kval.length;
			                        	
			                        	if (current_length == 2 & current_length == last_length)
			                        	{
			                        		if (str_index.length >= 3)
			                        		{
					                        	int layer = Integer.valueOf(str_index[1]);
					                        	int neuron = Integer.valueOf(str_index[2]);
					                        	int wi = Integer.valueOf(str_index[3]);
					                        	double wgt = Double.valueOf(str_kval[0]);
					                        	double bias = Double.valueOf(str_kval[1]);
				                        		
						                        
					                        	weights[layer][neuron][wi] = wgt;
					                        	biases[layer][neuron][wi] = bias;
				                        		
					                        	
					                            System.out.println("Read Weights for Training: " + line);
			                        		}
			                        	}
			                        	last_length = current_length;
			                            line=br.readLine();
//			                            line=br.readLine();
			                        } //while read lines
	                        
							} //is files, not dir
	                        
						  } //whie read many files
							    
						} //end satus has files
		  			}
		  			catch(Exception e)
		            {
		  				
		            }
		  		//initialize slave network
		  		network = new ActivationNetwork (new BipolarSigmoidFunction(sigmoidAlphaValue), inputsCount, neuronsCount, weights, biases); //use constructor for training (with pre-set weights)
		  		trainner = new BackPropagationLearning( network, learningRate, momentum);
				}
		    }
		    
		    //SAVE NEURONS' WEIGHTS AFTER TRAINING
		    public void finalize(Context context) throws IOException, InterruptedException 
		    {
		    	for (int i=0, l=network.layersCount; i<l; i++)
				{
					for (int j=0, n=network.layers[i].neuronsCount; j<n; j++)
														
					{
						for (int k=0, w=network.layers[i].neurons[j].inputsCount; k<w; k++)
						{

							double wgt = network.layers[i].neurons[j].weights[k];
							double bias = network.layers[i].neurons[j].bias;
							String vtext = String.valueOf(count) + "_" + String.valueOf(wgt) + "_" + String.valueOf(error) + "_" + String.valueOf(bias);
							out_val.set(vtext);
							
							String ktext = "w_" + String.valueOf(i) + "_" + String.valueOf(j) + "_" + String.valueOf(k);
							out_key.set(ktext);
							
							context.write(out_key, out_val);
							System.out.println("Map: " + ktext + "  " + vtext);
							
						}
						
						
					}
					
				}
		    	
		    	String vtext = String.valueOf(count) + "_" + String.valueOf(tp_pp) + "_" + String.valueOf(tp_pn) + "_" + String.valueOf(tn_pp) + "_" + String.valueOf(tn_pn);
				out_val.set(vtext);
				String ktext = "PREDICTION";
				out_key.set(ktext);
		    	context.write(out_key, out_val);
		    	
		    	
		    	
		    }
		    
		    @Override
		    public void run(Context context) throws IOException, InterruptedException 
		    {
		        setup(context);
		        initialize(context); //setup neural network paramters
		        
		        try 
		        {
		          while (context.nextKeyValue()) 
		          {
		            map(context.getCurrentKey(), context.getCurrentValue(), context);
		          }
		          
		          
		          finalize(context); //write neural network parameters
		          
		        } 
		        finally 
		        {
		        	cleanup(context);
		        }
		     }
		    

		    //read split of data file
		    public void map (LongWritable key, Text value, Context context) throws IOException, InterruptedException
		    {
		    	
		    
		    	 
			      String [] pval = value.toString().trim().split("[ \t]+");
			      count += 1;
			      
			      
			      //use for loop to parse with known dimension vector
			      double input[][] = new double [1][inputsCount]; //Map read 1 record each time       
			      for (int i=0; i< inputsCount; i++)
			      {
			    	  String pvi = pval[i];
			    	  input[0][i] = Double.parseDouble(pvi); //Map read 1 record each time       
			    	  
			      }
			      double output[][] = new double [1][neuronsCount[neuronsCount.length  - 1]];	//ideal ouput for training, read 1 record
			      for (int i=0; i< output[0].length; i++)
			      {
			    	  String pvi = pval[pval.length - output[0].length + i - 1]; //ingonre the last column which is used for RapidMiner
			    	  output[0][i] = Double.parseDouble(pvi); //Map read 1 record each time       
			    	  
			      }


				        // run epoch of learning procedure
						error += trainner.RunEpoch( input, output );
						errorsList.add( error );
						System.out.println("MAP TRAINING error: " + String.valueOf(error));

						
						double k = 0;
						double computed_output = 0;
						double ideal_output;
						String str_ideal_output = "";
						String str_computed_output = "";
						
						for (int i=0; i< output[0].length; i++)
						{
							str_ideal_output += "_" + String.valueOf(output[0][i]);
							str_computed_output += "_" + String.valueOf(network.layers[neuronsCount.length - 1].neurons[i].output);
							
							if (computed_output < network.layers[neuronsCount.length - 1].neurons[i].output)
							{
								computed_output = network.layers[neuronsCount.length - 1].neurons[i].output;
								k = i;
							}
						}
						
						String class_result = "";
						
						if (output[0][0] == 1.0) //ideal output: positive
						{
							if (k == 0.0)
							{
								tp_pp++;
								class_result = "True PO_Predict PO";
								System.out.println(class_result + ": " + tp_pp);
							}
//							else if (k == 1)
//							{
//								
//							}
							else
							{
								tp_pn++;
								class_result = "True PO_Predict NE";
								System.out.println(class_result + ": " + tp_pn);
							}
							
						}
//						else if (output[0][1] == 1.0) //ideal output: neutral
//						{
//
//						}
						else //ideal output: negative
						{

							if (k == 0.0)
							{
								tn_pp++;
								class_result = "True NE_Predict PO";
								System.out.println(class_result + ": " + tn_pp);
							}
//							else if (k == 1)
//							{
//							}
							else
							{
								tn_pn++;
								class_result = "True NE_Predict NE";
								System.out.println(class_result + ": " + tn_pn);
							}
						}
						
						
					
//			      }
//		    	} //end while(nextKey())
		    	
		    	
		    	


		    } //end map function


	  } // end map class






	  //
	  //REDUCE_TRANNING
	  //
	  public static class Reduce_Training extends Reducer<Text,Text,Text,Text>
	  {


	    	Boolean has_key = true;
	    
	        private Text rkey = new Text();
			private Text result = new Text();
//		    public static final String SLAVES_COUNTER_NAME = "Groups";
		    
		    @Override
		    public void run(Context context) throws IOException, InterruptedException 
		    {
		        setup(context);
		        initialize();
		        try 
		        {
		          while (context.nextKey()) 
		          {
		            reduce(context.getCurrentKey(), context.getValues(), context);
		            // If a back up store is used, reset it
		            Iterator<Text> iter = context.getValues().iterator();
		            if(iter instanceof ReduceContext.ValueIterator) 
		            {
		              ((ReduceContext.ValueIterator<Text>)iter).resetBackupStore();        
		            }
		          }
		          
		          
		         finalize();
		        } 
		        finally 
		        {
		          cleanup(context);
		        }
		    }
		    
		    public void initialize()
		    {
		    	
		    }
		    
		    public void finalize()
		    {
		    	
		    }
		    
		    

		    public void reduce(Text key,  Iterable<Text> values, Context context) throws IOException, InterruptedException
		    { 
		    	//summarize weights from all mappers
		    	double wgt;
		    	int count;
		    	double error = 0;
		    	double bias = 0;
		    	double sum_bias = 0;
		    	double avg_bias = 0;
		    	double sum_count = 0;
		    	double sum_wgt = 0;
		    	double avg_wgt = 0;
		    	String avg__wgt_str;
		    	String avg_bias_str;
		    	
		    	//summarize accuracy from all mappers
		    	double accuracy = 0;
				int tp_pp = 0;
				int tp_pn = 0;
				int tn_pp = 0;
				int tn_pn = 0;
				int sum_tp_pp = 0;
				int sum_tp_pn = 0;
				int sum_tn_pp = 0;
				int sum_tn_pn = 0;
				int m_count = 0;
				int m_sum = 0;
				String accuracy_str = "";
				int current_ep = 0;
				
				double best_accuracy = 0;
				int best_ep = 0;
				String best_str = "";
				
				Configuration conf = context.getConfiguration();
		    	
//		    	while (has_key == true)
//		    	{
		    	
		    	int loop = 0;
		    	
		    	String r_key = key.toString();
		    	rkey.set(r_key);
		    	
		    	//use array to store value (can iterate many rounds, while iterator object only allow iterate 1 round)
		    	ArrayList<Text> cache = new ArrayList<Text>(); //need to cache object, not value
		    	int i = 0;
		    	
		    	String m_result = "";
		    	
			      for (Text val : values)
			      {
			    	  cache.add(val);
			    	  i++;
			    	  
			    	  m_result = cache.get(i-1).toString(); //get string of value from cache
			    	  
			    	  String [] pval = m_result.split("_");
			    	 		    	  		    	  
		    		  
			    	  if (pval.length > 2)
			    	  {
			    		  if (!r_key.equals("PREDICTION")) //summarize accuracy
					    	{
					    	  count = Integer.valueOf(pval[0]);
					    	  wgt = Double.valueOf(pval[1]);
					    	  error = Double.valueOf(pval[2]);
					    	  bias = Double.valueOf(pval[3]);
					    	  
					    	  sum_count += count;
					    	  sum_wgt += count*wgt;
					    	  sum_bias += count*bias;
					    	  
					    	}
			    		  else
			    		  {
					    	  m_count = Integer.valueOf(pval[0]); //summarzie neurons'weights
				    		  tp_pp = Integer.valueOf(pval[1]);
				    		  tp_pn = Integer.valueOf(pval[2]);
				    		  tn_pp = Integer.valueOf(pval[3]);
				    		  tn_pn = Integer.valueOf(pval[4]);
					    	  
				    		  m_sum		+= m_count;
				    		  sum_tp_pp += tp_pp;
				    		  sum_tp_pn += tp_pn;
				    		  sum_tn_pp += tn_pp;
				    		  sum_tn_pn += tn_pn;
			    		  }
			    	  }
			    	  else
			    	  {
			    		  loop++; //monitor loop of reduce function
			    		  
			    		  
			    	  }
			    	  
			    	  context.write(key, val);
			      }
			      
			      if (sum_count > 0) //summarzie neurons'weights
			      {
			    	  avg_bias = sum_bias/sum_count;
				      avg_wgt = sum_wgt/sum_count;
				      avg__wgt_str = String.valueOf(avg_wgt);
				      avg_bias_str = String.valueOf(bias);
				      result.set(avg__wgt_str + "_" + avg_bias_str);
				      System.out.println("Reduce: " + rkey.toString() + "  " + result.toString() + " (" + loop + ")");// + "  " + String.valueOf(error));

			      }
			      
			      if (m_sum > 0) //summarize accuracy
			      {
			    	  accuracy = (sum_tp_pp + sum_tn_pn + 0.0)/m_sum;
			    	  current_ep = Integer.valueOf(conf.get("last_ep")) + 1;
			    	  accuracy_str = accuracy + "_" + current_ep;
				      result.set(accuracy_str);
				      System.out.println("Reduce Accuracy: " + rkey.toString() + "  " + result.toString() + " (" + loop + ")");// + "  " + String.valueOf(error));
				      
			      }	    	

		    	
			  context.write(rkey, result);

		      
		    }
	    	    	    	    	   	    
	  }


	




	  //DRIVER PROGRAM
	  public static void main(String[] args) throws Exception
	  {
		//*********************************************************************************************
		//TRAINING JOB
		  
	    Configuration conf = new Configuration();
	    String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs(); // get all args
	    System.out.println(Arrays.toString(args));
	    
	        
	    Job training_job = null;
	    
	    //read number of training epochs
		epochs = Integer.valueOf(otherArgs[otherArgs.length-5]);
		String odd;
		if (epochs%2 == 2)
		{
			odd = "0";
		}
		else
		{
			odd = "1";
		}
		conf.set("epochs", odd);
		
	        
	    // set the HDFS path
	    String input_string2 = otherArgs[otherArgs.length-3];
		String input_string = otherArgs[otherArgs.length-4];
		inPath = new Path(input_string);
		
        //set temp path for mappers to find weights from previous epoch
		String [] input_array = input_string.split(".txt");
		input_string = input_array[0];
		conf.set("temp_string", input_string);
		
        //set 2 temporary folders to store training weights
		inPath_temp_1 = new Path(input_string + "_temp1");
		inPath_temp_2 = new Path(input_string + "_temp2");
	    
	    	    
	    FileSystem fs = FileSystem.get(conf);
	    
//	    if (fs.exists(inPath))
//	    {
//		    //test reading input file in hdfs
//		    BufferedReader br=new BufferedReader(new InputStreamReader(fs.open(inPath)));	
//		    String line = br.readLine();
//		    System.out.println(line);
//		    
//		    System.out.println(inPath_temp_1);
//		    System.out.println(inPath_temp_2);
//	    }
//	    else
//	    {
//	    	System.out.println("PATH ERROR");
//	    }
	    
	    
	    
	 //Training through nunmber of Epochs
	    int code_training = 0;
	    for (int ep=1; ep<=epochs; ep++)
	    {
	    	conf.set("last_ep", String.valueOf(ep-1));
	    	// create a job with name "training_job"
		    training_job = new Job(conf, "training_job");
		    
		    training_job.setJarByClass(senti_classify.class);
		    training_job.setMapperClass(Map_Training.class);
		    training_job.setReducerClass(Reduce_Training.class);

		    //add the Combiner
		    training_job.setCombinerClass(Reduce_Training.class);

		    //set output key type (for Map)
		    training_job.setMapOutputKeyClass(Text.class); 
		    training_job.setMapOutputValueClass(Text.class);

		    // set output key type (for Reduce, and for all if not explicitly declare above)
		    training_job.setOutputKeyClass(Text.class);
		    // set output value type
		    training_job.setOutputValueClass(Text.class);
		    //set the HDFS path of the input data
//		    FileInputFormat.setInputDirRecursive(training_job, true); //read all files in folder => use for validation stage
		    FileInputFormat.addInputPath(training_job, inPath);
		    FileInputFormat.addInputPath(training_job, new Path(input_string2));
		    
		    
		    
	    	if (code_training == 0)
		    {
	    		iteration = ep;
			    if (ep%2 == 0)
			    {
				    
				    // set the HDFS path for the output
			    	/*Check if output path (args[1])exist or not*/
				    if(fs.exists(inPath_temp_2))
				    {
				       /*If exist delete the output path*/
				       fs.delete(inPath_temp_2,true);
				    }
				    FileOutputFormat.setOutputPath(training_job, inPath_temp_2);
//				    path_index = 2;
				    
			    }
			    else
			    {
				    
				    // set the HDFS path for the output
			    	/*Check if output path (args[1])exist or not*/
				    if(fs.exists(inPath_temp_1))
				    {
				       fs.delete(inPath_temp_1,true);
				    }
				    FileOutputFormat.setOutputPath(training_job, inPath_temp_1);
//				    path_index = 1;
			    }
			    
			    
			 // Execute tranning_job and grab exit code
			    
			    code_training = training_job.waitForCompletion(true) ? 0 : 1;
			    
		    }
	    }




	    if (code_training == 0) //if training job succeeded
	    {
		

			// Setup testing job
			Job testing_job = new Job(conf, "testing_job");

			testing_job.setJarByClass(senti_classify.class);
		    testing_job.setMapperClass(Map_Testing.class);
		    testing_job.setReducerClass(Reduce_Testing.class);

		    //add the Combiner
		    testing_job.setCombinerClass(Reduce_Testing.class);

		    //no reduce task for this job
		    testing_job.setNumReduceTasks(0);
		    
		    
		    
		    // set output key type
		    testing_job.setOutputKeyClass(Text.class);
		    // set output value type
		    testing_job.setOutputValueClass(IntWritable.class);
		    //set the HDFS path of the input data
		    FileInputFormat.addInputPath(testing_job, new Path(otherArgs[otherArgs.length-2]));
		    
		    
		    // set the HDFS path for the output
		    /*Check if output path (args[otherArgs.length-1])exist or not*/
		    if(fs.exists(new Path(args[otherArgs.length-1])))
		    {
		       /*If exist delete the output path*/
		       fs.delete(new Path(otherArgs[otherArgs.length-1]),true);
		    }
		   
		    FileOutputFormat.setOutputPath(testing_job, new Path(otherArgs[otherArgs.length-1]));

		    // Execute tranning_job and grab exit code
		    int code_testing = testing_job.waitForCompletion(true) ? 0 : 1;

		    if (code_testing == 0) //if job testing job succeeded
		    {
				//**************************************************
		    	//GET COUNTERS' VALUES FOR CLASSIFICATION REUSLTS
		    	
				double tpo_ppo = (double) testing_job.getCounters().findCounter(CONFUSION_MATRIX_COUNTER_GROUP, Map_Testing.TPO_PPO_COUNTER).getValue();
				double tpo_pnu = (double) testing_job.getCounters().findCounter(CONFUSION_MATRIX_COUNTER_GROUP, Map_Testing.TPO_PNU_COUNTER).getValue();
				double tpo_png = (double) testing_job.getCounters().findCounter(CONFUSION_MATRIX_COUNTER_GROUP, Map_Testing.TPO_PNG_COUNTER).getValue();
				
				double tnu_ppo = (double) testing_job.getCounters().findCounter(CONFUSION_MATRIX_COUNTER_GROUP, Map_Testing.TNU_PPO_COUNTER).getValue();
				double tnu_pnu = (double) testing_job.getCounters().findCounter(CONFUSION_MATRIX_COUNTER_GROUP, Map_Testing.TNU_PNU_COUNTER).getValue();
				double tnu_png = (double) testing_job.getCounters().findCounter(CONFUSION_MATRIX_COUNTER_GROUP, Map_Testing.TNU_PNG_COUNTER).getValue();
				
				double tng_ppo = (double) testing_job.getCounters().findCounter(CONFUSION_MATRIX_COUNTER_GROUP, Map_Testing.TNG_PPO_COUNTER).getValue();
				double tng_pnu = (double) testing_job.getCounters().findCounter(CONFUSION_MATRIX_COUNTER_GROUP, Map_Testing.TNG_PNU_COUNTER).getValue();
				double tng_png = (double) testing_job.getCounters().findCounter(CONFUSION_MATRIX_COUNTER_GROUP, Map_Testing.TNG_PNG_COUNTER).getValue();
							
				
				double accuracy = (tpo_ppo + tng_png)*100/(tpo_ppo + tpo_png + tng_ppo + tng_png);
				double precision_pos = (tpo_ppo)/(tpo_ppo + tng_ppo);
				double precision_neg = (tng_png)/(tpo_png + tng_png);
				double recall_pos = (tpo_ppo)/(tpo_ppo + tpo_png);
				double recall_neg = (tng_png)/(tng_ppo + tng_png);
					
				
				System.out.println("True PO_Predict PO: " + tpo_ppo);
				System.out.println("True PO_Predict NG: " + tpo_png);
				System.out.println("True NG_Predict PO: " + tng_ppo);
				System.out.println("True NG_Predict NG: " + tng_png);
				
				System.out.println("Accuracy: " + accuracy);
				System.out.println("Precison (positive): " + precision_pos);
				System.out.println("Precison (negative): " + precision_neg);
				System.out.println("Recall (positive): " + recall_pos);
				System.out.println("Recall (negative): " + recall_neg);			

		    }

	  }


	  }

	  

	  //
	  //MAP_TESTING
	  //
	  public static class Map_Testing extends Mapper<LongWritable, Text, Text, Text>
	  {

		    //////////////////////////////////////////////////////////////
		    //Network parameters
			private static double		learningRate = 0.1;
			private static double		momentum = 0.0;
			private static  double		sigmoidAlphaValue = 2.0;
			private static double		learningErrorLimit = 0.1;
			private static int			sigmoidType = 0;

			
			private static int inputsCount = 5; //input for first layer
			private static int [] neuronsCount = {5,5,2}; //layersCount = neuronsCount.length; previous layer is input for next layer
			private static double [][][] weights = {{},{},{}}; //store weight for 2nd epock, and for testing
			private static double [][][] biases = {{},{},{}};	
		  
		  
			ActivationNetwork network;
			BackPropagationLearning tester;
		    
			//map parameters
			private Text out_key = new Text();
		    private Text out_val = new Text();
		    
		    double error = learningErrorLimit;
		    ArrayList errorsList = new ArrayList( );
		    int count = 0;
		    Boolean cont = true;
		    public static final String RECORDS_COUNTER_NAME = "Records";
		   
		    //object parameters
		    int dim_max = inputsCount;
		    String input_string = null;
		    Path input_temp_1 = null;
		    Path input_temp_2 = null;
		    Path common_path = null;
		    Path common_folder = null;
		    FileSystem fst = null;
		    
		  //**************************************************
			//READ NEURON NEURONS' WEIGHTS FROM PREVIOUS EPOCH
		    public void initialize(Context context) throws IOException, InterruptedException 
		    {
		    	Configuration conf = context.getConfiguration();
		    	
		    	//initialize arrays to store neural network's parameters
				for (int i=0; i < neuronsCount.length; i++)
				{
					if (neuronsCount[i] > dim_max)
						dim_max = neuronsCount[i];
				}
				weights = new double [neuronsCount.length][dim_max][dim_max];
		    	biases =  new double [neuronsCount.length][dim_max][dim_max]; 
		    	
				
		    	input_string = conf.get("temp_string");
		    	
//				input_string = ""; //for local mode and debug
				input_temp_1 = new Path(input_string + "_temp1/_SUCCESS");
				input_temp_2 = new Path(input_string + "_temp2/_SUCCESS");
				
				
				
				fst = FileSystem.get(conf);
				
				
				String odd = conf.get("epochs");
				if (odd.equals("0"))
				{
					common_path = new Path(input_string + "_temp2/part-r-00000");
					common_folder = new Path(input_string + "_temp2");
				}
				else
				{
					common_path = new Path(input_string + "_temp1/part-r-00000");
					common_folder = new Path(input_string + "_temp1");
				}
				
				
				if (common_path != null)
				{
//					
					
					try
		  			{
	                        
						
						//READ ALL FILES IN OUPUT FOLDER OF PREVIOUS EPOCH
						FileStatus[] status_list = fst.listStatus(common_folder);
						if(status_list != null)
						{
							for(FileStatus status : status_list)
							{
								if (status.isFile())
								{
									BufferedReader br=new BufferedReader(new InputStreamReader(fst.open(status.getPath())));	
							
							
                  
	                     
						            String line;
			                        line=br.readLine();
			                        int last_length = 0;
			                        int current_length = 0;
			                               
			                        while (line != null)
			                        {	
			                        	String [] str_line = line.split("\t");
			                        	String [] str_index = str_line[0].split("_");
			                        	String [] str_kval =  str_line[1].split("_");
			                        	current_length = str_kval.length;
			                        	
			                        	if (current_length == 2 & current_length == last_length)
			                        	{
			                        		if (str_index.length >= 3)
			                        		{
					                        	int layer = Integer.valueOf(str_index[1]);
					                        	int neuron = Integer.valueOf(str_index[2]);
					                        	int wi = Integer.valueOf(str_index[3]);
					                        	double wgt = Double.valueOf(str_kval[0]);
					                        	double bias = Double.valueOf(str_kval[1]);
						                        
					                        	weights[layer][neuron][wi] = wgt;
					                        	biases[layer][neuron][wi] = bias;
					                        	
					                            System.out.println("Read Weights for Testing: " + line);
			                        		}
			                        	}
			                        	last_length = current_length;
			                            line=br.readLine();
//			                            line=br.readLine();
			                        } //while read lines
	                        
							} //is files, not dir
	                        
						  } //whie read many files
							    
						} //end satus has files
		  			}
		  			catch(Exception e)
		            {
		  				
		            }
		  		//initialize slave network
		  		network = new ActivationNetwork (new BipolarSigmoidFunction(sigmoidAlphaValue), inputsCount, neuronsCount, weights, biases); //use constructor for training (with pre-set weights)
		  		tester = new BackPropagationLearning( network, learningRate, momentum);
				}
		    }
		    
		    
		    public void finalize(Context context) throws IOException, InterruptedException 
		    {
	
		    }
		    
		    @Override
		    public void run(Context context) throws IOException, InterruptedException 
		    {
		        setup(context);
		        initialize(context); //setup neural network paramters
		        
		        try 
		        {
		          while (context.nextKeyValue()) 
		          {
		            map(context.getCurrentKey(), context.getCurrentValue(), context);
		          }
		          
		          
//		          finalize(context); //write neural network parameters
		          
		        } 
		        finally 
		        {
		        	cleanup(context);
		        }
		     }
		    

		    /////////////////////////////////////////////////////////////
		    
		    	      
		    
//		    public static final String RECORDS_COUNTER_NAME = "Records";
		    public static final String TPO_PPO_COUNTER = "TRUE POSTIVE_PREDICT POSITIVE";
		    public static final String TPO_PNU_COUNTER = "TRUE POSTIVE_PREDICT NEUTRAL";
		    public static final String TPO_PNG_COUNTER = "TRUE POSTIVE_PREDICT NEGATIVE";
		    
		    public static final String TNU_PPO_COUNTER = "TRUE NEUTRAL_PREDICT POSITIVE";
		    public static final String TNU_PNU_COUNTER = "TRUE NEUTRAL_PREDICT NEUTRAL";
		    public static final String TNU_PNG_COUNTER = "TRUE POSTIVE_PREDICT NEGATIVE";
		    
		    public static final String TNG_PPO_COUNTER = "TRUE NEGATIVE_PREDICT POSITIVE";
		    public static final String TNG_PNU_COUNTER = "TRUE NEGATIVE_PREDICT NEUTRAL";
		    public static final String TNG_PNG_COUNTER = "TRUE NEGATIVE_PREDICT NEGATIVE";
		    

		    //read sub data file (smaller than default hadoop block size) => then calculate independently (parallel)
		    public void map (LongWritable key, Text value, Context context) throws IOException, InterruptedException
		    {

		    	// each time, mapper read 1 line of csv file content
		    		String [] pval = value.toString().trim().split("[ \t]+");
//			      String [] pval = value.toString().split("\t");			      
			      count += 1;
			      
			      
			    //use for loop to parse with known dimension vector
			      
			      double input[][] = new double [1][inputsCount]; //Map read 1 record each time       
			      for (int i=0; i< inputsCount; i++)
			      {
			    	  String pvi = pval[i];
			    	  input[0][i] = Double.parseDouble(pvi); //Map read 1 record each time       
			    	  
			      }
			      double output[][] = new double [1][neuronsCount[neuronsCount.length  - 1]];	//ideal ouput for training, read 1 record
			      for (int i=0; i< output[0].length; i++)
			      {
			    	  String pvi = pval[pval.length - output[0].length + i - 1]; //the last column is used for RapidMiner
			    	  output[0][i] = Double.parseDouble(pvi); //Map read 1 record each time
			    	  
			    	  
			      }


				        // run epoch of learning procedure
						error += tester.RunEpoch( input, output );
						errorsList.add( error );
						System.out.println("MAP Testing error: " + String.valueOf(error));

						
						double k = 0;
						double computed_output = 0;
						double ideal_output;
						String str_ideal_output = "";
						String str_computed_output = "";
						
						for (int i=0; i< output[0].length; i++)
						{
							str_ideal_output += "_" + String.valueOf(output[0][i]);
							str_computed_output += "_" + String.valueOf(network.layers[neuronsCount.length - 1].neurons[i].output);
							
							if (computed_output < network.layers[neuronsCount.length - 1].neurons[i].output)
							{
								computed_output = network.layers[neuronsCount.length - 1].neurons[i].output;
								k = i;
							}
						}
						
						String class_result = "";
						
						if (output[0][0] == 1.0) //ideal output: positive
						{
							if (k == 0.0)
							{
								long cter = context.getCounter(CONFUSION_MATRIX_COUNTER_GROUP, TPO_PPO_COUNTER).getValue();
								context.getCounter(CONFUSION_MATRIX_COUNTER_GROUP, TPO_PPO_COUNTER).increment(1);
								cter = context.getCounter(CONFUSION_MATRIX_COUNTER_GROUP, TPO_PPO_COUNTER).getValue();
								class_result = "True PO_Predict PO";
								System.out.println(class_result + ": " + cter);
							}
							else
							{
								long cter = context.getCounter(CONFUSION_MATRIX_COUNTER_GROUP, TPO_PNG_COUNTER).getValue();
								context.getCounter(CONFUSION_MATRIX_COUNTER_GROUP, TPO_PNG_COUNTER).increment(1);
								cter = context.getCounter(CONFUSION_MATRIX_COUNTER_GROUP, TPO_PNG_COUNTER).getValue();
								class_result = "True PO_Predict NG";
								System.out.println(class_result + ": " + cter);
							}
							
						}

						else //ideal output: negative
						{

							if (k == 0.0)
							{
								long cter = context.getCounter(CONFUSION_MATRIX_COUNTER_GROUP, TNG_PPO_COUNTER).getValue();
								context.getCounter(CONFUSION_MATRIX_COUNTER_GROUP, TNG_PPO_COUNTER).increment(1);
								cter = context.getCounter(CONFUSION_MATRIX_COUNTER_GROUP, TNG_PPO_COUNTER).getValue();
								class_result = "True NG_Predict PO";
								System.out.println(class_result + ": " + cter);
							}

							else
							{
								long cter = context.getCounter(CONFUSION_MATRIX_COUNTER_GROUP, TNG_PNG_COUNTER).getValue();
								context.getCounter(CONFUSION_MATRIX_COUNTER_GROUP, TNG_PNG_COUNTER).increment(1);
								cter = context.getCounter(CONFUSION_MATRIX_COUNTER_GROUP, TNG_PNG_COUNTER).getValue();
								class_result = "True NG_Predict NG";
								System.out.println(class_result + ": " + cter);
							}
						}
						
						
						

						//SAVE CLASSIFICATION RESULTS
						String ktext = String.valueOf(count);
						out_key.set(ktext);
						String vtext = str_ideal_output + "_" + str_computed_output + "_" + class_result;
						out_val.set(vtext);
						context.write(out_key, out_val);
						System.out.println("Map: " + ktext + "  " + vtext);
						
 	



		    } //end map function


	  } //end map class

	  //
	  //REDUCE_TESTING
	  //
	  public static class Reduce_Testing extends Reducer<Text,IntWritable,Text,IntWritable>
	  {



	  }



}
