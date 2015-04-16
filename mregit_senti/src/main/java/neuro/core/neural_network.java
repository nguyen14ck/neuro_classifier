package neuro.core;

import java.util.*;

public class neural_network {


//    final boolean isTrained = false;
//    final DecimalFormat df;
//    final Random rand = new Random();
//    final ArrayList<Neuron> inputLayer = new ArrayList<Neuron>();
//    final ArrayList<Neuron> hiddenLayer = new ArrayList<Neuron>();
//    final ArrayList<Neuron> outputLayer = new ArrayList<Neuron>();
//    final Neuron bias = new Neuron();
//    final int[] layers;
//    final int randomWeightMultiplier = 1;
//
//    final double epsilon = 0.00000000001;
//
//    final double learningRate = 0.9f;
//    final double momentum = 0.7f;
//
//    // Inputs for xor problem
//    final double inputs[][] = { { 1, 1 }, { 1, 0 }, { 0, 1 }, { 0, 0 } };
//
//    // Corresponding outputs, xor training data
//    final double expectedOutputs[][] = { { 0 }, { 1 }, { 1 }, { 0 } };
//    double resultOutputs[][] = { { -1 }, { -1 }, { -1 }, { -1 } }; // dummy init
//    double output[];
//
//    // for weight update all
//    final HashMap<String, Double> weightUpdate = new HashMap<String, Double>();

	private double		learningRate = 0.1;
	private double		momentum = 0.0;
	private double		sigmoidAlphaValue = 2.0;
	private double		learningErrorLimit = 0.1;
	private int			sigmoidType = 0;

//	// initialize input and output values
//				double[][] input = null;
//				double[][] output = null;
//
//				if ( sigmoidType == 0 )
//				{
//					// unipolar data
//					input = new double[4][] {
//												new double[] {0, 0},
//												new double[] {0, 1},
//												new double[] {1, 0},
//												new double[] {1, 1}
//											};
//					output = new double[4][] {
//												 new double[] {0},
//												 new double[] {1},
//												 new double[] {1},
//												 new double[] {0}
//											 };

					 private double input[][] =
						 {
						      {0, 0 },
						      {0, 1 },
						      {1, 0 },
						      {1, 1 }
					      };
					 private double output[][] =
						 {
						      { 0 },
						      { 1 },
						      { 0 },
						      { 1 }
					      };

					 ActivationNetwork	network;




	public neural_network()
	{
		// create perceptron
//		anetwork = new ActivationNetwork(
//			( sigmoidType == 0 ) ?
//				(IActivationFunction) new SigmoidFunction( sigmoidAlphaValue ) :
//				(IActivationFunction) new BipolarSigmoidFunction( sigmoidAlphaValue ),
//			2, 2, 1 );

		int [] neuronsCount = {2,1};
		network = new ActivationNetwork (new SigmoidFunction(sigmoidAlphaValue), 2, neuronsCount);

//		double [][] weights = {{0.5},{0.5}};
		// create teacher
		BackPropagationLearning teacher = new BackPropagationLearning( network, learningRate, momentum);
		// set learning rate and momentum
		teacher.LearningRate	= learningRate;
		teacher.Momentum		= momentum;

		// iterations
		int iteration = 1;

//		// statistic files
//		StreamWriter errorsFile = null;

		try
		{
//			// check if we need to save statistics to files
//			if ( saveStatisticsToFiles )
//			{
//				// open files
//				errorsFile	= File.CreateText( "errors.csv" );
//			}

			// erros list
			ArrayList errorsList = new ArrayList( );

			// loop
			while ( iteration <= 100 )
			{
				// run epoch of learning procedure
				double error = teacher.RunEpoch( input, output );
				errorsList.add( error );

				// save current error
//				if ( errorsFile != null )
//				{
//					errorsFile.WriteLine( error );
//				}

				// show current iteration & error
//				currentIterationBox.Text = iteration.ToString( );
//				currentErrorBox.Text = error.ToString( );
				iteration++;

				// check if we need to stop
				if ( error <= learningErrorLimit )
					break;
			}

			// show error's dynamics
			double[][] errors = new double[errorsList.size()][2];

			for ( int i = 0, n = errorsList.size(); i < n; i++ )
			{
				errors[i][0] = i;
				errors[i][1] = Double.valueOf(errorsList.get(i).toString());

				System.out.println(i);
				System.out.println(errors[i][1]);

			}

//			errorChart.RangeX = new DoubleRange( 0, errorsList.Count - 1 );
//			errorChart.UpdateDataSeries( "error", errors );
		}
//		catch()
//		{
//
//		}
		finally
		{

		}

	}








    public static void main(String[] args) {

//        NeuralNetwork nn = new NeuralNetwork(2, 4, 1);
//        int maxRuns = 50000;
//        double minErrorCondition = 0.001;
//        nn.run(maxRuns, minErrorCondition);

    	neural_network nn = new neural_network();
//    	int maxRuns = 50000;
//        double minErrorCondition = 0.001;


    }
}