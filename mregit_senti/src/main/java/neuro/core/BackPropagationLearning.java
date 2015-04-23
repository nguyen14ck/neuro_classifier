// AForge Neural Net Library
//
// Copyright © Andrew Kirillov, 2005-2006
// andrew.kirillov@gmail.com
//

package neuro.core;

import neuro.core.ActivationNetwork;

/// <summary>
	/// Back propagation learning algorithm
	/// </summary>
	///
	/// <remarks>The class implements back propagation learning algorithm,
	/// which is widely used for training multi-layer neural networks with
	/// continuous activation functions.</remarks>
	///

public class BackPropagationLearning implements ISupervisedLearning
	{
		// network to teach
		private ActivationNetwork network;
		// learning rate
		private double learningRate = 0.1;
		// momentum
		private double momentum = 0.0;

		// neuron's errors
		private double[][]		neuronErrors = null;
		// weight's updates
		private double[][][]	weightsUpdates = null;
		// bias's updates
		private double[][]		biassUpdates = null;

		

		/// <summary>
		/// Learning rate
		/// </summary>
		///
		/// <remarks>The value determines speed of learning. Default value equals to 0.1.</remarks>
		///
		public double LearningRate;
//		{
//			get { return learningRate; }
//			set
//			{
//				learningRate = Math.Max( 0.0, Math.Min( 1.0, value ) );
//			}
//		}

		/// <summary>
		/// Momentum
		/// </summary>
		///
		/// <remarks>The value determines the portion of previous weight's update
		/// to use on current iteration. Weight's update values are calculated on
		/// each iteration depending on neuron's error. The momentum specifies the amount
		/// of update to use from previous iteration and the amount of update
		/// to use from current iteration. If the value is equal to 0.1, for example,
		/// then 0.1 portion of previous update and 0.9 portion of current update are used
		/// to update weight's value.<br /><br />
		///	Default value equals to 0.0.</remarks>
		///
		public double Momentum;
//		{
//			get { return momentum; }
//			set
//			{
//				momentum = Math.Max( 0.0, Math.Min( 1.0, value ) );
//			}
//		}

		/// <summary>
		/// Initializes a new instance of the <see cref="BackPropagationLearning"/> class
		/// </summary>
		///
		/// <param name="network">Network to teach</param>
		///
		public BackPropagationLearning( ActivationNetwork network, double lr, double mn)
		{
			this.network = network;
			this.learningRate = lr;
			this.momentum = mn;
			
			

			// create error and deltas arrays
			neuronErrors = new double[network.layersCount][];
			weightsUpdates = new double[network.layersCount][][];
			biassUpdates = new double[network.layersCount][];

			// initialize errors and deltas arrays for each layer
			for ( int i = 0, n = network.layersCount; i < n; i++ )
			{
				Layer layer = network.layers[i];

				neuronErrors[i] = new double[layer.neuronsCount];
				weightsUpdates[i] = new double[layer.neuronsCount][];
				biassUpdates[i] = new double[layer.neuronsCount];

				// for each neuron
				for ( int j = 0; j < layer.neuronsCount; j++ )
				{
					weightsUpdates[i][j] = new double[layer.inputsCount];
				}
			}
		}



		/// <summary>
		/// Runs learning iteration
		/// </summary>
		///
		/// <param name="input">input vector</param>
		/// <param name="output">desired output vector</param>
		///
		/// <returns>Returns squared error of the last layer divided by 2</returns>
		///
		/// <remarks>Runs one learning iteration and updates neuron's
		/// weights.</remarks>
		///
		public double Run( double[] input, double[] output )
		{
			// compute the network's output
			network.Compute( input );

			// calculate network error
			double error = CalculateError( output );

			// calculate weights updates
			CalculateUpdates( input );

			// update the network
			UpdateNetwork( );

			return error;
		}

		/// <summary>
		/// Runs learning epoch
		/// </summary>
		///
		/// <param name="input">array of input vectors</param>
		/// <param name="output">array of output vectors</param>
		///
		/// <returns>Returns sum of squared errors of the last layer divided by 2</returns>
		///
		/// <remarks>Runs series of learning iterations - one iteration
		/// for each input sample. Updates neuron's weights after each sample
		/// presented.</remarks>
		///
		public double RunEpoch( double[][] input, double[][] output )
		{
			double error = 0.0;

			// run learning procedure for all samples
			for ( int i = 0, n = input.length; i < n; i++ )
			{
				error += Run( input[i], output[i] );
			}

			// return summary error
			return error;
		}


		/// <summary>
		/// Calculates error values for all neurons of the network
		/// </summary>
		///
		/// <param name="desiredOutput">Desired output vector</param>
		///
		/// <returns>Returns summary squared error of the last layer divided by 2</returns>
		///
		private double CalculateError( double[] desiredOutput )
		{
			// current and the next layers
			ActivationLayer	layer, layerNext;
			// current and the next errors arrays
			double[] errors, errorsNext;
			// error values
			double error = 0, e, sum;
			// neuron's output value
			double output;
			// layers count
			int layersCount = network.layersCount;

			// assume, that all neurons of the network have the same activation function
			IActivationFunction	function = network.getActivationLayer(0).getActivationNeuron(0).function;

			// calculate error values for the last layer first
			layer	= network.getActivationLayer(layersCount - 1);
			errors	= neuronErrors[layersCount - 1];

			for ( int i = 0, n = layer.neuronsCount; i < n; i++ )
			{
				output = layer.neurons[i].output;
				// error of the neuron
				e = desiredOutput[i] - output;
				// error multiplied with activation function's derivative
				errors[i] = e * function.Derivative2( output );
				// square the error and sum it
				error += ( e * e );
			}

			// calculate error values for other layers
			for ( int j = layersCount - 2; j >= 0; j-- )
			{
				layer		= network.getActivationLayer(j);
				layerNext	= network.getActivationLayer(j + 1);
				errors		= neuronErrors[j];
				errorsNext	= neuronErrors[j + 1];

				// for all neurons of the layer
				for ( int i = 0, n = layer.neuronsCount; i < n; i++ )
				{
					sum = 0.0;
					// for all neurons of the next layer
					for ( int k = 0, m = layerNext.neuronsCount; k < m; k++ )
					{
						sum += errorsNext[k] * layerNext.neurons[k].weights[i];
					}
					errors[i] = sum * function.Derivative2( layer.neurons[i].output );
				}
			}

			// return squared error of the last layer divided by 2
			return error / 2.0;
		}

		/// <summary>
		/// Calculate weights updates
		/// </summary>
		///
		/// <param name="input">Network's input vector</param>
		///
		private void CalculateUpdates( double[] input )
		{
			// current neuron
			ActivationNeuron	neuron;
			// current and previous layers
			ActivationLayer		layer, layerPrev;
			// layer's weights updates
			double[][]	layerWeightsUpdates;
			// layer's biass updates
			double[]	layerBiasUpdates;
			// layer's error
			double[]	errors;
			// neuron's weights updates
			double[]	neuronWeightUpdates;
			// error value
			double		error;

			// 1 - calculate updates for the last layer fisrt
			layer = network.getActivationLayer(0);
			errors = neuronErrors[0];
			layerWeightsUpdates = weightsUpdates[0];
			layerBiasUpdates = biassUpdates[0];

			// for each neuron of the layer
			for ( int i = 0, n = layer.neuronsCount; i < n; i++ )
			{
				neuron	= layer.getActivationNeuron(i);
				error	= errors[i];
				neuronWeightUpdates	= layerWeightsUpdates[i];

				// for each weight of the neuron
				for ( int j = 0, m = neuron.inputsCount; j < m; j++ )
				{
					// calculate weight update
					neuronWeightUpdates[j] = learningRate * (
						momentum * neuronWeightUpdates[j] +
						( 1.0 - momentum ) * error * input[j]
						);
				}

				// calculate bias update
				layerBiasUpdates[i] = learningRate * (
					momentum * layerBiasUpdates[i] +
					( 1.0 - momentum ) * error
					);
			}

			// 2 - for all other layers
			for ( int k = 1, l = network.layersCount; k < l; k++ )
			{
				layerPrev			= network.getActivationLayer(k-1);
				layer				= network.getActivationLayer(k);
				errors				= neuronErrors[k];
				layerWeightsUpdates	= weightsUpdates[k];
				layerBiasUpdates = biassUpdates[k];

				// for each neuron of the layer
				for ( int i = 0, n = layer.neuronsCount; i < n; i++ )
				{
					neuron	= layer.getActivationNeuron(i);
					error	= errors[i];
					neuronWeightUpdates	= layerWeightsUpdates[i];

					// for each synapse of the neuron
					for ( int j = 0, m = neuron.inputsCount; j < m; j++ )
					{
						// calculate weight update
						neuronWeightUpdates[j] = learningRate * (
							momentum * neuronWeightUpdates[j] +
							( 1.0 - momentum ) * error * layerPrev.neurons[j].output
							);
					}

					// calculate treshold update
					layerBiasUpdates[i] = learningRate * (
						momentum * layerBiasUpdates[i] +
						( 1.0 - momentum ) * error
						);
				}
			}
		}

		/// <summary>
		/// Update network'sweights
		/// </summary>
		///
		private void UpdateNetwork( )
		{
			// current neuron
			ActivationNeuron	neuron;
			// current layer
			ActivationLayer		layer;
			// layer's weights updates
			double[][]	layerWeightsUpdates;
			// layer's biass updates
			double[]	layerBiasUpdates;
			// neuron's weights updates
			double[]	neuronWeightUpdates;

			// for each layer of the network
			for ( int i = 0, n = network.layersCount; i < n; i++ )
			{
				layer = network.getActivationLayer(i);
				layerWeightsUpdates = weightsUpdates[i];
				layerBiasUpdates = biassUpdates[i];

				// for each neuron of the layer
				for ( int j = 0, m = layer.neuronsCount; j < m; j++ )
				{
					neuron = layer.getActivationNeuron(j);
					neuronWeightUpdates = layerWeightsUpdates[j];

					// for each weight of the neuron
					for ( int k = 0, s = neuron.inputsCount; k < s; k++ )
					{
						// update weight of neuron (aggregate)
						neuron.weights[k] += neuronWeightUpdates[k];
						
						// update weights to network storage
						//this.network.network_weigths[j][k] = neuron.weights[k];
					}
					// update bias
					neuron.bias += layerBiasUpdates[j];
					
					// need to update bias storage
				}
			}
		}
	}
