
// AForge Neural Net Library
//
// Copyright © Andrew Kirillov, 2005-2006
// andrew.kirillov@gmail.com
//

package neuro.core;


public class ActivationLayer extends Layer
	{
		/// <summary>
		/// Layer's neurons accessor
		/// </summary>
		///
		/// <param name="index">Neuron index</param>
		///
		/// <remarks>Allows to access layer's neurons.</remarks>
		///
//		public new ActivationNeuron this[int index]
//		{
//			get { return (ActivationNeuron) neurons[index]; }
//		}
		public ActivationNeuron getActivationNeuron (int index)
		{
			return (ActivationNeuron) neurons[index];
		}


		/// <summary>
		/// Initializes a new instance of the <see cref="ActivationLayer"/> class
		/// </summary>
		/// <param name="neuronsCount">Layer's neurons count</param>
		/// <param name="inputsCount">Layer's inputs count</param>
		/// <param name="function">Activation function of neurons of the layer</param>
		///
		/// <remarks>The new layet will be randomized (see <see cref="ActivationNeuron.Randomize"/>
		/// method) after it is created.</remarks>
		///
		public ActivationLayer( int neuronsCount, int inputsCount, IActivationFunction function )
		{
			super( neuronsCount, inputsCount );
			
			// create each neuron
			for ( int i = 0; i < neuronsCount; i++ )
				neurons[i] = new ActivationNeuron( inputsCount, function );
		}
	}
