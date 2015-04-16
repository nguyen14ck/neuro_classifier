package neuro.core;

import java.text.*;
import java.util.*;


/// <summary>
	/// Activation neuron
	/// </summary>
	///
	/// <remarks>Activation neuron computes weighted sum of its inputs, adds
	/// bias value and then applies activation function. The neuron is
	/// usually used in multi-layer neural networks.</remarks>
	///

public class ActivationNeuron extends Neuron
	{
		/// <summary>
		/// Bias value
		/// </summary>
		///
		/// <remarks>The value is added to inputs weighted sum.</remarks>
		///
		//public double bias = 0.0f;

		/// <summary>
		/// Activation function
		/// </summary>
		///
		/// <remarks>The function is applied to inputs weighted sum plus
		/// bias value.</remarks>
		///
		public IActivationFunction function;

		/// <summary>
		/// Bias value
		/// </summary>
		///
		/// <remarks>The value is added to inputs weighted sum.</remarks>
		///
		public double Bias;
//		{
//			get { return bias; }
//			set { bias = value; }
//		}

		/// <summary>
		/// Neuron's activation function
		/// </summary>
		///
		public IActivationFunction ActivationFunction;
//		{
//			get { return function; }
//		}

		/// <summary>
		/// Initializes a new instance of the <see cref="ActivationNeuron"/> class
		/// </summary>
		///
		/// <param name="inputs">Neuron's inputs count</param>
		/// <param name="function">Neuron's activation function</param>
		///
		public ActivationNeuron( int inputs, IActivationFunction function )
		{
			super( inputs );
			this.function = function;
		}

		/// <summary>
		/// Randomize neuron
		/// </summary>
		///
		/// <remarks>Calls base class <see cref="Neuron.Randomize">Randomize</see> method
		/// to randomize neuron's weights and then randomize bias's value.</remarks>
		///
		public void Randomize( )
		{
			// randomize weights
			super.Randomize( );
			// randomize bias
			bias = rand.nextDouble( ) * ( randRange.length ) + randRange.min;
		}

		/// <summary>
		/// Computes output value of neuron
		/// </summary>
		///
		/// <param name="input">Input vector</param>
		///
		/// <returns>Returns neuron's output value</returns>
		///
		/// <remarks>The output value of activation neuron is equal to value
		/// of nueron's activation function, which parameter is weighted sum
		/// of its inputs plus bias value. The output value is also stored
		/// in <see cref="Neuron.Output">Output</see> property.</remarks>
		///
		public double Compute( double[] input )
		{
			// check for corrent input vector
			if ( input.length != inputsCount )
				throw new IllegalArgumentException("Incorrect length of input vector");

			// initial sum value
			double sum = 0.0;

			// compute weighted sum of inputs
			for ( int i = 0; i < inputsCount; i++ )
			{
				sum += weights[i] * input[i];
			}
			sum += bias;

			return ( output = function.Function( sum ) );
		}
	}