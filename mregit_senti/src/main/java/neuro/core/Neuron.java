// AForge Neural Net Library
//
// Copyright © Andrew Kirillov, 2005-2006
// andrew.kirillov@gmail.com
//
// GPL3


package neuro.core;

import java.text.*;
import java.util.*;


public abstract class Neuron
	{

		/// <summary>
		/// Neuron's inputs count
		/// </summary>
		public int		inputsCount = 0;

		/// <summary>
		/// Nouron's wieghts
		/// </summary>
		public double[]	weights = null;
		public double bias = 0.0f;

		/// <summary>
		/// Neuron's output value
		/// </summary>
		public double	output = 0;

		/// <summary>
		/// Random number generator
		/// </summary>
		///
		/// <remarks>The generator is used for neuron's weights randomization</remarks>
		///
		public static Random	rand = new Random( (int) System.currentTimeMillis() );

		/// <summary>
		/// Random generator range
		/// </summary>
		///
		/// <remarks>Sets the range of random generator. Affects initial values of neuron's weight.
		/// Default value is [0, 1].</remarks>
		///
		public static DoubleRange randRange = new DoubleRange( 0.0, 0.25 );

		/// <summary>
		/// Random number generator
		/// </summary>
		///
		/// <remarks>The property allows to initialize random generator with a custom seed. The generator is
		/// used for neuron's weights randomization.</remarks>
		///
//		public static Random RandGenerator;
//		{
//			get { return rand; }
//			set
//			{
//				if ( value != null )
//				{
//					rand = value;
//				}
//			}
//		}

		/// <summary>
		/// Random generator range
		/// </summary>
//		public static DoubleRange RandRange;
//		{
//			get { return randRange; }
//			set
//			{
//				if ( value != null )
//				{
//					randRange = value;
//				}
//			}
//		}

		/// <summary>
		/// Neuron's inputs count
		/// </summary>
//		public int InputsCount;
//		{
//			get { return inputsCount; }
//		}

		/// <summary>
		/// Neuron's output value
		/// </summary>
		///
		/// <remarks>The calculation way of neuron's output value is determined by inherited class.</remarks>
		///
//		public double Output;
//		{
//			get { return output; }
//		}

		/// <summary>
		/// Neuron's weights accessor
		/// </summary>
		///
		/// <param name="index">Weight index</param>
		///
		/// <remarks>Allows to access neuron's weights.</remarks>
		///
//		public double this[int index];
//		{
//			get { return weights[index]; }
//			set { weights[index] = value; }
//		}

		/// <summary>
		/// Initializes a new instance of the <see cref="Neuron"/> class
		/// </summary>
		///
		/// <param name="inputs">Neuron's inputs count</param>
		///
		/// <remarks>The new neuron will be randomized (see <see cref="Randomize"/> method)
		/// after it is created.</remarks>
		///
		public Neuron( int inputs )
		{
			// allocate weights
			inputsCount = Math.max( 1, inputs );
			weights = new double[inputsCount];
			// randomize the neuron
			Randomize( );
		}

		/// <summary>
		/// Randomize neuron
		/// </summary>
		///
		/// <remarks>Initialize neuron's weights with random values within the range specified
		/// by <see cref="RandRange"/>.</remarks>
		///
		public void Randomize( ) //abstract
		{
			double d = randRange.length;

			// randomize weights
			for ( int i = 0; i < inputsCount; i++ )
				weights[i] = rand.nextDouble( ) * d + randRange.min;
		}

		/// <summary>
		/// Computes output value of neuron
		/// </summary>
		///
		/// <param name="input">Input vector</param>
		///
		/// <returns>Returns neuron's output value</returns>
		///
		/// <remarks>The actual neuron's output value is determined by inherited class.
		/// The output value is also stored in <see cref="Output"/> property.</remarks>
		///
		public abstract double Compute( double[] input );
	}
