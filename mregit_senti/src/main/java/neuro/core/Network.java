package neuro.core;

public abstract class Network
	{
		/// <summary>
		/// Network's inputs count
		/// </summary>
		public int	inputsCount;

		/// <summary>
		/// Network's layers count
		/// </summary>
		public int	layersCount;

		/// <summary>
		/// Network's layers
		/// </summary>
		public Layer[]	layers;

		/// <summary>
		/// Network's output vector
		/// </summary>
		public double[]	output;
		
		public double [][][] network_weigths;
		public double [][][] network_biases;
//		public double layer_thesholds;

		/// <summary>
		/// Network's inputs count
		/// </summary>
//		public int InputsCount;
//		{
//			get { return inputsCount; }
//		}

		/// <summary>
		/// Network's layers count
		/// </summary>
//		public int LayersCount;
//		{
//			get { return layersCount; }
//		}

		/// <summary>
		/// Network's output vector
		/// </summary>
		///
		/// <remarks>The calculation way of network's output vector is determined by
		/// inherited class.</remarks>
		///
//		public double[] Output;
//		{
//			get { return output; }
//		}

		/// <summary>
		/// Network's layers accessor
		/// </summary>
		///
		/// <param name="index">Layer index</param>
		///
		/// <remarks>Allows to access network's layer.</remarks>
		///
//		public Layer this[int index]
//		{
//			get { return layers[index]; }
//		}


		/// <summary>
		/// Initializes a new instance of the <see cref="Network"/> class
		/// </summary>
		///
		/// <param name="inputsCount">Network's inputs count</param>
		/// <param name="layersCount">Network's layers count</param>
		///
		/// <remarks>Protected constructor, which initializes <see cref="inputsCount"/>,
		/// <see cref="layersCount"/> and <see cref="layers"/> members.</remarks>
		///
		public Network( int inputsCount, int layersCount ) //for trainning
		{
			this.inputsCount = Math.max( 1, inputsCount );
			this.layersCount = Math.max( 1, layersCount );
			// create collection of layers
			layers = new Layer[this.layersCount];
		}
		 
		public Network( int inputsCount, int layersCount, double [][][] nww, double [][][] nwb ) //for testing
		{
			this.inputsCount = Math.max( 1, inputsCount );
			this.layersCount = Math.max( 1, layersCount );
			this.network_weigths = nww;
			this.network_biases = nwb;
			
			// create collection of layers
			layers = new Layer[this.layersCount];
		}
		/// <summary>
		/// Compute output vector of the network
		/// </summary>
		///
		/// <param name="input">Input vector</param>
		///
		/// <returns>Returns network's output vector</returns>
		///
		/// <remarks>The actual network's output vecor is determined by inherited class and it
		/// represents an output vector of the last layer of the network. The output vector is
		/// also stored in <see cref="Output"/> property.</remarks>
		///
		public double[] Compute( double[] input )
		{
			output = input;

			// compute each layer
			for ( Layer layer : layers )
			{
				output = layer.Compute( output );
			}

			return output;
		}

		/// <summary>
		/// Randomize layers of the network
		/// </summary>
		///
		/// <remarks>Randomizes network's layers by calling <see cref="Layer.Randomize"/> method
		/// of each layer.</remarks>
		///
		public void Randomize( )
		{
			for ( Layer layer : layers )
			{
				layer.Randomize();
			}
		}
	}