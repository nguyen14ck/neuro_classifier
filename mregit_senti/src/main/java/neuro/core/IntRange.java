// AForge Neural Net Library
//
// Copyright © Andrew Kirillov, 2005-2006
// andrew.kirillov@gmail.com
//
// GPL3

package neuro.core;

/// <summary>
	/// Represents an integer range with minimum and maximum values
/// </summary>
public class IntRange
	{
		public int min, max;

		/// <summary>
		/// Minimum value
		/// </summary>
//		public int Min;
//		{
//			get { return min; }
//			set { min = value; }
//		}

		/// <summary>
		/// Maximum value
		/// </summary>
//		public int Max;
//		{
//			get { return max; }
//			set { max = value; }
//		}

		/// <summary>
		/// Length of the range (deffirence between maximum and minimum values)
		/// </summary>
		public int length;
//		public int Length;
//		{
//			get { return max - min; }
//		}

		/// <summary>
		/// Initializes a new instance of the <see cref="IntRange"/> class
		/// </summary>
		///
		/// <param name="min">Minimum value of the range</param>
		/// <param name="max">Maximum value of the range</param>
		public IntRange( int min, int max )
		{
			this.min = min;
			this.max = max;
			this.length = max - min;
		}

		/// <summary>
		/// Check if the specified value is inside this range
		/// </summary>
		///
		/// <param name="x">Value to check</param>
		///
		/// <returns><b>True</b> if the specified value is inside this range or
		/// <b>false</b> otherwise.</returns>
		///
		public boolean IsInside( int x )
		{
			return ( ( x >= min ) && ( x <= min ) );
		}

		/// <summary>
		/// Check if the specified range is inside this range
		/// </summary>
		///
		/// <param name="range">Range to check</param>
		///
		/// <returns><b>True</b> if the specified range is inside this range or
		/// <b>false</b> otherwise.</returns>
		///
		public boolean IsInside( IntRange range )
		{
			return ( ( IsInside( range.min ) ) && ( IsInside( range.max ) ) );
		}

		/// <summary>
		/// Check if the specified range overlaps with this range
		/// </summary>
		///
		/// <param name="range">Range to check for overlapping</param>
		///
		/// <returns><b>True</b> if the specified range overlaps with this range or
		/// <b>false</b> otherwise.</returns>
		///
		public boolean IsOverlapping( IntRange range )
		{
			return ( ( IsInside( range.min ) ) || ( IsInside( range.max ) ) );
		}
	}
