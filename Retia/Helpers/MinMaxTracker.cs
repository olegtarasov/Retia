namespace Retia.Helpers
{
    /// <summary>
    /// Tracks mininmum and maximum values.
    /// </summary>
	public class MinMaxTracker
	{
        /// <summary>
        /// Minimum value.
        /// </summary>
		public double Min { get; private set; } = double.NaN;

        /// <summary>
        /// Maximum value.
        /// </summary>
		public double Max { get; private set; } = double.NaN;

        /// <summary>
        /// Min - Max.
        /// </summary>
	    public double Range { get; private set; } = double.NaN;

        /// <summary>
        /// Analyzes numbers and updates min and max values if neccessary.
        /// </summary>
        /// <param name="numbers">Numbers to analyze.</param>
		public void Track(params double[] numbers)
		{
			for (int i = 0; i < numbers.Length; i++)
			{
			    if (double.IsNaN(numbers[i]))
			    {
			        continue;
			    }

				if (double.IsNaN(Max) || numbers[i] > Max)
				{
					Max = numbers[i];
				}

				if (double.IsNaN(Min) || numbers[i] < Min)
				{
					Min = numbers[i];
				}

			    Range = Max - Min;
			}
		}

        /// <summary>
        /// Normalizes a value according to tracked minimum and maximum.
        /// </summary>
        /// <param name="value">Value to normalize.</param>
        public double Normalize(double value)
	    {
	        return (value - Min) / Range;
	    }

        /// <summary>
        /// Resets tracked values.
        /// </summary>
	    public void Reset()
	    {
	        Min = double.MaxValue;
	        Max = double.MinValue;
	        Range = double.NaN;
	    }
	}
}