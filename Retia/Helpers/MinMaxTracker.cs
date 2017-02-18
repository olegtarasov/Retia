namespace Retia.Helpers
{
	public class MinMaxTracker
	{
		public double Min { get; private set; } = double.NaN;
		public double Max { get; private set; } = double.NaN;

	    public double Range { get; private set; } = double.NaN;

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

	    public double Normalize(double value)
	    {
	        return (value - Min) / Range;
	    }

	    public void Reset()
	    {
	        Min = double.MaxValue;
	        Max = double.MinValue;
	        Range = double.NaN;
	    }
	}
}