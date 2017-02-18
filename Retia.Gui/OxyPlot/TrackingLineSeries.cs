using OxyPlot;
using OxyPlot.Series;

namespace Retia.Gui.OxyPlot
{
	public class TrackingLineSeries : LineSeries
	{
		public double MininimumY { get; private set; } = double.MaxValue;
		public double MaximumY { get; private set; } = double.MinValue;

		public void AddPoint(double x, double y)
		{
			if (y < MininimumY)
			{
				MininimumY = y;
			}

			if (y > MaximumY)
			{
				MaximumY = y;
			}

			Points.Add(new DataPoint(x, y));
		}
	}
}