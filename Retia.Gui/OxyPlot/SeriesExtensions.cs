using System;
using OxyPlot.Axes;

namespace Retia.Gui.OxyPlot
{
	public static class SeriesExtensions
	{
		public static double ToDouble(this DateTime time)
		{
			return DateTimeAxis.ToDouble(time.ToLocalTime());
		}
	}
}