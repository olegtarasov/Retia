using System;
using System.Collections.Generic;
using OxyPlot;
using OxyPlot.Series;

namespace Retia.Gui.OxyPlot
{
    public static class SeriesBuilder
    {
        public static SeriesWrapper<CandleStickSeries, HighLowItem, T> CandleStick<T>(IEnumerable<T> items, Func<T, HighLowItem> itemMapper)
        {
            var series = new CandleStickSeries();

            foreach (var item in items)
            {
                series.Items.Add(itemMapper(item));
            }

            return new SeriesWrapper<CandleStickSeries, HighLowItem, T>(series, series.Items, item => new PointWrapper(item.X, item.High, item.Low), itemMapper);
        }

        public static LineSeriesWrapper<T> LineSeries<T>(IEnumerable<T> items, Func<T, DataPoint> itemMapper)
        {
            var series = new LineSeries
                   {
                       StrokeThickness = 1,
                       CanTrackerInterpolatePoints = false
                   };

            foreach (var item in items)
            {
                series.Points.Add(itemMapper(item));
            }

            return new LineSeriesWrapper<T>(series, series.Points, point => new PointWrapper(point.X, point.Y, point.Y), itemMapper);
        }
    }

    public class LineSeriesWrapper<T> : SeriesWrapper<LineSeries, DataPoint, T>
    {
        public LineSeriesWrapper(LineSeries series, List<DataPoint> points, Func<DataPoint, PointWrapper> pointWrapperFunc, Func<T, DataPoint> sourceToSeriesFunc) : base(series, points, pointWrapperFunc, sourceToSeriesFunc)
        {
        }
    }
}