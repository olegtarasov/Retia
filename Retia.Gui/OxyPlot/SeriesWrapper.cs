using System;
using System.Collections.Generic;
using System.Linq;
using OxyPlot.Series;
using Retia.Helpers;

namespace Retia.Gui.OxyPlot
{
    public abstract class SeriesWrapperBase
    {
        public XYAxisSeries Series { get; protected set; }

        public abstract MinMaxTracker GetMinMaxY(double startX, double endX);
        public abstract MinMax<double>? GetMinMaxX();
    }

    public class SeriesWrapper<TSeries, TSeriesPoint, TSourcePoint> : SeriesWrapperBase where TSeries : XYAxisSeries
    {
        private readonly List<TSeriesPoint> _points;
        private readonly Func<TSeriesPoint, PointWrapper> _pointWrapperFunc;
        private readonly Func<TSourcePoint, TSeriesPoint> _sourceToSeriesFunc;

        private int _curIdx = 0;

        public SeriesWrapper(TSeries series, List<TSeriesPoint> points, Func<TSeriesPoint, PointWrapper> pointWrapperFunc, Func<TSourcePoint, TSeriesPoint> sourceToSeriesFunc)
        {
            if (series == null) throw new ArgumentNullException(nameof(series));
            if (points == null) throw new ArgumentNullException(nameof(points));
            if (pointWrapperFunc == null) throw new ArgumentNullException(nameof(pointWrapperFunc));
            
            Series = series;
            _points = points;
            _pointWrapperFunc = pointWrapperFunc;
            _sourceToSeriesFunc = sourceToSeriesFunc;
        }

        private PointWrapper CurrentPoint => _pointWrapperFunc(_points[_curIdx]);

        public IReadOnlyList<TSeriesPoint> SeriesPoints => _points;
        public new TSeries Series { get { return (TSeries)base.Series; } private set { base.Series = value; } }

        public override MinMaxTracker GetMinMaxY(double startX, double endX)
        {
            var result = new MinMaxTracker();
            if (_points.Count == 0)
            {
                return result;
            }

            GotoX(startX);

            PointWrapper point;
            int idx = _curIdx;

            while (idx < _points.Count && (point = _pointWrapperFunc(_points[idx])).X < endX)
            {
                result.Track(point.MaxY, point.MinY);
                idx++;
            }

            return result;
        }

        public override MinMax<double>? GetMinMaxX()
        {
            if (_points.Count == 0)
            {
                return null;
            }

            return new MinMax<double> {Min = _pointWrapperFunc(_points[0]).X, Max = _pointWrapperFunc(_points[_points.Count - 1]).X};
        }

        public void AddPoint(TSourcePoint point)
        {
            _points.Add(_sourceToSeriesFunc(point));
        }

        public void AddPoints(IEnumerable<TSourcePoint> points)
        {
            _points.AddRange(points.Select(x => _sourceToSeriesFunc(x)));
        }

        public void ClearPoints()
        {
            _points.Clear();
        }

        private void GotoX(double x)
        {
            if (CurrentPoint.X < x)
            {
                while (_curIdx < _points.Count - 1 && CurrentPoint.X < x)
                {
                    _curIdx++;
                }
            }
            else if (CurrentPoint.X > x)
            {
                while (_curIdx > 0 && CurrentPoint.X > x)
                {
                    _curIdx--;
                }
            }
        }
    }

    public struct PointWrapper
    {
        public PointWrapper(double x, double maxY, double minY)
        {
            X = x;
            MaxY = maxY;
            MinY = minY;
        }

        public double X;
        public double MaxY;
        public double MinY;
    }
}