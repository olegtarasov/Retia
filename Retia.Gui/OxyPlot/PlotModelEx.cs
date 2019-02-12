using System;
using System.Collections.Generic;
using System.Linq;
using OxyPlot;
using OxyPlot.Axes;
using OxyPlot.Series;
using Retia.Helpers;
using Retia.Integration.Helpers;

namespace Retia.Gui.OxyPlot
{
    public class PlotModelEx : PlotModel
    {
        private readonly List<List<SeriesWrapperBase>> _sections = new List<List<SeriesWrapperBase>>();
        private readonly MinMaxTracker _xTracker = new MinMaxTracker();
        
        private Axis _xAxis;
        private bool _autoZoomY;
        
        public SeriesWrapperBase this[int section, int series]
        {
            get
            {
                if (section >= _sections.Count)
                {
                    throw new ArgumentOutOfRangeException(nameof(section));
                }
                if (series >= _sections[section].Count)
                {
                    throw new ArgumentOutOfRangeException(nameof(series));
                }

                return _sections[section][series];
            }
            set
            {
                while (section >= _sections.Count)
                {
                    _sections.Add(new List<SeriesWrapperBase>());
                }

                var list = _sections[section];
                while (series >= list.Count)
                {
                    list.Add(null);
                }

                list[series] = value;

                RebuildSeries();
            }
        }

        public Func<Axis> YAxisFactory { get; set; } = () => new LinearAxis
                                                                   {
                                                                       Position = AxisPosition.Left,
                                                                       IsZoomEnabled = true,
                                                                       IsPanEnabled = true,
                                                                       MajorGridlineStyle = LineStyle.Solid,
                                                                       MinorGridlineStyle = LineStyle.Solid,
                                                                       MajorGridlineColor = OxyColor.FromAColor(40, OxyColors.DarkBlue),
                                                                       MinorGridlineColor = OxyColor.FromAColor(20, OxyColors.DarkBlue),
                                                                       AxislineStyle = LineStyle.Solid,
                                                                   };

        public bool AutoZoomY
        {
            get
            {
                return _autoZoomY;
            }
            set
            {
                if (_autoZoomY == value)
                {
                    return;
                }

                _autoZoomY = value;

                if (XAxis != null)
                {
                    if (value)
                    {
                        XAxis.AxisChanged += XAxisOnAxisChanged;
                    }
                    else
                    {
                        XAxis.AxisChanged -= XAxisOnAxisChanged;
                    }
                }
            }
        }

        public Axis XAxis
        {
            get
            {
                return _xAxis;
            }
            set
            {
                if (_xAxis != null && _autoZoomY)
                {
                    _xAxis.AxisChanged -= XAxisOnAxisChanged;
                }

                _xAxis = value;

                if (_autoZoomY)
                {
                    _xAxis.AxisChanged += XAxisOnAxisChanged;
                }
            }
        }

        private IEnumerable<SeriesWrapperBase> FlatSeries => _sections.SelectMany(x => x);

        public void ZoomAtBeginning(double percent)
        {
            CheckXAxis();

            XAxis.Zoom(_xTracker.Min, _xTracker.Min + ((_xTracker.Max - _xTracker.Min) * percent));
        }

        public void DateTimeXAxis()
        {
            if (XAxis != null)
            {
                Axes.Remove(XAxis);
            }

            XAxis = new DateTimeAxis
                     {
                         Position = AxisPosition.Bottom,
                         IsZoomEnabled = true,
                         IsPanEnabled = true,
                         MajorGridlineStyle = LineStyle.Solid,
                         MinorGridlineStyle = LineStyle.Solid,
                         MajorGridlineColor = OxyColor.FromAColor(40, OxyColors.DarkBlue),
                         MinorGridlineColor = OxyColor.FromAColor(20, OxyColors.DarkBlue),
                         AxislineStyle = LineStyle.Solid
                     };
            Axes.Add(XAxis);
        }

        public void TimeSpanAxis()
        {
            if (XAxis != null)
            {
                Axes.Remove(XAxis);
            }

            XAxis = new TimeSpanAxis
            {
                Position = AxisPosition.Bottom,
                IsZoomEnabled = true,
                IsPanEnabled = true,
                MajorGridlineStyle = LineStyle.Solid,
                MinorGridlineStyle = LineStyle.Solid,
                MajorGridlineColor = OxyColor.FromAColor(40, OxyColors.DarkBlue),
                MinorGridlineColor = OxyColor.FromAColor(20, OxyColors.DarkBlue),
                AxislineStyle = LineStyle.Solid
            };
            Axes.Add(XAxis);
        }

        public void LinearXAxis()
        {
            if (XAxis != null)
            {
                Axes.Remove(XAxis);
            }

            XAxis = new LinearAxis
            {
                Position = AxisPosition.Bottom,
                IsZoomEnabled = true,
                IsPanEnabled = true,
                MajorGridlineStyle = LineStyle.Solid,
                MinorGridlineStyle = LineStyle.Solid,
                MajorGridlineColor = OxyColor.FromAColor(40, OxyColors.DarkBlue),
                MinorGridlineColor = OxyColor.FromAColor(20, OxyColors.DarkBlue),
                AxislineStyle = LineStyle.Solid
            };
            Axes.Add(XAxis);
        }

        public void ZoomY()
        {
            if (XAxis == null)
            {
                return;
            }

            foreach (var group in _sections.SelectMany(x => x).Where(x => x != null).GroupBy(x => x.Series.YAxisKey))
            {
                var axisMinMax = new MinMaxTracker();

                foreach (var wrapper in group)
                {
                    var minMax = wrapper.GetMinMaxY(XAxis.ActualMinimum, XAxis.ActualMaximum);
                    if (!double.IsNaN(minMax.Min) && !double.IsNaN(minMax.Max))
                    {
                        axisMinMax.Track(minMax.Min, minMax.Max);
                    }
                }

                if (!double.IsNaN(axisMinMax.Min) && !double.IsNaN(axisMinMax.Max))
                {
                    double delta = Math.Abs(axisMinMax.Max - axisMinMax.Min) * 0.02;
                    Axes.First(x => x.Key == group.Key).Zoom(axisMinMax.Min - delta, axisMinMax.Max + delta);
                }
            }
        }

        public void ZoomX()
        {
            if (XAxis == null)
            {
                return;
            }

            var tracker = new MinMaxTracker();
            foreach (var series in FlatSeries)
            {
                var minMax = series.GetMinMaxX();
                if (minMax != null)
                {
                    tracker.Track(minMax.Value.Min, minMax.Value.Max);
                }
            }

            double delta = Math.Abs(tracker.Max - tracker.Min) * 0.02;
            XAxis.Zoom(tracker.Min - delta, tracker.Max + delta);
        }

        private void XAxisOnAxisChanged(object sender, AxisChangedEventArgs axisChangedEventArgs)
        {
            ZoomY();
        }

        private void CheckXAxis()
        {
            if (XAxis == null)
            {
                throw new InvalidOperationException("X axis is not defined!");
            }
        }

        private void RebuildSeries()
        {
            _xTracker.Reset();
            Series.Clear();
            foreach (var axis in Axes.Where(x => x.Position == AxisPosition.Left).ToList())
            {
                Axes.Remove(axis);
            }

            var trueSections = _sections.Where(x => x.Any(s => s != null)).ToList();
            double sectionHeight = 1.0d / trueSections.Count;

            for (int i = 0; i < trueSections.Count; i++)
            {
                string axisKey = Guid.NewGuid().ToString();
                var axis = YAxisFactory();
                axis.StartPosition = sectionHeight * i;
                axis.EndPosition = sectionHeight * i + sectionHeight * (i == trueSections.Count - 1 ? 1.0 : 0.9);
                axis.Key = axisKey;

                Axes.Add(axis);

                foreach (var series in trueSections[i].Where(x => x != null))
                {
                    series.Series.YAxisKey = axisKey;
                    Series.Add(series.Series);
                }
            }

            ((IPlotModel)this).Update(true);
            foreach (var series in Series.OfType<XYAxisSeries>())
            {
                _xTracker.Track(series.MinX);
				_xTracker.Track(series.MaxX);
            }
        }
    }
}