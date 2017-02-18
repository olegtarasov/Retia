using System;
using System.Linq;
using OxyPlot;
using PropertyChanged;
using Retia.Gui.Messages;
using Retia.Gui.OxyPlot;
using Retia.Training.Trainers;

namespace Retia.Gui.Models
{
    [ImplementPropertyChanged]
    public class TrainingReportModel
    {
        private readonly LineSeriesWrapper<DataPoint> _errorSeries;
        private readonly LineSeriesWrapper<DataPoint> _testErrorSeries;

        public TrainingReportModel()
        {
            var model = new PlotModelEx { LegendPosition = LegendPosition.TopRight};
            model.LinearXAxis();

            _errorSeries = SeriesBuilder.LineSeries(Enumerable.Empty<DataPoint>(), x => x);
            _errorSeries.Series.Title = "Training error";

            _testErrorSeries = SeriesBuilder.LineSeries(Enumerable.Empty<DataPoint>(), x => x);
            _testErrorSeries.Series.Title = "Test error";

            model[0, 0] = _errorSeries;
            model[0, 1] = _testErrorSeries;

            PlotModel = model;
        }

        public PlotModelEx PlotModel { get; set; }

        public OptimizationReportEventArgs Report { get; set; }

        public string Message { get; set; } = string.Empty;

        internal void AddTestError(double error)
        {
            _testErrorSeries.AddPoint(new DataPoint(_errorSeries.SeriesPoints.Count, error));
            PlotModel.InvalidatePlot(true);
        }

        internal void UpdateReport(OptimizationReportEventArgs report)
        {
            Report = report;
            foreach (var error in report.Errors)
            {
                _errorSeries.AddPoint(new DataPoint(_errorSeries.SeriesPoints.Count, error));
            }

            if (_errorSeries.SeriesPoints.Count < 500)
            {
                PlotModel.ZoomX();
            }
            else
            {
                PlotModel.XAxis.Zoom(_errorSeries.SeriesPoints.Count - 500, _errorSeries.SeriesPoints.Count);
            }

            PlotModel.ZoomY();
            
            PlotModel.InvalidatePlot(true);
        }

        internal void AddMessage(string message)
        {
            const int maxBuffer = 1024 * 100;
            if (Message.Length >= maxBuffer)
            {
                Message = string.Empty;
            }
            Message += message + Environment.NewLine;
        }
    }
}