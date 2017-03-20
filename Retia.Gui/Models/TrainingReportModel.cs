using System;
using System.Linq;
using OxyPlot;
using PropertyChanged;
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
            _errorSeries.Series.Decimator = Decimator.Decimate;

            _testErrorSeries = SeriesBuilder.LineSeries(Enumerable.Empty<DataPoint>(), x => x);
            _testErrorSeries.Series.Title = "Test error";
            _testErrorSeries.Series.Decimator = Decimator.Decimate;

            model[0, 0] = _errorSeries;
            model[0, 1] = _testErrorSeries;

            PlotModel = model;
        }

        public PlotModelEx PlotModel { get; set; }

        // Workaround for Ammy's faulty support for class instances in resources
        public OxyPlotController PlotController { get; } = new OxyPlotController();

        public OptimizationReportEventArgs Report { get; set; }

        public int PlotRange { get; set; } = 10000;

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
                _errorSeries.AddPoint(new DataPoint(_errorSeries.SeriesPoints.Count, error.FilteredError));
            }

            if (PlotRange == 0 || _errorSeries.SeriesPoints.Count < PlotRange)
            {
                PlotModel.ZoomX();
            }
            else
            {
                PlotModel.XAxis.Zoom(_errorSeries.SeriesPoints.Count - PlotRange, _errorSeries.SeriesPoints.Count);
            }

            PlotModel.ZoomY();
            
            PlotModel.InvalidatePlot(true);
        }

        //internal void AddMessage(string message)
        //{
        //    const int maxBuffer = 1024 * 100;
        //    if (Message.Length >= maxBuffer)
        //    {
        //        Message = string.Empty;
        //    }
        //    Message += message + Environment.NewLine;
        //}
    }
}