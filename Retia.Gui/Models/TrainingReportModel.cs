using System;
using System.Linq;
using OxyPlot;
using PropertyChanged;
using Retia.Gui.OxyPlot;
using Retia.Training.Trainers;
using Retia.Training.Trainers.Sessions;

namespace Retia.Gui.Models
{
    [ImplementPropertyChanged]
    public class TrainingReportModel
    {
        private readonly LineSeriesWrapper<DataPoint> _errorSeries;
        private readonly LineSeriesWrapper<DataPoint> _testErrorSeries;

        private int _errDecCnt = 0;
        private long _errIdx = 0;
        private long _errDecimator = 0;

        private int _testDecCnt = 0;
        private long _testIdx = 0;
        private long _testDecimator = 0;

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

        public int PlotResolution { get; set; } = 1000;

        internal void AddTestError(TrainingSessionBase session, double error)
        {
            int iters = session.Epoch * session.IterationsPerEpoch;

            if (_testDecimator == 0 || _testIdx >= _testDecimator)
            {
                _testIdx = 0;
                _testErrorSeries.AddPoint(new DataPoint(iters + session.Iteration, error));
            }
            else
            {
                _testIdx++;
            }

            if (_testErrorSeries.SeriesPoints.Count >= PlotResolution)
            {
                _testErrorSeries.DecimatePoints(2);
                _testDecCnt++;
                _testDecimator = _testDecCnt * 2;
                _testIdx = 0;
            }


            PlotModel.InvalidatePlot(true);
        }

        internal void UpdateReport(OptimizationReportEventArgs report)
        {
            Report = report;
            int iters = report.Session.Epoch * report.Session.IterationsPerEpoch;
            for (int i = 0; i < report.Errors.Count; i++, _errIdx++)
            {
                if (_errDecimator == 0 || _errIdx >= _errDecimator)
                {
                    _errIdx = 0;
                    _errorSeries.AddPoint(new DataPoint(iters + report.Session.Iteration - report.Errors.Count + i, report.Errors[i].FilteredError));
                }
            }

            if (_errorSeries.SeriesPoints.Count >= PlotResolution)
            {
                _errorSeries.DecimatePoints(2);
                _errDecCnt++;
                _errDecimator = _errDecCnt * 2;
                _errIdx = 0;
            }

            PlotModel.ZoomX();
            PlotModel.ZoomY();

            PlotModel.InvalidatePlot(true);
        }
    }
}