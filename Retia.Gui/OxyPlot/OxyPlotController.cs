using OxyPlot;
using OxyPlot.Wpf;
using Axis = OxyPlot.Axes.Axis;

namespace Retia.Gui.OxyPlot
{
	public class OxyPlotController : PlotController
	{
		public override bool HandleMouseWheel(IView view, OxyMouseWheelEventArgs args)
		{
			var plotView = view as PlotView;
			if (plotView == null)
			{
				return base.HandleMouseWheel(view, args);
			}

			var model = plotView.ActualModel;
			if (model == null)
			{
				return base.HandleMouseWheel(view, args);
			}

			if (args.IsControlDown)
			{
				if (args.IsShiftDown)
				{
					model.DefaultXAxis.Zoom(model.DefaultXAxis.Scale * (args.Delta > 0 ? 1.2 : 0.8));
					plotView.InvalidatePlot(false);
					return true;
				}

				Axis xAxis, yAxis;
				model.GetAxesFromPoint(args.Position, out xAxis, out yAxis);

				if (yAxis == null)
				{
					return base.HandleMouseWheel(view, args);
				}

				double y = yAxis.InverseTransform(args.Position.Y);

				yAxis.ZoomAt(args.Delta > 0 ? 1.2 : 0.8, y);
				plotView.InvalidatePlot(false);
				return true;
			}

			if (args.IsShiftDown)
			{
				plotView.Model.DefaultXAxis.Pan((double)args.Delta / 3);
				plotView.InvalidatePlot(false);
				return true;
			}

			return base.HandleMouseWheel(view, args);
		}
	}
}