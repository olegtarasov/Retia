using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MarkdownLog;

namespace Retia.Training.Testers
{
	public class ConfusionMatrix : TestResultBase
	{
		private readonly List<ClassResult> _classes;

		public ConfusionMatrix(int classCount)
		{
			_classes = Enumerable.Range(0, classCount).Select(x => new ClassResult(x, classCount)).ToList();
		}

		public ConfusionMatrix(List<string> classNames)
		{
			_classes = Enumerable.Range(0, classNames.Count).Select(x => new ClassResult(x, classNames.Count, classNames[x])).ToList();
		}

		public IReadOnlyList<ClassResult> Classes => _classes;

		public double AvgPrecision { get; set; }
		public double AvgRecall { get; set; }
		public double AvgAccuracy { get; set; }
		public double AvgF1 { get; set; }
		public double Error { get; set; }

		public void CalculateResult(int sampleCount)
		{
			for (int i = 0; i < _classes.Count; i++)
			{
				var cl = _classes[i];

				cl.TargetPositiveCount = cl.TruePositives + cl.FalseNegatives;
				cl.PredictedPositiveCount = cl.TruePositives + cl.FalsePositives;

				cl.Recall = (double)cl.TruePositives / cl.TargetPositiveCount;
				cl.Precision = (double)cl.TruePositives / cl.PredictedPositiveCount;
				cl.Accuracy = (double)(cl.TruePositives + cl.TrueNegatives) / sampleCount;
				cl.F1 = 2 * cl.Precision * cl.Recall / (cl.Precision + cl.Recall);

				AvgPrecision += cl.Precision;
				AvgRecall += cl.Recall;
				AvgAccuracy += cl.Accuracy;
				AvgF1 += cl.F1;
			}

			AvgPrecision /= _classes.Count;
			AvgRecall /= _classes.Count;
			AvgAccuracy /= _classes.Count;
			AvgF1 /= _classes.Count;
		}

		public void Prediction(int targetClass, int predictedClass)
		{
			_classes[predictedClass].ClassPredictions[targetClass] = _classes[predictedClass].ClassPredictions[targetClass] + 1;

			for (int i = 0; i < _classes.Count; i++)
			{
				var cl = _classes[i];

				if (i == predictedClass)
				{
					if (i == targetClass)
					{
						cl.TruePositives++;
					}
					else
					{
						cl.FalsePositives++;
					}
				}
				else
				{
					if (i == targetClass)
					{
						cl.FalseNegatives++;
					}
					else
					{
						cl.TrueNegatives++;
					}
				}
			}
		}

		public override string GetReport()
		{
			// Column getters
			var funcs = new List<Func<ClassResult, string>>();

			// Row title column
			funcs.Add(result => $"P-{result.ClassTitle}");

			// Prediction columns
			for (int i = 0; i < _classes.Count; i++)
			{
				int idx = i;
				funcs.Add(new Func<ClassResult, string>(result => result.ClassPredictions[idx].ToString()));
			}

			// Total predicted column
			funcs.Add(result => result.PredictedPositiveCount.ToString());

			var table = _classes.ToMarkdownTable(funcs.ToArray());

			// Add column labels manually
			var columns = new List<TableColumn>();
			columns.Add(new TableColumn()); // Empty for row titles
			columns.AddRange(_classes.Select(x => new TableColumn {HeaderCell = new TableCell {Text = $"T-{x.ClassTitle}"}, Alignment = TableColumnAlignment.Center}));
			columns.Add(new TableColumn {HeaderCell = new TableCell {Text = "Total"}, Alignment = TableColumnAlignment.Center});

			table.Columns = columns;

			// Add custom rows
			var rows = table.Rows.ToList();
			rows.Add(GetCustomRow("Target total", result => result.TargetPositiveCount.ToString()));
			rows.Add(GetCustomRow("Accuracy", result => (result.Accuracy*100.0).ToString("F2")));
			rows.Add(GetCustomRow("Precision", result => (result.Precision * 100.0).ToString("F2")));
			rows.Add(GetCustomRow("Recall", result => (result.Recall * 100.0).ToString("F2")));
			rows.Add(GetCustomRow("F1", result => (result.F1 * 100.0).ToString("F2")));

			table.Rows = rows;

			// Add averages
			var sb = new StringBuilder(table.ToMarkdown());
			sb.AppendLine();
			sb.AppendLine($"Average accuracy: {AvgAccuracy * 100.0:F2}%");
			sb.AppendLine($"Average precision: {AvgPrecision * 100.0:F2}%");
			sb.AppendLine($"Average recall: {AvgRecall * 100.0:F2}%");
			sb.AppendLine($"Average F1: {AvgF1 * 100.0:F2}%");
			sb.AppendLine($"Error: {Error:F3}");

			return sb.ToString();
		}

		private TableRow GetCustomRow(string title, Func<ClassResult, string> cellFunc)
		{
			var cells = new List<TableCell>();
			cells.Add(new TableCell {Text = title});
			cells.AddRange(_classes.Select(x => new TableCell {Text = cellFunc(x)}));

			return new TableRow {Cells = cells};
		}
	}
}