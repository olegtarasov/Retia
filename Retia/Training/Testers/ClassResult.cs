using System.Collections.Generic;
using System.Linq;

namespace Retia.Training.Testers
{
	public class ClassResult
	{
		public ClassResult(int classIndex, int classCount, string className) : this(classIndex, classCount)
		{
			ClassName = className;
		}

		public ClassResult(int classIndex, int classCount)
		{
			ClassIndex = classIndex;
			ClassPredictions = Enumerable.Range(0, classCount).ToDictionary(x => x, x => 0);
		}

		public double Precision { get; set; }
		public double Recall { get; set; }
		public double F1 { get; set; }
		public double Accuracy { get; set; }

		public int TargetPositiveCount { get; set; }
		public int PredictedPositiveCount { get; set; }

		public int TruePositives { get; set; }
		public int FalsePositives { get; set; }
		public int TrueNegatives { get; set; }
		public int FalseNegatives { get; set; }

		/// <summary>
		/// Contains a map where key is class number and value is 
		/// a number of times that class was predicted instead of
		/// the current class. Also contains number of predictions
		/// for the current class (true positives).
		/// </summary>
		public Dictionary<int, int> ClassPredictions { get; }

		public int ClassIndex { get; }
		public string ClassName { get; }

		public string ClassTitle => string.IsNullOrEmpty(ClassName) ? ClassIndex.ToString() : ClassName;
	}
}