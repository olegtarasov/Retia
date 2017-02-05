using System;
using System.Collections.Generic;
using Retia.Training.Testers;
using Xunit;

namespace Retia.Tests.Training
{
	public class ConfusionTableTests
	{
		[Fact]
		public void Test()
		{
			const int ClassCount = 3;
			var matrix = new ConfusionMatrix(new List<string> {"Foo", "Bar", "Baz"});
			var rnd = new Random();

			for (int i = 0; i < 1000; i++)
			{
				int target = rnd.Next(ClassCount);
				int predicted = rnd.Next(ClassCount);

				matrix.Prediction(target, predicted);
			}

			matrix.CalculateResult(1000);
			string result = matrix.GetReport();
		}
	}
}