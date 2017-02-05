using System;

namespace Retia.Integration
{
	public class ProgressEventArgs : EventArgs
	{
		public ProgressEventArgs()
		{
		}

		public ProgressEventArgs(string message, double value, double maxValue)
		{
			Message = $"{message} {value} of {maxValue}";
			MaxValue = maxValue;
			Value = value;
		}

		public ProgressEventArgs(string message, bool isIntermediate)
		{
			Message = message;
			IsIntermediate = isIntermediate;
		}

		public ProgressEventArgs(bool isComplete)
		{
			IsComplete = isComplete;
		}

		public string Message { get; set; }
		public double Value { get; set; }
		public double MaxValue { get; set; }
		public bool IsComplete { get; set; }
		public bool IsIntermediate { get; set; }
	}
}