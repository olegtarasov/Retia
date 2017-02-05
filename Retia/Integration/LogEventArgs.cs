using System;

namespace Retia.Integration
{
	public class LogEventArgs : EventArgs
	{
		public LogEventArgs()
		{
		}

		public LogEventArgs(string message)
		{
			Message = message;
		}

		public string Message { get; set; }
	}
}