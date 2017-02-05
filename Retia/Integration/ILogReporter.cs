using System;

namespace Retia.Integration
{
	public interface ILogReporter
	{
		event EventHandler<LogEventArgs> Message;
	}
}