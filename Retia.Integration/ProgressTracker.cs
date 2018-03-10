namespace Retia.Integration
{
    /// <summary>
    /// Tracks the progress of an operation and indicates when to report progress.
    /// </summary>
	public class ProgressTracker
	{
	    private readonly int _count;
	    private readonly int _reportCount;

	    private int _nextReport;

        /// <summary>
        /// Creates a new tracker with specified reporting frequency.
        /// </summary>
        /// <param name="count">The total number of operation ticks.</param>
        /// <param name="reportPercent">Report frequency in percent [0..100].</param>
	    public ProgressTracker(int count, int reportPercent = 100)
		{
			_count = count;
			_reportCount = count / reportPercent;
			_nextReport = _reportCount;
		}

        /// <summary>
        /// The number of tracked ticks.
        /// </summary>
	    public int Current { get; private set; }

        /// <summary>
        /// Indicates whether the progress should be reported according to
        /// reporting frequency.
        /// </summary>
        /// <param name="current">Current tick.</param>
        public bool ShouldReport(int current)
		{
			if (current > _nextReport)
			{
				_nextReport += _reportCount;
				return true;
			}

			return false;
		}

        /// <summary>
        /// Indicates whether the progress should be reported according to
        /// reporting frequency. Increases tick count by 1.
        /// </summary>
        public bool ShouldReport()
	    {
	        return ShouldReport(Current++);
	    }
	}
}