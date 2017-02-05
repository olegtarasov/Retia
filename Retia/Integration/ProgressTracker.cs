namespace Retia.Integration
{
	public class ProgressTracker
	{
	    private readonly int _count;
	    private readonly int _reportCount;

	    private int _nextReport;

	    public ProgressTracker(int count, int reportPercent = 100)
		{
			_count = count;
			_reportCount = count / reportPercent;
			_nextReport = _reportCount;
		}

	    public int Current { get; private set; }

	    public bool ShouldReport(int current)
		{
			if (current > _nextReport)
			{
				_nextReport += _reportCount;
				return true;
			}

			return false;
		}

	    public bool ShouldReport()
	    {
	        return ShouldReport(Current++);
	    }
	}
}