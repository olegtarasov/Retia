using System.Collections.Concurrent;

namespace Retia.Helpers
{
    public static class CollectionExtensions
    {
        public static double ConcurrentMax(this ConcurrentQueue<double> queue)
        {
            double result = double.MinValue;
            double cur;
            while (queue.TryDequeue(out cur))
            {
                if (cur > result)
                {
                    result = cur;
                }
            }

            return result;
        }
    }
}