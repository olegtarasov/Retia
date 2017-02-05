using System.Collections.Generic;

namespace Retia.RandomGenerator
{
    public abstract class BaseRandom : IRandom
    {
        public abstract int Next();
        public abstract int Next(int maxValue);
        public abstract int Next(int minValue, int maxValue);
        public abstract double NextDouble();
        public abstract double NextDouble(double minValue, double maxValue);

        public T NextFrom<T>(IList<T> items)
        {
            return items[Next(items.Count)];
        }
    }
}