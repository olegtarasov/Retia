using System;
using System.Collections.Generic;
using System.Threading;
using Retia.RandomGenerator;

namespace Retia.Tests.Plumbing
{
    public class FakeRandom : BaseRandom, IDisposable
    {
        private readonly double[] _values;
        private readonly IRandom _oldRandom;

        private int _idx = 0;

        public FakeRandom(params double[] values)
        {
            _values = values;
            _oldRandom = SafeRandom.TestGenerator?.Value;

            SafeRandom.SetTestGenerator(this);
        }

        public override int Next()
        {
            return (int)NextDouble();
        }

        public override int Next(int maxValue)
        {
            return (int)NextDouble();
        }

        public override int Next(int minValue, int maxValue)
        {
            return (int)NextDouble();
        }

        public override double NextDouble()
        {
            if (_idx == _values.Length)
            {
                throw new IndexOutOfRangeException();
            }

            var result = _values[_idx];
            _idx++;

            return result;
        }

        public override double NextDouble(double minValue, double maxValue)
        {
            return NextDouble();
        }

        public void Dispose()
        {
            if (_oldRandom != null)
            {
                SafeRandom.SetTestGenerator(_oldRandom);
            }
            else
            {
                SafeRandom.ClearTestGenerator();
            }
        }
    }
}