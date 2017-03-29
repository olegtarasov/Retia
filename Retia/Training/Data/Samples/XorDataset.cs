using System;
using System.IO;
using System.Linq;
using Retia.Mathematics;
using Retia.RandomGenerator;

namespace Retia.Training.Data.Samples
{
    public class XorDataset : IDataSet<float>
    {
        private int cnt = 0;
        private bool _rand;

        public XorDataset(bool random)
        {
            _rand = random;
        }
        public IDataSet<float> Clone()
        {
            throw new NotSupportedException();
        }

        public void Save(Stream stream)
        {
            throw new NotSupportedException();
        }

        public event EventHandler DataSetReset;
        public Sample<float> GetNextSample()
        {
            throw new NotSupportedException();
        }

        public TrainingSequence<float> GetNextSamples(int count)
        {
            var tuples = Enumerable.Range(0, count)
                                   .Select(x =>
                                   {
                                       int a, b;
                                       if (_rand)
                                       {
                                           a = SafeRandom.Generator.Next(2);
                                           b = SafeRandom.Generator.Next(2);
                                       }
                                       else
                                       {
                                           a = cnt & 0x01;
                                           b = (cnt & 0x02) >> 1;
                                       }
                                       cnt++;
                                       return new Tuple<int, int, int>(a, b, a ^ b);
                                   }).ToList();
            return new TrainingSequence<float>(tuples.Select(x => MatrixFactory.Create<float>(2, 1, x.Item1, x.Item2)).ToList(), tuples.Select(x => MatrixFactory.Create<float>(1, 1, x.Item3)).ToList());
        }

        public void Reset()
        {
        }

        public int SampleCount { get; } = 0;
        public int InputSize { get; } = 2;
        public int TargetSize { get; } = 1;
        public int BatchSize { get; } = 1;
    }
}