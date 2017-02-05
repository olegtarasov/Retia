using System;
using System.Security.Cryptography;

namespace Retia.RandomGenerator
{
    /// <summary>
    /// Standard .NET random generator with uniform probability distribution.
    /// Random seed is generated using high-quality RNG from <see cref="RNGCryptoServiceProvider "/>
    /// </summary>
    internal class UniformRandom : BaseRandom
    {
        private static readonly RandomNumberGenerator _global = RandomNumberGenerator.Create();

        private readonly Random _rnd;

        public UniformRandom()
        {
            byte[] buffer = new byte[4];
            _global.GetBytes(buffer);
            _rnd = new Random(BitConverter.ToInt32(buffer, 0));
        }

        public override int Next()
        {
            return _rnd.Next();
        }

        public override int Next(int maxValue)
        {
            return _rnd.Next(maxValue);
        }

        public override int Next(int minValue, int maxValue)
        {
            return _rnd.Next(minValue, maxValue);
        }

        public override double NextDouble()
        {
            return _rnd.NextDouble();
        }

        public override double NextDouble(double minValue, double maxValue)
        {
            var r = _rnd.NextDouble() * (maxValue - minValue);
            return minValue + r;
        }
    }
}