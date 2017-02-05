using System.Threading;

namespace Retia.RandomGenerator
{
    /// <summary>
    /// Thread-safe uniform random generator implementation.
    /// </summary>
    public static class SafeRandom
    {
        private static readonly ThreadLocal<IRandom> _random = new ThreadLocal<IRandom>(() => new UniformRandom());

        /// <summary>
        /// Thread-safe random generator instance.
        /// </summary>
        public static IRandom Generator => TestGenerator.Value ?? _random.Value;

        /// <summary>
        /// Random generator used for testing.
        /// </summary>
        public static ThreadLocal<IRandom> TestGenerator { get; private set; } = new ThreadLocal<IRandom>();

        /// <summary>
        /// Sets a test RNG for the current thread.
        /// </summary>
        /// <param name="random"></param>
        public static void SetTestGenerator(IRandom random)
        {
            TestGenerator.Value = random;
        }

        /// <summary>
        /// Removes a test RNG for the current thread.
        /// </summary>
        public static void ClearTestGenerator()
        {
            TestGenerator.Value = null;
        }
    }
}