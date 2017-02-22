using System;

namespace Retia.Mathematics
{
    /// <summary>
    /// A singleton class to access a specific math provider.
    /// </summary>
    /// <typeparam name="T">Data type</typeparam>
    public static class MathProvider<T> where T : struct, IEquatable<T>, IFormattable
    {
        private static Lazy<MathProviderBase<T>> _instance = new Lazy<MathProviderBase<T>>(() =>
        {
            return typeof(T) == typeof(double) 
                ? new MathProviderImplD() as MathProviderBase<T> 
                : new MathProviderImpl() as MathProviderBase<T>;
        });

        /// <summary>
        /// Math provider instance.
        /// </summary>
        public static MathProviderBase<T> Instance => _instance.Value;
    }
}