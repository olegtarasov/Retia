using System;

namespace Retia.Mathematics
{
    public static class MathProvider<T> where T : struct, IEquatable<T>, IFormattable
    {
        private static Lazy<MathProviderBase<T>> _instance = new Lazy<MathProviderBase<T>>(() =>
        {
            return typeof(T) == typeof(double) 
                ? new MathProviderImplD() as MathProviderBase<T> 
                : new MathProviderImpl() as MathProviderBase<T>;
        });

        public static MathProviderBase<T> Instance => _instance.Value;
    }
}