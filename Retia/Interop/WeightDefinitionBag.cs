using System;
using MathNet.Numerics.LinearAlgebra;
using Retia.Helpers;
using Retia.Neural;

namespace Retia.Interop
{
    public struct WeightDefinitionBag<T> : IDisposable where T : struct, IEquatable<T>, IFormattable
    {
        private readonly MatrixPointersBag<T> _ptrs;
        private readonly WeightDefinition[] _defs;

        private bool _disposed;

        public WeightDefinitionBag(params NeuroWeight<T>[] weights) : this(false, weights)
        {
        }

        public WeightDefinitionBag(bool rowMajor, params NeuroWeight<T>[] weights)
        {
            _disposed = false;

            var matrices = new Matrix<T>[weights.Length * 5];
            int cnt = -1;
            for (int i = 0; i < weights.Length; i++)
            {
                var weigth = weights[i];
                matrices[++cnt] = weigth.Weight;
                matrices[++cnt] = weigth.Gradient;
                matrices[++cnt] = weigth.Cache1;
                matrices[++cnt] = weigth.Cache2;
                matrices[++cnt] = weigth.CacheM;
            }

            _ptrs = new MatrixPointersBag<T>(rowMajor, false, matrices);

            _defs = new WeightDefinition[weights.Length];
            cnt = -1;
            for (int i = 0; i < weights.Length; i++)
            {
                var weight = weights[i];
                _defs[i] = new WeightDefinition(weight.Weight.RowCount, weight.Weight.ColumnCount, 1,
                    _ptrs[++cnt], _ptrs[++cnt], _ptrs[++cnt], _ptrs[++cnt], _ptrs[++cnt]); // Oh yeah.
            }
        }

        public WeightDefinition[] Definitions => _defs;

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;

            // ReSharper disable once ImpureMethodCallOnReadonlyValueField
            _ptrs.Dispose();
        }
    }
}