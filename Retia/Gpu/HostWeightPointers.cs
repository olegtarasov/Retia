using System;
using MathNet.Numerics.LinearAlgebra;
using Retia.Helpers;
using Retia.Neural;

namespace Retia.Gpu
{
    public struct HostWeightPointers<T> : IDisposable where T : struct, IEquatable<T>, IFormattable
    {
        private readonly MatrixPointers<T> _ptrs;
        private readonly HostWeightDefinition[] _defs;

        private bool _disposed;

        public HostWeightPointers(params NeuroWeight<T>[] weights) : this(false, weights)
        {
        }

        public HostWeightPointers(bool rowMajor, params NeuroWeight<T>[] weights)
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

            _ptrs = new MatrixPointers<T>(rowMajor, matrices);

            _defs = new HostWeightDefinition[weights.Length];
            cnt = -1;
            for (int i = 0; i < weights.Length; i++)
            {
                var weight = weights[i];

                _defs[i].Rows = weight.Weight.RowCount;
                _defs[i].Columns = weight.Weight.ColumnCount;
                _defs[i].SeqLength = 1;
                _defs[i].WeightPtr = _ptrs[++cnt];
                _defs[i].GradPtr = _ptrs[++cnt];
                _defs[i].Cache1Ptr = _ptrs[++cnt];
                _defs[i].Cache2Ptr = _ptrs[++cnt];
                _defs[i].CacheMPtr = _ptrs[++cnt];
            }
        }

        public HostWeightDefinition[] Definitions => _defs;

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