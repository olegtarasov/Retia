using System;
using MathNet.Numerics.LinearAlgebra;
using Retia.Helpers;

namespace Retia.Gpu
{
    public struct HostMatrixPointers<T> : IDisposable where T : struct, IEquatable<T>, IFormattable
    {
        private readonly MatrixPointers<T> _ptrs;
        private readonly HostMatrixDefinition[] _defs;

        private bool _disposed;

        public HostMatrixPointers(params Matrix<T>[] matrices) : this(false, matrices)
        {
        }

        public HostMatrixPointers(bool rowMajor, params Matrix<T>[] matrices)
        {
            _disposed = false;
            _ptrs = new MatrixPointers<T>(rowMajor, matrices);

            _defs = new HostMatrixDefinition[matrices.Length];
            for (int i = 0; i < matrices.Length; i++)
            {
                var mat = matrices[i];
                
                _defs[i].Rows = mat.RowCount;
                _defs[i].Columns = mat.ColumnCount;
                _defs[i].SeqLength = 1;
                _defs[i].Pointer = _ptrs[i];
            }
        }

        public HostMatrixDefinition[] Definitions => _defs;

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