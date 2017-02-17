using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using MathNet.Numerics.LinearAlgebra;

namespace Retia.Helpers
{
    public struct MatrixPointers<T> : IDisposable where T : struct, IEquatable<T>, IFormattable
    {
        private readonly GCHandle[] _handles;

        private bool _disposed;

        public MatrixPointers(params Matrix<T>[] matrices)
        {
            _disposed = false;
            _handles = new GCHandle[matrices.Length];

            for (int i = 0; i < matrices.Length; i++)
            {
                _handles[i] = GCHandle.Alloc(matrices[i].AsColumnMajorArray(), GCHandleType.Pinned);
            }
        }

        public IntPtr this[int idx]
        {
            get
            {
                if (_disposed)
                    throw new ObjectDisposedException(nameof(MatrixPointers<T>));

                return _handles[idx].AddrOfPinnedObject();
            }
        }

        public void Dispose()
        {
            _disposed = true;
            for (int i = 0; i < _handles.Length; i++)
            {
                _handles[i].Free();
            }
        }
    }
}