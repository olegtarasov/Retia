using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using MathNet.Numerics.LinearAlgebra;

namespace Retia.Helpers
{
    /// <summary>
    /// A collection of pinned matrix pointers.
    /// Pointers are unpinned when an object is disposed.
    /// </summary>
    /// <typeparam name="T">Data type.</typeparam>
    public struct MatrixPointers<T> : IDisposable where T : struct, IEquatable<T>, IFormattable
    {
        private readonly GCHandle[] _handles;

        private bool _disposed;

        /// <summary>
        /// Pins pointers to underlying matrix arrays and stores them for later use.
        /// The order of matrices is preserved.
        /// </summary>
        /// <param name="matrices">Matrices to pin pointers to.</param>
        public MatrixPointers(params Matrix<T>[] matrices)
        {
            _disposed = false;
            _handles = new GCHandle[matrices.Length];

            for (int i = 0; i < matrices.Length; i++)
            {
                _handles[i] = GCHandle.Alloc(matrices[i].AsColumnMajorArray(), GCHandleType.Pinned);
            }
        }

        /// <summary>
        /// Get a pointer to the matrix.
        /// </summary>
        /// <param name="idx">Matrix index.</param>
        /// <returns>Pointer to an underlying matrix array.</returns>
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