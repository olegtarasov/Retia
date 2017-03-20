using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Storage;

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
        private readonly IntPtr[] _pointers;
        private readonly bool _rowMajor;
        private readonly Dictionary<Matrix<T>, T[]> _arrayMap;

        private bool _disposed;

        /// <summary>
        /// Pins pointers to underlying matrix arrays and stores them for later use.
        /// The order of matrices is preserved.
        /// </summary>
        /// <param name="matrices">Matrices to pin pointers to.</param>
        public MatrixPointers(params Matrix<T>[] matrices) : this(false, matrices)
        {
        }

        /// <summary>
        /// Pins pointers to underlying matrix arrays and stores them for later use.
        /// The order of matrices is preserved.
        /// </summary>
        /// <remarks>
        /// When <see cref="rowMajor"/> is set, arrays are decoupled from original matrices.
        /// Changes made to arrays will be synchronized only upon <see cref="Dispose"/>.
        /// </remarks>
        /// <param name="rowMajor">
        /// Indicates whether matrices should be converted to row-major format.
        /// </param>
        /// <param name="matrices">Matrices to pin pointers to.</param>
        public MatrixPointers(bool rowMajor, params Matrix<T>[] matrices)
        {
            _rowMajor = rowMajor;

            _arrayMap = rowMajor ? new Dictionary<Matrix<T>, T[]>() : null;

            _disposed = false;
            _handles = new GCHandle[matrices.Length];
            _pointers = new IntPtr[matrices.Length];

            for (int i = 0; i < matrices.Length; i++)
            {
                var matrix = matrices[i];
                T[] array;
                if (rowMajor)
                {
                    array = matrix.ToRowMajorArray();
                    _arrayMap[matrix] = array;
                }
                else
                {
                    array = matrix.AsColumnMajorArray();
                }
                
                _handles[i] = GCHandle.Alloc(array, GCHandleType.Pinned);
                _pointers[i] = _handles[i].AddrOfPinnedObject();
            }
        }

        /// <summary>
        /// Get a pointer to the matrix.
        /// </summary>
        /// <param name="idx">Matrix index.</param>
        /// <returns>Pointer to an underlying matrix array.</returns>
        public IntPtr this[int idx] => _pointers[idx];

        public IntPtr[] Pointers => _pointers;

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;

            if (_rowMajor)
            {
                foreach (var pair in _arrayMap)
                {
                    var matrix = pair.Key;
                    var storage = DenseColumnMajorMatrixStorage<T>.OfRowMajorArray(matrix.RowCount, matrix.ColumnCount, pair.Value);
                    storage.CopyTo(matrix.Storage);
                }

                _arrayMap.Clear();
            }

            for (int i = 0; i < _handles.Length; i++)
            {
                _handles[i].Free();
            }
        }
    }
}