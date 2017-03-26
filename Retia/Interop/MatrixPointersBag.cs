using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Storage;

namespace Retia.Interop
{
    /// <summary>
    /// A collection of pinned matrix pointers.
    /// Pointers are unpinned when an object is disposed.
    /// </summary>
    /// <typeparam name="T">Data type.</typeparam>
    public struct MatrixPointersBag<T> : IDisposable where T : struct, IEquatable<T>, IFormattable
    {
        private readonly GCHandle[] _handles;
        private readonly IntPtr[] _pointers;
        private readonly MatrixDefinition[] _defs;
        private readonly bool _rowMajor;
        private readonly List<Tuple<Matrix<T>, T[]>> _arrayMap;

        private bool _disposed;

        /// <summary>
        /// Pins pointers to underlying matrix arrays and stores them for later use.
        /// The order of matrices is preserved.
        /// </summary>
        /// <param name="matrices">Matrices to pin pointers to.</param>
        public MatrixPointersBag(params Matrix<T>[] matrices) : this(false, matrices)
        {
        }

        /// <summary>
        /// Pins pointers to underlying matrix arrays and stores them for later use.
        /// The order of matrices is preserved.
        /// </summary>
        /// <param name="generateDefinitions">Whether to generate matrix definitions for GPU transfer.</param>
        /// <param name="matrices">Matrices to pin pointers to.</param>
        public MatrixPointersBag(bool generateDefinitions, params Matrix<T>[] matrices) : this(false, generateDefinitions, matrices)
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
        /// <param name="generateDefinitions">Whether to generate matrix definitions for GPU transfer.</param>
        /// <param name="matrices">Matrices to pin pointers to.</param>
        public MatrixPointersBag(bool rowMajor, bool generateDefinitions, params Matrix<T>[] matrices)
        {
            _rowMajor = rowMajor;

            _arrayMap = rowMajor ? new List<Tuple<Matrix<T>, T[]>>() : null;
            _defs = generateDefinitions ? new MatrixDefinition[matrices.Length] : null;

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
                    _arrayMap.Add(new Tuple<Matrix<T>, T[]>(matrix, array));
                }
                else
                {
                    array = matrix.AsColumnMajorArray();
                }
                
                _handles[i] = GCHandle.Alloc(array, GCHandleType.Pinned);

                var ptr = _handles[i].AddrOfPinnedObject();
                _pointers[i] = ptr;

                if (generateDefinitions)
                {
                    _defs[i] = new MatrixDefinition(matrix.RowCount, matrix.ColumnCount, 1, ptr);
                }
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
                    throw new ObjectDisposedException(nameof(MatrixPointersBag<T>));

                return _pointers[idx];
            }
        }

        public MatrixDefinition[] Definitions
        {
            get
            {
                if (_defs == null)
                    throw new InvalidOperationException("Matrix definitions has not been generated!");

                return _defs;
            }
        }

        public void Dispose()
        {
            if (_disposed)
                return;

            _disposed = true;

            if (_rowMajor)
            {
                foreach (var pair in _arrayMap)
                {
                    var matrix = pair.Item1;
                    var storage = DenseColumnMajorMatrixStorage<T>.OfRowMajorArray(matrix.RowCount, matrix.ColumnCount, pair.Item2);
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