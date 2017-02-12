using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Diagnostics.Eventing.Reader;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Runtime;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Serialization;
using System.Text;
using System.Text.RegularExpressions;
using Retia.Integration;
using Retia.RandomGenerator;

namespace Retia.Mathematics
{
    public enum TransposeOptions
    {
        NoTranspose = 111,
        Transpose = 112,
    }

    [Serializable]
    public class Matrix : ICloneable<Matrix>, IStreamWritable, ISerializable
    {
        private const string BLAS_DLL_S = "mkl_rt.dll";
        private static BlasBackendBase _backend = new MklBlasBackend();

        public readonly int Rows;
        public readonly int Cols;
        public readonly int Length;

        private readonly float[] _storage;

        public Matrix(int rows, int cols) : this(rows, cols, new float[rows * cols])
        {
        }

        public Matrix(int rows, int cols, IList<float> contents) : this(rows, cols)
        {
            if (contents.Count < Length)
            {
                throw new InvalidOperationException("Not enough data for the matrix!");
            }

            for (int i = 0; i < Length; i++)
            {
                _storage[i] = contents[i];
            }
        }

        public Matrix(int rows, int cols, float value) : this(rows, cols, Enumerable.Repeat(value, rows * cols).ToArray())
        {
        }

        [ExcludeFromCodeCoverage]
        public Matrix(SerializationInfo info, StreamingContext context)
        {
            Rows = info.GetInt32(nameof(Rows));
            Cols = info.GetInt32(nameof(Cols));
            Length = Rows * Cols;

            _storage = new float[Length];
            for (int i = 0; i < Length; i++)
            {
                _storage[i] = info.GetSingle($"_{i}");
            }
        }

        private Matrix(int rows, int cols, float[] storage)
        {
            Rows = rows;
            Cols = cols;
            Length = Rows * Cols;
            _storage = storage;
        }

        public static BlasBackendBase Backend { get { return _backend; } set { _backend = value; } }

        // TODO: This is an estimate of L2 norm, should replace with true norm or abandon
        //public float L2
        //{
        //    get
        //    {
        //        var norm = 0.0;
        //        for (int i = 0; i < Length; i++)
        //            norm += _storage[i] * _storage[i];
        //        return norm;
        //    }
        //}

        // Testing only.
        [ExcludeFromCodeCoverage]
        internal float[] Storage => _storage;

        [ExcludeFromCodeCoverage]
        public string Text => ToString();

        [ExcludeFromCodeCoverage]
        public bool IsColumnVector => Cols == 1;

        [ExcludeFromCodeCoverage]
        public bool IsRowVector => Rows == 1;

        [ExcludeFromCodeCoverage]
        public bool IsVector => Cols == 1 || Rows == 1;

        public float this[int row, int col]
        {
            get
            {
                CheckBounds(row, col);
                return _storage[col * Rows + row];
            }
            set
            {
                CheckBounds(row, col);
                _storage[col * Rows + row] = value;
            }
        }

        public static Matrix Load(Stream stream)
        {
            using (var reader = new BinaryReader(stream, Encoding.UTF8, true))
            {
                int rows = reader.ReadInt32();
                int cols = reader.ReadInt32();
                int len = rows * cols;
                var mat = new float[len];
                for (int i = 0; i < len; i++)
                {
                    mat[i] = reader.ReadSingle();
                }

                return new Matrix(rows, cols, mat);
            }
        }

        public static Matrix ParseString(string input)
        {
            input = input.Replace(',', '.').Trim();

            var rows = input.Split(new[] { "\r", "\n", "\r\n" }, StringSplitOptions.RemoveEmptyEntries);
            if (rows.Length == 0)
            {
                throw new InvalidOperationException("Invalid matrix format — no rows detected!");
            }

            var cols = rows[0].Trim().Split(new[] { '\t', ' ' }, StringSplitOptions.RemoveEmptyEntries);
            if (cols.Length == 0)
            {
                throw new InvalidOperationException("Invalid matrix format — no columns detected!");
            }

            int numCols = cols.Length;

            var matrix = new Matrix(rows.Length, cols.Length);
            for (int row = 0; row < rows.Length; row++)
            {
                cols = rows[row].Trim().Split(new[] { '\t', ' ' }, StringSplitOptions.RemoveEmptyEntries);

                if (cols.Length != numCols)
                {
                    throw new InvalidOperationException($"Not enough columns in row {row}!");
                }

                for (int col = 0; col < cols.Length; col++)
                {
                    matrix[row, col] = float.Parse(cols[col], NumberFormatInfo.InvariantInfo); // Bubble up the format error
                }
            }

            return matrix;
        }

        public static Matrix RandomMatrix(int rows, int cols, float min, float max)
        {
            var random = SafeRandom.Generator;
            var matrix = new Matrix(rows, cols);
            for (int i = 0; i < matrix.Length; i++)
                matrix._storage[i] = (float)random.NextDouble(min, max);
            return matrix;
        }

        public static Matrix RandomMatrix(int rows, int cols, float dispersion)
        {
            return RandomMatrix(rows, cols, -dispersion, dispersion);
        }

        public static Matrix RandomMaskMatrix(int rows, int cols, float trueProb)
        {
            var random = SafeRandom.Generator;
            var matrix = new Matrix(rows, cols);
            for (int i = 0; i < matrix.Length; i++)
                matrix._storage[i] = random.NextDouble() < trueProb ? 1 : 0;
            return matrix;
        }


        //   O P E R A T O R S
        //May St. Bjarne Stroustrup have mercy on my soul for violating incapsulation rules
        //but this is the only way to make BLAS happy
        public static implicit operator float[] (Matrix m)
        {
            return m._storage;
        }

        public static explicit operator Matrix(List<float> vectorContents)
        {
            return new Matrix(vectorContents.Count, 1, vectorContents);
        }

        public static explicit operator Matrix(float[] vectorContents)
        {
            return new Matrix(vectorContents.Length, 1, vectorContents);
        }

        public static Matrix operator -(Matrix m)
        {
            return Multiply(-1, m);
        }

        public static Matrix operator +(Matrix m1, Matrix m2)
        {
            return Add(m1, m2);
        }

        public static Matrix operator -(Matrix m1, Matrix m2)
        {
            return Add(m1, -m2);
        }

        public static Matrix operator *(float n, Matrix m)
        {
            return Multiply(n, m);
        }

        public static Matrix operator *(Matrix m1, Matrix m2)
        {
            var result = new Matrix(m1.Rows, m2.Cols);

            result.Accumulate(m1, m2);
            return result;
        }

        public static Matrix operator ^(Matrix m1, Matrix m2)
        {
            return HadamardMul(m1, m2);
        }

        public Matrix Clone()
        {
            var matrix = new Matrix(Rows, Cols);
            Array.Copy(_storage, matrix._storage, _storage.Length);

            return matrix;
        }

        public void Save(Stream stream)
        {
            using (var writer = new BinaryWriter(stream, Encoding.UTF8, true))
            {
                writer.Write(Rows);
                writer.Write(Cols);
                for (int i = 0; i < Length; i++)
                    writer.Write(_storage[i]);
            }
        }

        public void Apply(Func<float, float> function)
        {
            for (int i = 0; i < Length; i++)
                _storage[i] = function(_storage[i]);
        }

        public void Clamp(float min, float max)
        {
            for (int i = 0; i < Length; i++)
            {
                if (_storage[i] < min)
                    _storage[i] = min;
                else if (_storage[i] > max)
                    _storage[i] = max;
            }
        }

        public float Sum()
        {
            float sum = 0.0f;
            for (int i = 0; i < Length; i++)
                sum += _storage[i];
            return sum;
        }

        public Matrix TileVector(int cols)
        {
            if (Cols > 1)
            {
                throw new InvalidOperationException("Source matrix is not a column vector!");
            }

            var matrix = new Matrix(Rows, cols);

            for (int col = 0; col < cols; col++)
            {
                int offset = col * Rows;
                Array.Copy(_storage, 0, matrix._storage, offset, Rows);
            }

            return matrix;
        }

        public void SetColumn(int colIndex, Matrix col)
        {
            if (col.Rows != Rows)
            {
                throw new InvalidOperationException($"Can not write {col.Rows} elemnts to matrix with {Rows} rows!");
            }

            int offset = colIndex * Rows;
            Array.Copy(col._storage, 0, _storage, offset, Rows);
        }

        [ExcludeFromCodeCoverage]
        public override string ToString() // Function returns matrix as a string
        {
            var s = new StringBuilder();
            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Cols; j++)
                    s.Append($"{this[i, j],8:0.##############}" + " ");
                s.AppendLine();
            }
            return s.ToString();
        }

        /// <summary>
        ///     this = beta*this + alpha*AB;
        /// </summary>
        public void Accumulate(Matrix A, Matrix B, float beta = 0.0f, float alpha = 1.0f, TransposeOptions transposeA = TransposeOptions.NoTranspose, TransposeOptions transposeB = TransposeOptions.NoTranspose)
        {
            if ((B.Cols > 1 && transposeB == TransposeOptions.NoTranspose) || (B.Rows > 1 && transposeB == TransposeOptions.Transpose))
            {
                DotMatrix(A, B, this, beta, alpha, transposeA, transposeB);
            }
            else
            {
                if (A.Cols > 1)
                    DotVec(A, B, this, beta, alpha, transposeA);
                else
                    UpdMatFromVec(A, B, this, alpha);
            }
        }

        /// <summary>
        ///     this = alpha*A+this
        /// </summary>
        public void Accumulate(Matrix A, float alpha = 1.0f)
        {
            if (A.Cols == 1)
            {
                SumVec(A, this, alpha);
            }
            else
            {
                var result = alpha == 1.0d ? A : alpha * A;
                Array.Copy((result + this)._storage, _storage, _storage.Length);
            }
        }


        public bool EqualsTo(Matrix other)
        {
            if (ReferenceEquals(this, other) || ReferenceEquals(_storage, other._storage))
            {
                return true;
            }

            if (_storage.Length != other._storage.Length)
            {
                return false;
            }

            for (int i = 0; i < _storage.Length; i++)
            {
                if (Math.Abs(_storage[i] - other._storage[i]) > 1e-5)
                {
                    return false;
                }
            }

            return true;
        }

        public void CopyToArray(float[] dest, ref int idx)
        {
            if (dest.Length - idx < _storage.Length)
            {
                throw new InvalidOperationException("Not enough space in target array!");
            }

            Array.Copy(_storage, 0, dest, idx, _storage.Length);
            idx += _storage.Length;
        }

        public void CopyFromArray(float[] src, ref int idx)
        {
            if (src.Length - idx < _storage.Length)
            {
                throw new InvalidOperationException("Source array doesn't have enough data!");
            }

            Array.Copy(src, idx, _storage, 0, _storage.Length);
            idx += _storage.Length;
        }

        public void Blt(Matrix matrix, int sRow, int sCol, int dRow, int dCol, int rows, int cols)
        {
            //todo: Need more checks!!
            if (sRow + rows > matrix.Rows || sCol + cols > matrix.Cols)
                throw new InvalidOperationException(
                    $"Can not fit matrix with dimensions {matrix.Rows}x{matrix.Cols} in {Rows}x{Cols} matrix at ({dRow},{dCol})");
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                {
                    this[i + dRow, j + dCol] = matrix[i + sRow, j + sCol];
                }
        }

        [ExcludeFromCodeCoverage]
        public void GetObjectData(SerializationInfo info, StreamingContext context)
        {
            info.AddValue(nameof(Rows), Rows);
            info.AddValue(nameof(Cols), Cols);

            for (int i = 0; i < Length; i++)
            {
                info.AddValue($"_{i}", _storage[i]);
            }
        }

        /// <summary>
        ///     A = alpha*xyT + A
        /// </summary>
        private static unsafe void UpdMatFromVec(Matrix x, Matrix y, Matrix A, float alpha = 1.0f)
        {
            if (x.Rows != y.Cols)
            {
                throw new InvalidOperationException("Vector dimensions do not agree.");
            }

            _backend.ger(x.Rows, y.Rows, alpha, x, 1, y, 1, A, A.Rows);
        }

        /// <summary>
        ///     y = beta*y + alpha*Ax;
        /// </summary>
        private static unsafe void DotVec(Matrix A, Matrix x, Matrix y, float beta, float alpha, TransposeOptions transposeA)
        {
            int aCols = transposeA == TransposeOptions.NoTranspose ? A.Cols : A.Rows;

            if (aCols != x.Rows)
            {
                throw new InvalidOperationException("Matrix and vector dimensions do not agree.");
            }

            _backend.gemv(transposeA, A.Rows, A.Cols, alpha, A._storage, A.Rows, x._storage, 1, beta, y._storage, 1);
        }

        /// <summary>
        ///     C = alpha*AB + beta*C
        /// </summary>
        private static unsafe void DotMatrix(Matrix A, Matrix B, Matrix C, float beta = 0.0f, float alpha = 1.0f, TransposeOptions transposeA = TransposeOptions.NoTranspose, TransposeOptions transponseB = TransposeOptions.NoTranspose)
        {
            int m = C.Rows;
            int n = C.Cols;
            int k = (transposeA == TransposeOptions.NoTranspose) ? A.Cols : A.Rows;
            int bRows = (transponseB == TransposeOptions.NoTranspose) ? B.Rows : B.Cols;

            if (k != bRows)
            {
                throw new InvalidOperationException("Matrix dimensions don't agree.");
            }

            _backend.gemm(transposeA, transponseB, m, n, k, alpha, A, A.Rows, B, B.Rows, beta, C, C.Rows);
        }

        /// <summary>
        ///     y=alpha*x + y;
        /// </summary>
        private static unsafe void SumVec(Matrix x, Matrix y, float alpha)
        {
            if (y.Cols > 1 || x.Cols > 1)
                throw new Exception("Vector BLAS function is called with matrix argument!");

            if (y.Rows != x.Rows)
                throw new Exception("Vector dimensions must agree!");

            _backend.axpy(x.Rows, alpha, x, 1, y, 1);
        }

        private static Matrix Multiply(float n, Matrix m)
        {
            var r = new Matrix(m.Rows, m.Cols);
            for (int i = 0; i < m.Length; i++)
                r._storage[i] = m._storage[i] * n;
            return r;
        }

        private static Matrix Add(Matrix m1, Matrix m2)
        {
            if (m1.Rows != m2.Rows || m1.Cols != m2.Cols) throw new InvalidOperationException("Matrices must have the same dimensions!");

            var r = new Matrix(m1.Rows, m1.Cols);
            for (int i = 0; i < m1.Length; i++)
                r._storage[i] = m1._storage[i] + m2._storage[i];
            return r;
        }

        private static Matrix HadamardMul(Matrix m1, Matrix m2)
        {
            if (m1.Rows != m2.Rows || m1.Cols != m2.Cols) throw new InvalidOperationException("Matrices must have the same dimensions!");
            var r = new Matrix(m1.Rows, m1.Cols);
            for (int i = 0; i < m1.Length; i++)
                r._storage[i] = m1._storage[i] * m2._storage[i];
            return r;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void CheckBounds(int row, int col)
        {
            if (row < 0 || row >= Rows)
            {
                throw new InvalidOperationException("Matrix row is out of range.");
            }

            if (col < 0 || col >= Cols)
            {
                throw new InvalidOperationException("Matrix column is out of range.");
            }
        }
    }
}