using System;
using System.Globalization;
using System.IO;
using MathNet.Numerics.LinearAlgebra;
using Retia.Helpers;
using Retia.Integration.Helpers;

namespace Retia.Mathematics
{
    /// <summary>
    /// Helpers for matrix creation, saving and loading.
    /// </summary>
    public static class MatrixFactory
    {
        /// <summary>
        /// Create a new column-major matrix with specified data. The resulting matrix will be decoupled
        /// from the <see cref="data"/> array.
        /// </summary>
        /// <param name="rows">Number of rows.</param>
        /// <param name="columns">Number of columns.</param>
        /// <param name="data">Data to create matrix from. Matrix will receive a <b>copy</b> of this data.</param>
        public static Matrix<T> Create<T>(int rows, int columns, params float[] data) where T : struct, IEquatable<T>, IFormattable
        {
            if (rows * columns != data.Length)
                throw new InvalidOperationException("Number of rows and columns doesn't equal to data length.");

            var array = MathProvider<T>.Instance.Array(data);
            return Matrix<T>.Build.Dense(rows, columns, array);
        }

        /// <summary>
        /// Creates a new column-major matrix.
        /// </summary>
        /// <param name="rows">Number of rows.</param>
        /// <param name="columns">Number of columns.</param>
        public static Matrix<T> Create<T>(int rows, int columns) where T : struct, IEquatable<T>, IFormattable
        {
            return Matrix<T>.Build.Dense(rows, columns);
        }

        /// <summary>
        /// Loads matrix from the stream.
        /// </summary>
        /// <param name="stream">The stream.</param>
        public static Matrix<T> Load<T>(Stream stream) where T : struct, IEquatable<T>, IFormattable
        {
            if (typeof(T) == typeof(float))
            {
                return LoadS(stream) as Matrix<T>;
            }
            else
            {
                return LoadD(stream) as Matrix<T>;
            }
        }

        /// <summary>
        /// Parses a matrix from a string.
        /// </summary>
        /// <param name="input">String to parse.</param>
        public static Matrix<T> ParseString<T>(string input) where T : struct, IEquatable<T>, IFormattable
        {
            var mathProvider = MathProvider<T>.Instance;

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

            var matrix = Matrix<T>.Build.Dense(rows.Length, cols.Length);
            for (int row = 0; row < rows.Length; row++)
            {
                cols = rows[row].Trim().Split(new[] { '\t', ' ' }, StringSplitOptions.RemoveEmptyEntries);

                if (cols.Length != numCols)
                {
                    throw new InvalidOperationException($"Not enough columns in row {row}!");
                }

                for (int col = 0; col < cols.Length; col++)
                {
                    matrix[row, col] = mathProvider.Scalar(double.Parse(cols[col], NumberFormatInfo.InvariantInfo)); // Bubble up the format error
                }
            }

            return matrix;
        }

        /// <summary>
        /// Creates a random mask matrix. For each element roll the dice in range of [0..1) and
        /// if the value is less than <see cref="trueProb"/> set the element to 1. Otherwise set
        /// to 0.
        /// </summary>
        /// <param name="rows">Number of rows.</param>
        /// <param name="cols">Number of columns.</param>
        /// <param name="trueProb">Probability of setting an element to 1, range [0..1].</param>
        public static Matrix<T> RandomMaskMatrix<T>(int rows, int cols, float trueProb) where T : struct, IEquatable<T>, IFormattable
        {
            return MathProvider<T>.Instance.RandomMaskMatrix(rows, cols, trueProb);
        }

        /// <summary>
        /// Creates a random matrix in the range of [min;max).
        /// </summary>
        /// <param name="rows">Number of rows.</param>
        /// <param name="cols">Number of columns.</param>
        /// <param name="min">Minimum value.</param>
        /// <param name="max">Maximum value.</param>
        public static Matrix<T> RandomMatrix<T>(int rows, int cols, float min, float max) where T : struct, IEquatable<T>, IFormattable
        {
            return MathProvider<T>.Instance.RandomMatrix(rows, cols, min, max);
        }

        /// <summary>
        /// Creates a random matrix in the range of [-dispersion;dispersion).
        /// </summary>
        /// <param name="rows">Number of rows.</param>
        /// <param name="cols">Number of columns.</param>
        /// <param name="dispersion">Dispersion.</param>
        public static Matrix<T> RandomMatrix<T>(int rows, int cols, float dispersion) where T : struct, IEquatable<T>, IFormattable
        {
            return RandomMatrix<T>(rows, cols, -dispersion, dispersion);
        }

        /// <summary>
        /// Saves a matrix to a stream.
        /// </summary>
        /// <param name="matrix">Matrix to save.</param>
        /// <param name="stream">Stream.</param>
        public static void Save<T>(this Matrix<T> matrix, Stream stream) where T : struct, IEquatable<T>, IFormattable
        {
            MathProvider<T>.Instance.SaveMatrix(matrix, stream);
        }

        private static Matrix<double> LoadD(Stream stream)
        {
            using (var reader = stream.NonGreedyReader())
            {
                int rows = reader.ReadInt32();
                int cols = reader.ReadInt32();

                var arr = new double[rows * cols];
                for (int i = 0; i < arr.Length; i++)
                {
                    arr[i] = reader.ReadDouble();
                }

                return Matrix<double>.Build.Dense(rows, cols, arr);
            }
        }


        private static Matrix<float> LoadS(Stream stream)
        {
            using (var reader = stream.NonGreedyReader())
            {
                int rows = reader.ReadInt32();
                int cols = reader.ReadInt32();

                var arr = new float[rows * cols];
                for (int i = 0; i < arr.Length; i++)
                {
                    arr[i] = reader.ReadSingle();
                }

                return Matrix<float>.Build.Dense(rows, cols, arr);
            }
        }
    }
}