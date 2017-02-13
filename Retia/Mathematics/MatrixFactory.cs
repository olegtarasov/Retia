using System;
using System.Globalization;
using System.IO;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using Retia.Helpers;
using Retia.RandomGenerator;

namespace Retia.Mathematics
{
    public static class MatrixFactory
    {
        public static void Save<T>(this Matrix<T> matrix, Stream stream) where T : struct, IEquatable<T>, IFormattable
        {
            if (typeof(T) == typeof(float))
            {
                Save(matrix as Matrix<float>, stream);
            }
            else
            {
                Save(matrix as Matrix<double>, stream);
            }
        }

        public static void Save(this Matrix<float> matrix, Stream stream) 
        {
            using (var writer = stream.NonGreedyWriter())
            {
                writer.Write(matrix.RowCount);
                writer.Write(matrix.ColumnCount);

                var arr = matrix.AsColumnMajorArray();
                for (int i = 0; i < arr.Length; i++)
                {
                    writer.Write(arr[i]);
                }
            }
        }

        public static void Save(this Matrix<double> matrix, Stream stream)
        {
            using (var writer = stream.NonGreedyWriter())
            {
                writer.Write(matrix.RowCount);
                writer.Write(matrix.ColumnCount);

                var arr = matrix.AsColumnMajorArray();
                for (int i = 0; i < arr.Length; i++)
                {
                    writer.Write(arr[i]);
                }
            }
        }

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


        public static Matrix<float> LoadS(Stream stream)
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

        public static Matrix<double> LoadD(Stream stream)
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

        public static Matrix<T> ParseString<T>(string input) where T : struct, IEquatable<T>, IFormattable
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
                    matrix[row, col] = (T)Convert.ChangeType(double.Parse(cols[col], NumberFormatInfo.InvariantInfo), typeof(T)); // Bubble up the format error
                }
            }

            return matrix;
        }

        public static Matrix<T> RandomMatrix<T>(int rows, int cols, float min, float max) where T : struct, IEquatable<T>, IFormattable
        {
            return MathProvider<T>.Instance.RandomMatrix(rows, cols, min, max);
        }

        public static Matrix<T> RandomMatrix<T>(int rows, int cols, float dispersion) where T : struct, IEquatable<T>, IFormattable
        {
            return RandomMatrix<T>(rows, cols, -dispersion, dispersion);
        }

        public static Matrix<T> RandomMaskMatrix<T>(int rows, int cols, float trueProb) where T : struct, IEquatable<T>, IFormattable
        {
            if (typeof(T) == typeof(float))
            {
                return RandomMaskMatrixS(rows, cols, trueProb) as Matrix<T>;
            }
            else
            {
                return RandomMaskMatrixD(rows, cols, trueProb) as Matrix<T>;
            }
        }

        public static Matrix<float> RandomMaskMatrixS(int rows, int cols, float trueProb)
        {
            var random = SafeRandom.Generator;
            var arr = new float[rows * cols];
            for (int i = 0; i < arr.Length; i++)
                arr[i] = random.NextDouble() < trueProb ? 1.0f : 0.0f;
            return Matrix<float>.Build.Dense(rows, cols, arr);
        }

        public static Matrix<double> RandomMaskMatrixD(int rows, int cols, float trueProb)
        {
            var random = SafeRandom.Generator;
            var arr = new double[rows * cols];
            for (int i = 0; i < arr.Length; i++)
                arr[i] = random.NextDouble() < trueProb ? 1.0d : 0.0d;
            return Matrix<double>.Build.Dense(rows, cols, arr);
        }
    }
}