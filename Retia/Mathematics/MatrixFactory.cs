using System;
using System.Globalization;
using System.IO;
using MathNet.Numerics.LinearAlgebra.Single;
using Retia.Helpers;
using Retia.RandomGenerator;

namespace Retia.Mathematics
{
    public static class MatrixFactory
    {
        public static void Save(this Matrix matrix, Stream stream)
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

        public static Matrix Load(Stream stream)
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

                return DenseMatrix.OfColumnMajor(rows, cols, arr);
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

            var matrix = new DenseMatrix(rows.Length, cols.Length);
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
            var arr = new float[rows * cols];
            for (int i = 0; i < arr.Length; i++)
                arr[i] = (float)random.NextDouble(min, max);
            return new DenseMatrix(rows, cols, arr);
        }

        public static Matrix RandomMatrix(int rows, int cols, float dispersion)
        {
            return RandomMatrix(rows, cols, -dispersion, dispersion);
        }

        public static Matrix RandomMaskMatrix(int rows, int cols, float trueProb)
        {
            var random = SafeRandom.Generator;
            var arr = new float[rows * cols];
            for (int i = 0; i < arr.Length; i++)
                arr[i] = random.NextDouble() < trueProb ? 1.0f : 0.0f;
            return new DenseMatrix(rows, cols, arr);
        }
    }
}