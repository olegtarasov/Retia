using System;
using System.IO;
using System.Text;
using MathNet.Numerics.LinearAlgebra;
using Retia.Integration;
using Retia.Mathematics;

namespace Retia.Neural
{
	public class NeuroWeight<T> : ICloneable<NeuroWeight<T>>, IStreamWritable where T : struct, IEquatable<T>, IFormattable
    {
        /// <summary>
        /// Weight matrix
        /// </summary>
		public Matrix<T> Weight { get; private set; }

        /// <summary>
        /// Current gradient matrix of Weight matrix
        /// </summary>
		public Matrix<T> Gradient { get; private set; }

        /// <summary>
        /// Grad1 avg cache
        /// </summary>
		public Matrix<T> Cache1 { get; private set; }

        /// <summary>
        /// Grad2 avg cache
        /// </summary>
        public Matrix<T> Cache2 { get; private set; }

        /// <summary>
        /// Momentum cache
        /// </summary>
        public Matrix<T> CacheM { get; private set; }

        /// <summary>
        /// Timestep
        /// </summary>
        public int Timestep { get; set; } = 0;

        public NeuroWeight()
		{
		}

		public NeuroWeight(Matrix<T> weight) : this()
		{
			Weight = weight.CloneMatrix();
			Gradient = Matrix<T>.Build.Dense(weight.RowCount, weight.ColumnCount);
            Cache1 = Matrix<T>.Build.Dense(weight.RowCount, weight.ColumnCount);
            Cache2 = Matrix<T>.Build.Dense(weight.RowCount, weight.ColumnCount);
            CacheM = Matrix<T>.Build.Dense(weight.RowCount, weight.ColumnCount);
		    Timestep = 0;
		}

		private NeuroWeight(NeuroWeight<T> other)
		{
		    Weight = other.Weight.CloneMatrix();
			Gradient = other.Gradient.CloneMatrix();
			Cache1 = other.Cache1.CloneMatrix();
            Cache2 = other.Cache2.CloneMatrix();
            CacheM = other.CacheM.CloneMatrix();
		    Timestep = other.Timestep;
		}

		public static NeuroWeight<T> Load(Stream stream)
		{
			var result = new NeuroWeight<T>();
			using (var reader = new BinaryReader(stream, Encoding.UTF8, true))
			{
                result.Weight = MatrixFactory.Load<T>(stream);
				bool saveCache = reader.ReadBoolean();
				bool saveGrad = reader.ReadBoolean();

			    if (saveCache)
			    {
			        result.Cache1 = MatrixFactory.Load<T>(stream);
                    result.Cache2 = MatrixFactory.Load<T>(stream);
                    result.CacheM = MatrixFactory.Load<T>(stream);
			        result.Timestep = reader.ReadInt32();
			    }
			    else
			    {
                    result.Cache1 = Matrix<T>.Build.Dense(result.Weight.RowCount, result.Weight.ColumnCount);
                    result.Cache2 = Matrix<T>.Build.Dense(result.Weight.RowCount, result.Weight.ColumnCount);
                    result.CacheM = Matrix<T>.Build.Dense(result.Weight.RowCount, result.Weight.ColumnCount);
                }
			    if (saveGrad)
					result.Gradient = MatrixFactory.Load<T>(stream);
			    else
                    result.Gradient = Matrix<T>.Build.Dense(result.Weight.RowCount, result.Weight.ColumnCount);
            }

			return result;
		}

		public NeuroWeight<T> Clone()
		{
			return new NeuroWeight<T>(this);
		}

		public void Save(Stream stream)
		{
			Save(stream, true, false);
		}

		public void Save(Stream s, bool saveCache, bool saveGrad)
		{
			using (var writer = new BinaryWriter(s, Encoding.UTF8, true))
			{
				Weight.Save(s);
				writer.Write(saveCache);
				writer.Write(saveGrad);
			    if (saveCache)
			    {
                    Cache1.Save(s);
                    Cache2.Save(s);
                    CacheM.Save(s);
                    writer.Write(Timestep);
			    }
			    if (saveGrad)
                    Gradient.Save(s);
			}
		}

		public void ClearGrad()
		{
			Gradient.Clear();
		}

		public void ClearCache()
		{
			Cache1.Clear();
            Cache2.Clear();
            CacheM.Clear();
		    Timestep = 0;
		}

		public static implicit operator NeuroWeight<T>(Matrix<T> weight)
		{
			return new NeuroWeight<T>(weight);
		}
	}
}