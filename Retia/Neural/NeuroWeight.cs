using System.IO;
using System.Text;
using MathNet.Numerics.LinearAlgebra.Single;
using Retia.Integration;
using Retia.Mathematics;

namespace Retia.Neural
{
	public class NeuroWeight : ICloneable<NeuroWeight>, IStreamWritable
	{
        /// <summary>
        /// Weight matrix
        /// </summary>
		public Matrix Weight { get; private set; }

        /// <summary>
        /// Current gradient matrix of Weight matrix
        /// </summary>
		public Matrix Gradient { get; private set; }

        /// <summary>
        /// Grad1 avg cache
        /// </summary>
		public Matrix Cache1 { get; private set; }

        /// <summary>
        /// Grad2 avg cache
        /// </summary>
        public Matrix Cache2 { get; private set; }

        /// <summary>
        /// Momentum cache
        /// </summary>
        public Matrix CacheM { get; private set; }

        public NeuroWeight()
		{
		}

		public NeuroWeight(Matrix weight) : this()
		{
			Weight = weight.CloneMatrix();
			Gradient = new DenseMatrix(weight.RowCount, weight.ColumnCount);
            Cache1 = new DenseMatrix(weight.RowCount, weight.ColumnCount);
            Cache2 = new DenseMatrix(weight.RowCount, weight.ColumnCount);
            CacheM = new DenseMatrix(weight.RowCount, weight.ColumnCount);
        }

		private NeuroWeight(NeuroWeight other)
		{
		    Weight = other.Weight.CloneMatrix();
			Gradient = other.Gradient.CloneMatrix();
			Cache1 = other.Cache1.CloneMatrix();
            Cache2 = other.Cache2.CloneMatrix();
            CacheM = other.CacheM.CloneMatrix();
        }

		public static NeuroWeight Load(Stream stream)
		{
			var result = new NeuroWeight();
			using (var reader = new BinaryReader(stream, Encoding.UTF8, true))
			{
                result.Weight = MatrixFactory.Load(stream);
				bool saveCache = reader.ReadBoolean();
				bool saveGrad = reader.ReadBoolean();

			    if (saveCache)
			    {
			        result.Cache1 = MatrixFactory.Load(stream);
                    result.Cache2 = MatrixFactory.Load(stream);
                    result.CacheM = MatrixFactory.Load(stream);
                }
			    else
			    {
                    result.Cache1 = new DenseMatrix(result.Weight.RowCount, result.Weight.ColumnCount);
                    result.Cache2 = new DenseMatrix(result.Weight.RowCount, result.Weight.ColumnCount);
                    result.CacheM = new DenseMatrix(result.Weight.RowCount, result.Weight.ColumnCount);
                }
			    if (saveGrad)
					result.Gradient = MatrixFactory.Load(stream);
			    else
                    result.Gradient = new DenseMatrix(result.Weight.RowCount, result.Weight.ColumnCount);
            }

			return result;
		}

		public NeuroWeight Clone()
		{
			return new NeuroWeight(this);
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
        }

		public static implicit operator NeuroWeight(Matrix weight)
		{
			return new NeuroWeight(weight);
		}
	}
}