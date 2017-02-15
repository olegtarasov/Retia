using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using MathNet.Numerics.LinearAlgebra;
using Retia.Integration;
using Retia.Mathematics;

namespace Retia.Training.Data
{
	public class SequentialDataSet<T> : IDataSet<T>, IStreamWritable where T : struct, IEquatable<T>, IFormattable
    {
	    private readonly List<Matrix<T>> _matrices;
	    private int _curIdx = 0;

	    public SequentialDataSet(List<Matrix<T>> matrices)
		{
			if (matrices == null) throw new ArgumentNullException(nameof(matrices));
			//Todo: Temporary disabled to allow empty test sets
            //if (matrices.Count == 0) throw new InvalidOperationException("The are no matrices!");

			_matrices = matrices;
		}

	    private SequentialDataSet(SequentialDataSet<T> other)
		{
			_matrices = new List<Matrix<T>>(other._matrices);
			_curIdx = other._curIdx;
		}

	    public int Size => _matrices[0].RowCount;
	    public int InputSize => Size;
	    public int TargetSize => Size;
	    public int BatchSize => _matrices[0].ColumnCount;

	    public int SampleCount => _matrices.Count - 1;

	    public static SequentialDataSet<T> Load(Stream stream)
		{
			using (var reader = new BinaryReader(stream, Encoding.UTF8, true))
			{
				int count = reader.ReadInt32();
				var list = new List<Matrix<T>>(count);

				for (int i = 0; i < count; i++)
				{
					list.Add(MatrixFactory.Load<T>(stream));
				}

				return new SequentialDataSet<T>(list);
			}
		}

	    public Sample<T> GetNextSample()
		{
			if (_curIdx == _matrices.Count - 1)
			{
				_curIdx = 0;
                OnDataSetReset();
			    return null;
			}

			var result = new Sample<T>(_matrices[_curIdx], _matrices[_curIdx + 1]);
			_curIdx++;

			return result;
		}

	    public void Reset()
		{
            _curIdx = 0;
		}

	    public IDataSet<T> Clone()
		{
			return new SequentialDataSet<T>(this);
		}

	    public void Save(Stream stream)
		{
			using (var writer = new BinaryWriter(stream, Encoding.UTF8, true))
			{
				writer.Write(_matrices.Count);
				foreach (var matrix in _matrices)
				{
					MatrixFactory.Save<T>(matrix, stream);
				}
			}
		}

	    public TrainingSequence<T> GetNextSamples(int count)
	    {
	        if (count + 1 > _matrices.Count)
	        {
	            throw new InvalidOperationException("Requested more sample than DataSet can provide!");
	        }

	        if (_curIdx + count + 1 > _matrices.Count)
	        {
	            _curIdx = 0;
                OnDataSetReset();
	            return null;
	        }

            var inputs = new List<Matrix<T>>(count);
            var targets = new List<Matrix<T>>(count);
	        for (int i = 0; i < count; i++)
	        {
	            inputs.Add(_matrices[_curIdx + i]);
                targets.Add(_matrices[_curIdx + i + 1]);
	        }

	        _curIdx += count;

	        return new TrainingSequence<T>(inputs, targets);
	    }

	    public event EventHandler DataSetReset;

	    protected virtual void OnDataSetReset()
	    {
	        DataSetReset?.Invoke(this, EventArgs.Empty);
	    }
	}
}