using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using MathNet.Numerics.LinearAlgebra.Single;
using Retia.Integration;
using Retia.Mathematics;

namespace Retia.Training.Data
{
	public class SequentialDataSet : IDataSet, IStreamWritable
	{
	    private readonly List<Matrix> _matrices;
	    private int _curIdx = 0;

	    public SequentialDataSet(List<Matrix> matrices)
		{
			if (matrices == null) throw new ArgumentNullException(nameof(matrices));
			//Todo: Temporary disabled to allow empty test sets
            //if (matrices.Count == 0) throw new InvalidOperationException("The are no matrices!");

			_matrices = matrices;
		}

	    private SequentialDataSet(SequentialDataSet other)
		{
			_matrices = new List<Matrix>(other._matrices);
			_curIdx = other._curIdx;
		}

	    public int Size => _matrices[0].RowCount;
	    public int InputSize => Size;
	    public int TargetSize => Size;
	    public int BatchSize => _matrices[0].ColumnCount;

	    public int SampleCount => _matrices.Count - 1;

	    public static SequentialDataSet Load(Stream stream)
		{
			using (var reader = new BinaryReader(stream, Encoding.UTF8, true))
			{
				int count = reader.ReadInt32();
				var list = new List<Matrix>(count);

				for (int i = 0; i < count; i++)
				{
					list.Add(MatrixFactory.Load(stream));
				}

				return new SequentialDataSet(list);
			}
		}

	    public Sample GetNextSample()
		{
			if (_curIdx == _matrices.Count - 1)
			{
				_curIdx = 0;
                OnDataSetReset();
			    return null;
			}

			var result = new Sample(_matrices[_curIdx], _matrices[_curIdx + 1]);
			_curIdx++;

			return result;
		}

	    public void Reset()
		{
            _curIdx = 0;
		}

	    public IDataSet Clone()
		{
			return new SequentialDataSet(this);
		}

	    public void Save(Stream stream)
		{
			using (var writer = new BinaryWriter(stream, Encoding.UTF8, true))
			{
				writer.Write(_matrices.Count);
				foreach (var matrix in _matrices)
				{
					MatrixFactory.Save(matrix, stream);
				}
			}
		}

	    public TrainingSequence GetNextSamples(int count)
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

            var inputs = new List<Matrix>(count);
            var targets = new List<Matrix>(count);
	        for (int i = 0; i < count; i++)
	        {
	            inputs.Add(_matrices[_curIdx + i]);
                targets.Add(_matrices[_curIdx + i + 1]);
	        }

	        _curIdx += count;

	        return new TrainingSequence(inputs, targets);
	    }

	    public event EventHandler DataSetReset;

	    protected virtual void OnDataSetReset()
	    {
	        DataSetReset?.Invoke(this, EventArgs.Empty);
	    }
	}
}