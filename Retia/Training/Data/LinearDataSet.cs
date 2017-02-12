using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using MathNet.Numerics.LinearAlgebra;
using Retia.Integration;
using Retia.Mathematics;

namespace Retia.Training.Data
{
	public class LinearDataSet<T> : IDataSet<T>, IStreamWritable, ICloneable<LinearDataSet<T>> where T : struct, IEquatable<T>, IFormattable
    {
	    private int _curIdx = 0;

	    public LinearDataSet(List<Sample<T>> samples)
		{
			if (samples == null) throw new ArgumentNullException(nameof(samples));
			if (samples.Count == 0) throw new InvalidOperationException("Samples are empty!");

			Samples = new List<Sample<T>>(samples);
		}

	    private LinearDataSet(LinearDataSet<T> other)
		{
			Samples = new List<Sample<T>>(other.Samples);
		}

	    public List<Sample<T>> Samples { get; }

	    public int SampleCount => Samples.Count;

	    public int InputSize => Samples[0].Input.RowCount;
	    public int TargetSize => Samples[0].Target.RowCount;
	    public int BatchSize => Samples[0].Input.ColumnCount;

	    public static LinearDataSet<T> Load(Stream stream)
		{
			using (var reader = new BinaryReader(stream, Encoding.UTF8, true))
			{
				var samples = new List<Sample<T>>();
				int cnt = reader.ReadInt32();

				for (int i = 0; i < cnt; i++)
					samples.Add(Sample<T>.Load(stream));

				return new LinearDataSet<T>(samples);
			}
		}

	    public Sample<T> GetNextSample()
		{
			if (_curIdx >= Samples.Count)
			{
			    _curIdx = 0;
                OnDataSetReset();
			    return null;
			}

			return Samples[_curIdx++];
		}

	    public void Reset()
		{
		    _curIdx = 0;
		}

	    public void Save(Stream stream)
		{
			using (var writer = new BinaryWriter(stream, Encoding.UTF8, true))
			{
				writer.Write(Samples.Count);
				foreach (var sample in Samples)
				{
					sample.Save(stream);
				}
			}
		}

	    public LinearDataSet<T> Clone()
		{
			return new LinearDataSet<T>(this);
		}

	    public TrainingSequence<T> GetNextSamples(int count)
	    {
	        if (count < Samples.Count)
	        {
	            throw new InvalidOperationException("Requested more samples than DataSet can provide!");
	        }

	        if (_curIdx + count > Samples.Count)
	        {
	            _curIdx = 0;
                OnDataSetReset();
	            return null;
	        }

            var inputs = new List<Matrix<T>>(count);
            var targets = new List<Matrix<T>>(count);
	        for (int i = 0; i < count; i++)
	        {
	            var sample = Samples[_curIdx + i];
                inputs.Add(sample.Input);
                targets.Add(sample.Target);
	        }

	        _curIdx += count;

	        return new TrainingSequence<T>(inputs, targets);
	    }

	    public event EventHandler DataSetReset;

	    protected virtual void OnDataSetReset()
	    {
	        DataSetReset?.Invoke(this, EventArgs.Empty);
	    }

	    IDataSet<T> ICloneable<IDataSet<T>>.Clone()
		{
			return Clone();
		}
	}
}