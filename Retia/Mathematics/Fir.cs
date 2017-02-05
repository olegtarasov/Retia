using System.Collections.Generic;

namespace Retia.Mathematics
{
	public class Fir
	{
		private LinkedList<double> _buffer;
		private LinkedList<double> _coefficients;
		private int _iterations;

		public double FilteredValue { get; private set; }
		public bool FirstTurnComplete => _iterations >= _buffer.Count;

		public Fir() //Default constructor: moving average, N=4
		{
			InitAvg(4);
		}

		public Fir(int order, double initialValue = 0.0d) //Constructor: moving average
		{
			InitAvg(order, initialValue);
		}


		public Fir(double[] coefficients, double initialBufferValue = 0.0d) //Constructor: Fir
		{
			_buffer = new LinkedList<double>();
			_coefficients = new LinkedList<double>();
			for (int i = 0; i < coefficients.Length; i++)
			{
				_coefficients.AddLast(coefficients[i]);
				_buffer.AddLast(initialBufferValue);
			}

			FilteredValue = initialBufferValue;
		}

		private Fir(Fir other)
		{
			_buffer = new LinkedList<double>(other._buffer);
			_coefficients = new LinkedList<double>(other._coefficients);
			FilteredValue = other.FilteredValue;
		}

		public Fir Clone()
		{
			return new Fir(this);
		}
		
		private void InitAvg(int order, double initialValue = 0.0d)
		{
			_iterations = 0;
			_buffer = new LinkedList<double>();
			_coefficients = new LinkedList<double>();
			double coefficient = 1.0d / (order);
			for (int i = 0; i < order; i++)
			{
				_coefficients.AddLast(coefficient);
				_buffer.AddLast(initialValue);
			}

			FilteredValue = initialValue;
		}

		public double FilterSample(double sample)
		{
			double output = 0;
			_buffer.AddFirst(sample);
			_buffer.RemoveLast();

			using (var bufEnum = _buffer.GetEnumerator())
			{
				foreach (var coefficient in _coefficients)
				{
					bufEnum.MoveNext();
					output += coefficient * bufEnum.Current;
				}
			}

			FilteredValue = output;

			if (_iterations < _buffer.Count)
			{
				_iterations++;
			}

			return output;
		}
	}
}