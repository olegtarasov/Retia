using System;
using System.Collections.Generic;

namespace Retia.Helpers
{
	public struct MinMax<T>
	{
		public T Min, Max;
	}

	public static class MathHelpers
	{
        /// <summary>
        /// Element-wize array subtraction.
        /// </summary>
        public static double[] Subtract(this double[] from, double[] subtractor)
        {
            if (from == null) throw new ArgumentNullException(nameof(from));
            if (subtractor == null) throw new ArgumentNullException(nameof(subtractor));
            if (from.Length != subtractor.Length) throw new InvalidOperationException("Arrays are of different sizes!");

            var result = new double[from.Length];
            for (int i = 0; i < from.Length; i++)
            {
                result[i] = from[i] - subtractor[i];
            }

            return result;
        }

        /// <summary>
        /// Element-wize array multiplication with a constant
        /// </summary>
        public static double[] Multiply(this double[] array, double multiplier)
        {
            if (array == null) throw new ArgumentNullException(nameof(array));
            
            var result = new double[array.Length];
            for (int i = 0; i < array.Length; i++)
            {
                result[i] = array[i] * multiplier;
            }

            return result;
        }

        public static List<double> Normalize(this IReadOnlyList<double> source)
		{
			var result = new List<double>(source.Count);
			double total = 0.0d;

			for (int i = 0; i < source.Count; i++)
				total += source[i];

			for (var i = 0; i < source.Count; i++)
			{
				double res = source[i] / total;
				result.Add(double.IsNaN(res) ? 0.0d : res);
			}

			return result;
		}

		public static double Max(params double[] nums)
		{
			double max = double.MinValue;

			for (int i = 0; i < nums.Length; i++)
				if (nums[i] > max)
					max = nums[i];

			return max;
		}

		public static double Min(params double[] nums)
		{
			double min = double.MaxValue;

			for (int i = 0; i < nums.Length; i++)
				if (nums[i] < min)
					min = nums[i];

			return min;
		}

		public static int? Max(params int?[] nums)
		{
			int max = int.MinValue;
			bool notNull = false;

			for (int i = 0; i < nums.Length; i++)
			{
				if (!nums[i].HasValue)
				{
					continue;
				}

				notNull = true;
				if (nums[i].Value > max)
					max = nums[i].Value;
			}

			return notNull ? max : (int?)null;
		}

		public static int? Min(params int?[] nums)
		{
			int min = int.MaxValue;
			bool notNull = false;

			for (int i = 0; i < nums.Length; i++)
			{
				if (!nums[i].HasValue)
				{
					continue;
				}

				notNull = true;
				if (nums[i].Value < min)
					min = nums[i].Value;
			}

			return notNull ? min : (int?)null;
		}

		public static void MinMax(double a, double b, out double min, out double max)
		{
			if (a > b)
			{
				min = b;
				max = a;
			}
			else
			{
				min = a;
				max = b;
			}
		}

		public static MinMax<double> MinMax(params double[] nums)
		{
			var result = new MinMax<double> {Max = double.MinValue, Min = double.MaxValue};

			for (int i = 0; i < nums.Length; i++)
			{
				if (nums[i] > result.Max)
					result.Max = nums[i];
				if (nums[i] < result.Min)
					result.Min = nums[i];
			}

			return result;
		}

		public static bool EqualsEps(this double a, double b, double epsilon = 0.000001f)
		{
			return (Math.Abs(a - b) < epsilon);
		}

		public static int GetDecimalPlaces(float value)
		{
			int result = 0;
			
			while (((int)(value *= 10)) % 10 != 0)
				result++;

			return result;
		}

		public static double RoundedStep(double value, int nums = 1)
		{
			if (value == 0)
				return 0;

			long integral = (long)value;

			if (integral != 0)
				return integral;

			double tmp = value;
			int cnt = 1;
			
			while (((long)(tmp *= 10)) == 0)
				cnt++;

			if (nums > 1)
			{
				cnt += nums - 1;
				tmp *= Math.Pow(10, nums - 1);
			}

			return ((long)tmp) / Math.Pow(10, cnt);
		}

		public static double Round(this double value, int nums = 4)
		{
			return Math.Round(value, nums);
		}

		public static double? Abs(double? value)
		{
			return value != null ? Math.Abs(value.Value) : (double?)null;
		}
	}
}