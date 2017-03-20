using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using Retia.Helpers;
using Retia.Integration;
using Retia.Mathematics;
using Retia.Optimizers;

namespace Retia.Neural
{
    public abstract class NeuralNet<T> : ICloneable<NeuralNet<T>>, INeuralNet where T : struct, IEquatable<T>, IFormattable
    {
        protected MathProviderBase<T> MathProvider = MathProvider<T>.Instance;
        public virtual OptimizerBase<T> Optimizer { get; set; }

		public abstract void Save(Stream s);
		public abstract NeuralNet<T> Clone();

		public abstract void Optimize();

        public abstract List<Matrix<T>> BackPropagate(List<Matrix<T>> targets, bool needInputSense = false);

        public abstract double Error(Matrix<T> y, Matrix<T> target);

        public abstract Matrix<T> Step(Matrix<T> input, bool inTraining = false);
        
		public abstract void ResetMemory();
		public abstract void ResetOptimizer();
		public abstract void InitSequence();

        public abstract int InputSize { get; }
        public abstract int OutputSize { get; }
		public abstract int TotalParamCount { get; }

        public abstract IReadOnlyList<NeuroWeight<T>> Weights { get; }

        protected NeuralNet()
        {
        }

		protected NeuralNet(NeuralNet<T> other)
		{
			Optimizer = other.Optimizer.Clone();
		}

		public void Save(string filename)
        {
			this.SaveObject(filename);
        }

        public virtual List<Matrix<T>> ProcessSequence(List<Matrix<T>> inputs)
        {
            var yList = new List<Matrix<T>>(inputs.Count);
            foreach (var input in inputs)
            {
                var y = Step(input);
                yList.Add(y);
            }
            return yList;
        }

        public virtual List<Matrix<T>> TestSequence(List<Matrix<T>> inputs, List<Matrix<T>> targets, out List<double> errors)
        {
            var yList = new List<Matrix<T>>(inputs.Count);
            errors=new List<double>(inputs.Count);

            for (int i = 0; i < inputs.Count; i++)
            {
                var input = inputs[i];
                var target = targets[i];
                var y = Step(input);
                var e = Error(y, target);
                errors.Add(e);
                yList.Add(y);
            }
            return yList;
        }

	    public virtual double TrainSequence(List<Matrix<T>> inputs, List<Matrix<T>> targets)
	    {
            if (inputs.Count != targets.Count || targets.Count == 0)
                throw new Exception("Not enough targets or inputs provided!");

            var sequenceLen = inputs.Count;
            InitSequence();
            var error = new List<double>(sequenceLen);
            for (int i = 0; i < inputs.Count; i++)
            {
                var target = targets[i];
                var input = inputs[i];
                var y = Step(input, true);
                error.Add(Error(y, target));
            }
            BackPropagate(targets);
            var totalErr = error.Sum() / error.Count;
            return totalErr;
        }
    }
}