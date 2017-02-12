using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using Retia.Mathematics;

namespace Retia.Training.Data
{
    public class TrainingSequence<T> where T : struct, IEquatable<T>, IFormattable
    {
        public TrainingSequence(List<Matrix<T>> inputs, List<Matrix<T>> targets)
        {
            Inputs = inputs;
            Targets = targets;
        }

        public List<Matrix<T>> Inputs { get; set; }
        public List<Matrix<T>> Targets { get; set; }
    }
}