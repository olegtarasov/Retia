using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra.Double;
using Retia.Mathematics;

namespace Retia.Training.Data
{
    public class TrainingSequence
    {
        public TrainingSequence(List<Matrix> inputs, List<Matrix> targets)
        {
            Inputs = inputs;
            Targets = targets;
        }

        public List<Matrix> Inputs { get; set; }
        public List<Matrix> Targets { get; set; }
    }
}