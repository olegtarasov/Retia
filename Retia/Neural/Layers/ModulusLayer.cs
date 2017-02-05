using System;
using System.Collections.Generic;
using System.IO;
using MathNet.Numerics.LinearAlgebra.Single;
using Retia.Contracts;
using Retia.Helpers;
using Retia.Mathematics;
using Retia.Optimizers;

namespace Retia.Neural.Layers
{
    public class ModulusLayer : NeuroLayer
    {
        private readonly int _size;

        public ModulusLayer(int size)
        {
            _size = size;
        }

        public ModulusLayer(ModulusLayer other) : base(other)
        {
            _size = other._size;
        }

        public ModulusLayer(BinaryReader reader)
        {
            _size = reader.ReadInt32();
        }

        public override int InputSize => _size;
        public override int OutputSize => _size;
        public override int TotalParamCount => 0;

        public static Matrix ModulusNorm(Matrix input)
        {
            var p = input.CloneMatrix();
            var rawP = p.AsColumnMajorArray();
            int cols = input.ColumnCount;
            int size = rawP.Length;
            var sums = new float[cols];
            
            for (int i = 0; i < size; i++)
            {
                rawP[i] = Math.Abs(rawP[i]);
                var c = i % cols;
                sums[c] += rawP[i];
            }

            for (int i = 0; i < size; i++)
            {
                var c = i % cols;
                rawP[i] /= sums[c];
            }
            return p;
        }

        public static float[] ModulusNorm(float[] input)
        {
            float sum = 0.0f;
            float[] result = new float[input.Length];

            Array.Copy(input, result, input.Length);
            
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = Math.Abs(result[i]);
                sum += result[i];
            }

            for (int i = 0; i < result.Length; i++)
            {
                result[i] /= sum;
            }

            return result;
        }

        public override NeuroLayer Clone()
        {
            return new ModulusLayer(this);
        }

        public override void Save(Stream s)
        {
            using (var writer = s.NonGreedyWriter())
            {
                writer.Write(_size);
            }
        }

        public override void Optimize(OptimizerBase optimizer)
        {
        }

        public override Matrix Step(Matrix input, bool inTraining = false)
        {
            var output = ModulusNorm(input);
            if (inTraining)
            {
                Inputs.Add(input);
                Outputs.Add(output);
            }

            return output;
        }

        public override void ResetMemory()
        {
        }

        public override void ResetOptimizer()
        {
        }

        public override void InitBackPropagation()
        {
            Inputs.Clear();
            Outputs.Clear();
        }

        public override void ClampGrads(float limit)
        {
        }

        public override void ToVectorState(float[] destination, ref int idx, bool grad = false)
        {
        }

        public override void FromVectorState(float[] vector, ref int idx)
        {
        }

        public override List<Matrix> ErrorPropagate(List<Matrix> targets)
        {
            throw new NotSupportedException("This layer can't be used as last layer.");
        }

        protected override float Derivative(Matrix input, Matrix output, int batch, int i, int o)
        {
            var sum = Math.Abs(input[o, batch])/output[o, batch];
            //just take denominator from already calculated output to reduce calculations
            var sum2 = sum*sum;

            var sign = input[i, batch]/Math.Abs(input[i, batch]);

            if (i != o)
                return -sign*output[o,batch]/sum;
            return sign*(sum - Math.Abs(input[i, batch]))/sum2;
        }

        public override List<Matrix> BackPropagate(List<Matrix> outSens, bool needInputSens = true)
        {
            return PropagateSensitivity(outSens);
        }

        public override LayerSpecBase CreateSpec()
        {
            throw new NotSupportedException();
        }
    }
}