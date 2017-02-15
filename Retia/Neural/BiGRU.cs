using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using Retia.Mathematics;
using Retia.Neural.Layers;
using Retia.Optimizers;

namespace Retia.Neural
{
    public class BiGRU<T> : NeuralNet<T> where T : struct, IEquatable<T>, IFormattable
    {
        private NeuralNet<T> forward;
        private NeuralNet<T> backward;
        private NeuralNet<T> outNet;
        public BiGRU(int xSize, int hSize, int ySize, int batchSize, int seqLen)
        {
            forward = new LayeredNet<T>(batchSize, seqLen, new GruLayer<T>(xSize, hSize));
            backward = new LayeredNet<T>(batchSize, seqLen, new GruLayer<T>(xSize, hSize));
            outNet = new LayeredNet<T>(batchSize, seqLen, new GruLayer<T>(hSize * 2, hSize), new LinearLayer<T>(hSize, ySize));
        }

        public BiGRU(BiGRU<T> other)
        {
            forward = other.forward.Clone();
            backward = other.backward.Clone();
            outNet = other.outNet.Clone();
        }

        public override void Save(Stream s)
        {
            forward.Save(s);
            backward.Save(s);
            outNet.Save(s);
        }

        public override NeuralNet<T> Clone()
        {
            return new BiGRU<T>(this);
        }

        public override void Optimize()
        {
            forward.Optimize();
            backward.Optimize();
            outNet.Optimize();
        }

        public override List<Matrix<T>> BackPropagate(List<Matrix<T>> targets, bool needInputSense = false)
        {
            var prop = outNet.BackPropagate(targets, true);

            var batchSize = targets[0].ColumnCount;

            var fSens = new List<Matrix<T>>(targets.Count);
            var bSens = new List<Matrix<T>>(targets.Count);
            for (int i = 0; i < prop.Count; i++)
            {
                var sens = prop[i];

                var f = Matrix<T>.Build.Dense(forward.OutputSize, batchSize);
                f.SetSubMatrix(0, 0, f.RowCount, 0, 0, f.ColumnCount, sens);
                fSens.Add(f);

                var b = Matrix<T>.Build.Dense(backward.OutputSize, batchSize);
                b.SetSubMatrix(0, f.RowCount, b.RowCount, 0, 0, b.ColumnCount, sens);
                bSens.Add(b);
            }
            bSens.Reverse();
            forward.BackPropagate(fSens);
            backward.BackPropagate(bSens);

            //todo: not very good decision if we want to cascasde them
            return null;
        }

        public override double Error(Matrix<T> y, Matrix<T> target)
        {
            return outNet.Error(y, target);
        }

        public override Matrix<T> Step(Matrix<T> input, bool inTraining = false)
        {
            throw new NotSupportedException("Can not make single step on BiGRU");
        }

        public override void ResetMemory()
        {
            forward.ResetMemory();
            backward.ResetMemory();
            outNet.ResetMemory();
        }

        public override void ResetOptimizer()
        {
            forward.ResetOptimizer();
            backward.ResetOptimizer();
            outNet.ResetOptimizer();
        }

        public override void InitSequence()
        {
            forward.InitSequence();
            backward.InitSequence();
            outNet.InitSequence();
        }

        public override OptimizerBase<T> Optimizer
        {
            get { return outNet.Optimizer; }
            set
            {
                forward.Optimizer = value;
                backward.Optimizer = value;
                outNet.Optimizer = value;
            }
        }

        public override int InputSize => forward.InputSize;
        public override int OutputSize => outNet.OutputSize;

        public override int TotalParamCount
            => forward.TotalParamCount + backward.TotalParamCount + outNet.TotalParamCount;

        public override List<Matrix<T>> ProcessSequence(List<Matrix<T>> inputs)
        {
            return ProcessSequence(inputs, false);
        }
        public List<Matrix<T>> ProcessSequence(List<Matrix<T>> inputs, bool inTraining)
        {
            var yList = new List<Matrix<T>>(inputs.Count);

            var combinedList = new List<Matrix<T>>(inputs.Count);
            var batchSize = inputs[0].ColumnCount;

            for (int i = 0; i < inputs.Count; i++)
                combinedList.Add(Matrix<T>.Build.Dense(forward.OutputSize * 2, batchSize));

            for (int i = 0; i < inputs.Count; i++)
            {
                var bI = inputs.Count - i - 1;

                var f = forward.Step(inputs[i], inTraining);
                var b = backward.Step(inputs[bI], inTraining);

                combinedList[i].SetSubMatrix(0, 0, f.RowCount, 0, 0, f.ColumnCount, f);
                combinedList[bI].SetSubMatrix(f.RowCount, 0, b.RowCount, 0, 0, b.ColumnCount, b);
            }

            foreach (var combined in combinedList)
            {
                var y = outNet.Step(combined, inTraining);
                yList.Add(y);
            }

            return yList;
        }

        public override double TrainSequence(List<Matrix<T>> inputs, List<Matrix<T>> targets)
        {
            if (inputs.Count != targets.Count || targets.Count == 0)
                throw new Exception("Not enough targets or inputs provided!");

            var sequenceLen = inputs.Count;
            InitSequence();
            //var yList = new List<Matrix>(sequenceLen);
            var error = new List<double>(sequenceLen);
            //var sw=new Stopwatch();
            //sw.Restart();
            var outs = ProcessSequence(inputs, true);
            //sw.Stop();
            //Console.WriteLine($"Elapsed {sw.Elapsed.TotalSeconds:f3}");
            for (int i = 0; i < outs.Count; i++)
            {
                error.Add(Error(outs[i], targets[i]));
            }
            BackPropagate(targets);
            var totalErr = error.Sum() / error.Count;
            return totalErr;
        }

        public override List<Matrix<T>> TestSequence(List<Matrix<T>> inputs, List<Matrix<T>> targets, out List<double> errors)
        {
            var yList = new List<Matrix<T>>(inputs.Count);
            errors = new List<double>(inputs.Count);

            var combinedList = new List<Matrix<T>>(inputs.Count);
            var batchSize = inputs[0].ColumnCount;

            for (int i = 0; i < inputs.Count; i++)
                combinedList.Add(Matrix<T>.Build.Dense(forward.OutputSize * 2, batchSize));

            for (int i = 0; i < inputs.Count; i++)
            {
                var bI = inputs.Count - i - 1;

                var f = forward.Step(inputs[i]);
                var b = backward.Step(inputs[bI]);

                combinedList[i].SetSubMatrix(0, 0, f.RowCount, 0, 0, f.ColumnCount, f);
                combinedList[bI].SetSubMatrix(f.RowCount, 0, b.RowCount, 0, 0, b.ColumnCount, b);
            }

            for (int i = 0; i < combinedList.Count; i++)
            {
                var combined = combinedList[i];
                var target = targets[i];

                var y = outNet.Step(combined);
                var e = Error(y, target);
                errors.Add(e);
                yList.Add(y);
            }

            return yList;
        }
    }
}