//using System;
//using System.Collections.Generic;
//using System.IO;
//using System.Linq;
//using System.Text;
//using System.Threading.Tasks;
//using MathNet.Numerics.LinearAlgebra.Single;
//using Retia.Contracts;
//using Retia.Helpers;
//using Retia.Mathematics;
//using Retia.Optimizers;

//namespace Retia.Neural.Layers
//{
//    public class TanhLayer : NeuroLayer
//    {
//        private readonly int _size;

//        private TanhLayer(TanhLayer other) : base(other)
//        {
//            _size = other._size;
//        }

//        public TanhLayer(int size)
//        {
//            _size = size;
//        }

//        public TanhLayer(BinaryReader reader)
//        {
//            _size = reader.ReadInt32();
//        }

//        public override int InputSize => _size;
//        public override int OutputSize => _size;
//        public override int TotalParamCount => 0;
//        public override void Save(Stream s)
//        {
//            using (var writer = s.NonGreedyWriter())
//            {
//                writer.Write(_size);
//            }
//        }

//        public override NeuroLayer Clone()
//        {
//            return new TanhLayer(_size);
//        }

//        public override void Optimize(OptimizerBase optimizer)
//        {
//        }

//        private double sigmoid(double x)
//        {
//            return 1.0/(1.0+Math.Exp(-x));
//        }

//        public override Matrix Step(Matrix input, bool inTraining = false)
//        {
//            var output = input.CloneMatrix();

//            ActivationFuncs.ApplyTanh(output);
            
//            if (inTraining)
//            {
//                Outputs.Add(output);
//            }

//            return output;
//        }

//        public override List<Matrix> BackPropagate(List<Matrix> outSens, bool needInputSens = true)
//        {
//            if (Outputs.Count == 0)
//                throw new Exception("Empty inputs history, nothing to propagate!");
//            if (outSens.Count != Outputs.Count)
//                throw new Exception("Not enough sensitivies in list!");

//            var iSens = new List<Matrix>();
//            for (int t = 0; t < outSens.Count; t++)
//            {
//                var output = Outputs[t];
//                var osens = outSens[t];
//                var fder = //output ^ (new Matrix(osens.Rows, osens.Cols, 1.0) - output);
//                    DenseMatrix.Create(osens.RowCount, osens.ColumnCount, DenseMatrix.One) - output.PointwisePower(2);
//                var isens = (Matrix)fder.PointwiseMultiply(osens);
//                iSens.Add(isens);
//            }
//            return iSens;
//        }

//        public override void ResetMemory()
//        {
//        }

//        public override void ResetOptimizer()
//        {
//        }

//        public override void InitSequence()
//        {
//            Outputs.Clear();
//        }

//        public override void ClampGrads(float limit)
//        {
//        }

//        public override void ToVectorState(float[] destination, ref int idx, bool grad = false)
//        {
//        }

//        public override void FromVectorState(float[] vector, ref int idx)
//        {
//        }

//        public override LayerSpecBase CreateSpec()
//        {
//            throw new NotSupportedException();
//        }
//    }
//}
