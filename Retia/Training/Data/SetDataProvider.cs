using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using Retia.Mathematics;
using Retia.RandomGenerator;

namespace Retia.Training.Data
{
    public class SetDataProvider<T> : DataProviderBase<T> where T : struct, IEquatable<T>, IFormattable
    {
        public int BatchSize
        {
            get { return batchSize; }
            set { batchSize = value < 1 ? 1 : value; }
        }

        public int SequenceLen { get; set; }

        private readonly List<List<Matrix<T>>> _trainingInputs = new List<List<Matrix<T>>>();
        private readonly List<List<Matrix<T>>> _trainingTargets = new List<List<Matrix<T>>>();
        private readonly List<List<Matrix<T>>> _testInputs = new List<List<Matrix<T>>>();
        private readonly List<List<Matrix<T>>> _testTargets = new List<List<Matrix<T>>>();
        private int batchSize;
        

        public SetDataProvider(IEnumerable<List<Matrix<T>>> trainingInputs, IEnumerable<List<Matrix<T>>> trainingOutputs,  IEnumerable<List<Matrix<T>>> testInputs = null, IEnumerable<List<Matrix<T>>> testOutputs = null)
        {
            _trainingInputs.AddRange(trainingInputs);
            _trainingTargets.AddRange(trainingOutputs);

            if (testInputs != null && testOutputs != null)
            {
                _testInputs.AddRange(testInputs);
                _testTargets.AddRange(testOutputs);
            }
            BatchSize = 1;
        }
        public static List<int> GetUniqueIndexes(int inputLen, int bsize)
        {
            var indexes = Enumerable.Range(0, inputLen).ToList();
            var rnd = SafeRandom.Generator;

            for (int i = 0; i < inputLen - bsize; i++)
                indexes.RemoveAt(rnd.Next(indexes.Count));
            return indexes;
        }
        private static List<Matrix<T>> GenerateBatch(List<List<Matrix<T>>> sequences, int maxSequenceLen)
        {
            var ioCount = sequences[0][0].RowCount;
            var result = new List<Matrix<T>>();
            for (int i = 0; i < maxSequenceLen; i++)
            {
                var batchedMatrix = Matrix<T>.Build.Dense(ioCount, sequences.Count);
                for (int b = 0; b < sequences.Count; b++)
                {
                    var sequence = sequences[b];
                    for (int j = 0; j < ioCount; j++)
                        batchedMatrix[j, b] = i < sequence.Count ? sequence[i][j, 0] : MathProvider.NaN();
                }
                result.Add(batchedMatrix);
            }
            return result;
        }


        private static List<Sample<T>> GenerateSamples(List<List<Matrix<T>>> inputs, List<List<Matrix<T>>> targets, int bsize, int seqLen)
        {
            if (bsize > inputs.Count)
                throw new NotSupportedException("Generating of repeated sequences in batches is not supported");
            if(inputs.Count!=targets.Count)
                throw new Exception("Inputs count != targets count");
            var batchSeqI = new List<List<Matrix<T>>>();
            var batchSeqO = new List<List<Matrix<T>>>();

            if (bsize == inputs.Count)
            {
                batchSeqI.AddRange(inputs);
                batchSeqO.AddRange(targets);
            }
            else
            {
                var indexes = GetUniqueIndexes(inputs.Count, bsize);
                foreach (var index in indexes)
                {
                    batchSeqI.Add(inputs[index]);
                    batchSeqO.Add(targets[index]);
                }
            }
            var batchI= GenerateBatch(batchSeqI, seqLen);
            var batchO = GenerateBatch(batchSeqO, seqLen);
            var result = new List<Sample<T>>();
            for (int i = 0; i < batchI.Count; i++)
                result.Add(new Sample<T>(batchI[i], batchO[i]));
            return result;
        } 

        public override IDataSet<T> CreateTrainingSet()
        {
            TrainingSet = new LinearDataSet<T>(GenerateSamples(_trainingInputs, _trainingTargets, batchSize, SequenceLen));
            return TrainingSet;
        }

        public override IDataSet<T> CreateTestSet()
        {
            // TODO: add some check or refactor. Implicit null return value is not good.
            if (_testInputs.Count > 0 && _testTargets.Count > 0)
                TestSet = new LinearDataSet<T>(GenerateSamples(_testInputs, _testTargets, batchSize, SequenceLen));

            return TestSet;
        }

        public override int InputSize => _trainingInputs[0][0].RowCount;
        public override int OutputSize => _trainingInputs[0][0].ColumnCount;
    }
}