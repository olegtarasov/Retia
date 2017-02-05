using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra.Single;
using Retia.Mathematics;
using Retia.RandomGenerator;

namespace Retia.Training.Data
{
    public class SetDataProvider: DataProviderBase
    {
        public int BatchSize
        {
            get { return batchSize; }
            set { batchSize = value < 1 ? 1 : value; }
        }

        public int SequenceLen { get; set; }

        private readonly List<List<Matrix>> _trainingInputs = new List<List<Matrix>>();
        private readonly List<List<Matrix>> _trainingTargets = new List<List<Matrix>>();
        private readonly List<List<Matrix>> _testInputs = new List<List<Matrix>>();
        private readonly List<List<Matrix>> _testTargets = new List<List<Matrix>>();
        private int batchSize;
        

        public SetDataProvider(IEnumerable<List<Matrix>> trainingInputs, IEnumerable<List<Matrix>> trainingOutputs,  IEnumerable<List<Matrix>> testInputs = null, IEnumerable<List<Matrix>> testOutputs = null)
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
        private static List<Matrix> GenerateBatch(List<List<Matrix>> sequences, int maxSequenceLen)
        {
            var ioCount = sequences[0][0].RowCount;
            var result = new List<Matrix>();
            for (int i = 0; i < maxSequenceLen; i++)
            {
                var batchedMatrix = new DenseMatrix(ioCount, sequences.Count);
                for (int b = 0; b < sequences.Count; b++)
                {
                    var sequence = sequences[b];
                    for (int j = 0; j < ioCount; j++)
                        batchedMatrix[j, b] = i < sequence.Count ? sequence[i][j, 0] : float.NaN;
                }
                result.Add(batchedMatrix);
            }
            return result;
        }


        private static List<Sample> GenerateSamples(List<List<Matrix>> inputs, List<List<Matrix>> targets, int bsize, int seqLen)
        {
            if (bsize > inputs.Count)
                throw new NotSupportedException("Generating of repeated sequences in batches is not supported");
            if(inputs.Count!=targets.Count)
                throw new Exception("Inputs count != targets count");
            var batchSeqI = new List<List<Matrix>>();
            var batchSeqO = new List<List<Matrix>>();

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
            var result = new List<Sample>();
            for (int i = 0; i < batchI.Count; i++)
                result.Add(new Sample(batchI[i], batchO[i]));
            return result;
        } 

        public override IDataSet CreateTrainingSet()
        {
            TrainingSet = new LinearDataSet(GenerateSamples(_trainingInputs, _trainingTargets, batchSize, SequenceLen));
            return TrainingSet;
        }

        public override IDataSet CreateTestSet()
        {
            // TODO: add some check or refactor. Implicit null return value is not good.
            if (_testInputs.Count > 0 && _testTargets.Count > 0)
                TestSet = new LinearDataSet(GenerateSamples(_testInputs, _testTargets, batchSize, SequenceLen));

            return TestSet;
        }

        public override int InputSize => _trainingInputs[0][0].RowCount;
        public override int OutputSize => _trainingInputs[0][0].ColumnCount;
    }
}