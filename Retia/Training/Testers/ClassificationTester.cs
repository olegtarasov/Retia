using System;
using System.Collections.Generic;
using System.Linq;
using Retia.Mathematics;
using Retia.Neural;
using Retia.Neural.Layers;
using Retia.Training.Data;

namespace Retia.Training.Testers
{
    public class ClassificationTester<T> : TesterBase<T, ConfusionMatrix> where T : struct, IEquatable<T>, IFormattable
    {
        private readonly List<string> _classNames = null;

        public ClassificationTester()
        {
        }

        public ClassificationTester(List<string> classNames)
        {
            _classNames = classNames;
        }

        protected override ConfusionMatrix TestInternal(NeuralNet<T> network, IDataSet<T> testSet)
        {
            var errors = new List<double>();
            int sampleCount = testSet.SampleCount;
            int classCount = testSet.TargetSize;
            int batchSize = testSet.BatchSize;
            ConfusionMatrix confusionMatrix;

            if (_classNames?.Count > 0)
            {
                if (classCount != _classNames?.Count)
                {
                    throw new InvalidOperationException("Class names count isn't equal to test set class count!");
                }

                confusionMatrix = new ConfusionMatrix(_classNames);
            }
            else
            {
                confusionMatrix = new ConfusionMatrix(classCount);
            }

            for (int i = 0; i < sampleCount; i++)
            {
                var sample = testSet.GetNextSample();
                var target = sample.Target;
                var stepResult = network.Step(sample.Input);
                var predicted = MathProvider.SoftMaxChoice(stepResult);

                int targetClass = -1;

                for (int colIdx = 0; colIdx < batchSize; colIdx++)
                {
                    for (int classIdx = 0; classIdx < classCount; classIdx++)
                    {
                        // TODO: Hacky convertion
                        if ((int)(object)target[classIdx, colIdx] == 1)
                        {
                            targetClass = classIdx;
                            break;
                        }
                    }

                    if (targetClass < 0)
                    {
                        throw new InvalidOperationException("Target vector doesn't contain a positive result!");
                    }

                    confusionMatrix.Prediction(targetClass, predicted[colIdx]);
                }

                errors.Add(MathProvider.CrossEntropyError(MathProvider.SoftMaxNorm(stepResult), sample.Target));
            }

            confusionMatrix.CalculateResult(sampleCount * batchSize);
            confusionMatrix.Error = errors.Sum() / errors.Count;

            return confusionMatrix;
        }
    }
}