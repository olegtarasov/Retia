#include "AlgorithmTester.h"
#include "Algorithms.h"
#include "Matrix.h"

double AlgorithmTester::CrossEntropyError(RawMatrixPtr& output, RawMatrixPtr& target)
{
	return Algorithms::CrossEntropyError(output, target);
}

void AlgorithmTester::BackpropagateCrossEntropy(RawMatrixPtr& output, RawMatrixPtr& target, RawMatrixPtr& result)
{
	Algorithms::PropagateError(output, target, result);
}

void AlgorithmTester::RMSPropOptimize(RawMatrixPtr& weight, RawMatrixPtr& gradient, RawMatrixPtr& cache1, RawMatrixPtr& cache2, RawMatrixPtr& cacheM, float learningRate, float decayRate, float momentum, float weightDecay)
{
	Algorithms::RMSPropOptimize(weight, gradient, cache1, cache2, cacheM, learningRate, decayRate, momentum, weightDecay);
}

void AlgorithmTester::ClampMatrix(RawMatrixPtr& matrix, float value)
{
	Algorithms::Clamp(matrix, value);
}
