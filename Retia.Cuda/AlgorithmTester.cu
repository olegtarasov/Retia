#include "AlgorithmTester.h"
#include "Algorithms.h"
#include "Matrix.h"

double AlgorithmTester::CrossEntropyError(HostMatrixPtr& output, HostMatrixPtr& target)
{
	return Algorithms::CrossEntropyError(output, target);
}

void AlgorithmTester::BackpropagateCrossEntropy(HostMatrixPtr& output, HostMatrixPtr& target, HostMatrixPtr& result)
{
	Algorithms::PropagateError(output, target, result);
}

void AlgorithmTester::RMSPropOptimize(HostMatrixPtr& weight, HostMatrixPtr& gradient, HostMatrixPtr& cache1, HostMatrixPtr& cache2, HostMatrixPtr& cacheM, float learningRate, float decayRate, float momentum, float weightDecay)
{
	Algorithms::RMSPropOptimize(weight, gradient, cache1, cache2, cacheM, learningRate, decayRate, momentum, weightDecay);
}

void AlgorithmTester::ClampMatrix(HostMatrixPtr& matrix, float value)
{
	Algorithms::Clamp(matrix, value);
}
