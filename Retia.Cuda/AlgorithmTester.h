#pragma once

#include "Matrix.h"

class AlgorithmTester
{
public:

	static double CrossEntropyError(HostMatrixPtr& output, HostMatrixPtr& target);
	static void BackpropagateCrossEntropy(HostMatrixPtr& output, HostMatrixPtr& target, HostMatrixPtr& result);
	static void RMSPropOptimize(HostMatrixPtr& weight, HostMatrixPtr& gradient, HostMatrixPtr& cache1, HostMatrixPtr& cache2, HostMatrixPtr& cacheM,
		float learningRate, float decayRate, float momentum, float weightDecay);
	static void ClampMatrix(HostMatrixPtr& matrix, float value);
};
