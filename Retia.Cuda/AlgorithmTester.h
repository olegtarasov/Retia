#pragma once
#include "RawMatrixPtr.h"

class AlgorithmTester
{
public:

	static double CrossEntropyError(RawMatrixPtr& output, RawMatrixPtr& target);
	static void BackpropagateCrossEntropy(RawMatrixPtr& output, RawMatrixPtr& target, RawMatrixPtr& result);
	static void RMSPropOptimize(RawMatrixPtr& weight, RawMatrixPtr& gradient, RawMatrixPtr& cache1, RawMatrixPtr& cache2, RawMatrixPtr& cacheM,
		float learningRate, float decayRate, float momentum, float weightDecay);
	static void ClampMatrix(RawMatrixPtr& matrix, float value);
};
