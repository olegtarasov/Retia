#pragma once

#include <memory>
#include "Matrix.h"

class WeightSyncContainer
{
public:
	WeightSyncContainer(int rows, int columns, int seqLen, float* weightPtr, float* gradPtr, float* cache1Ptr, float* cache2Ptr, float* cacheMPtr)
	{
		if (weightPtr == nullptr
			|| gradPtr == nullptr
			|| cache1Ptr == nullptr
			|| cache2Ptr == nullptr
			|| cacheMPtr == nullptr)
		{
			throw RetiaException("All matrices must exist on host!");
		}

		_weight = std::make_unique<HostMatrixPtr>(rows, columns, seqLen, weightPtr);
		_gradient = std::make_unique<HostMatrixPtr>(rows, columns, seqLen, gradPtr);
		_cache1 = std::make_unique<HostMatrixPtr>(rows, columns, seqLen, cache1Ptr);
		_cache2 = std::make_unique<HostMatrixPtr>(rows, columns, seqLen, cache2Ptr);
		_cacheM = std::make_unique<HostMatrixPtr>(rows, columns, seqLen, cacheMPtr);
	}

	HostMatrixPtr* weight() const
	{
		return _weight.get();
	}

	HostMatrixPtr* gradient() const
	{
		return _gradient.get();
	}

	HostMatrixPtr* cache1() const
	{
		return _cache1.get();
	}

	HostMatrixPtr* cache2() const
	{
		return _cache2.get();
	}

	HostMatrixPtr* cacheM() const
	{
		return _cacheM.get();
	}

private:
	std::unique_ptr<HostMatrixPtr> _weight, _gradient, _cache1, _cache2, _cacheM;
};
