#pragma once

#include <memory>
#include "Matrix.h"
#include "WeightSyncContainer.h"

template <class T>
class NeuroWeightBase
{
public:
	NeuroWeightBase(int rows, int columns = 1, int seqLength = 1)
	{
		_cache1 = std::make_unique<DeviceMatrix>(rows, columns, seqLength);
		_cache2 = std::make_unique<DeviceMatrix>(rows, columns, seqLength);
		_cacheM = std::make_unique<DeviceMatrix>(rows, columns, seqLength);

		_weight->ZeroMemory();
		_gradient->ZeroMemory();
		_cache1->ZeroMemory();
		_cache2->ZeroMemory();
		_cacheM->ZeroMemory();
	}

	DeviceMatrix& weight() const
	{
		return *_weight;
	}

	DeviceMatrix& gradient() const
	{
		return *_gradient;
	}

	DeviceMatrix& cache1() const
	{
		return *_cache1;
	}

	DeviceMatrix& cache2() const
	{
		return *_cache2;
	}

	DeviceMatrix& cache_m() const
	{
		return *_cacheM;
	}

	void ClearCache()
	{
		_cache1->ZeroMemory();
		_cache2->ZeroMemory();
		_cacheM->ZeroMemory();
	}

	void ClearGradient()
	{
		_gradient->ZeroMemory();
	}

	void TransferStateToDevice(WeightSyncContainer& container)
	{
		_weight->CopyFrom(*container.weight());
		_gradient->CopyFrom(*container.gradient());
		_cache1->CopyFrom(*container.cache1());
		_cache2->CopyFrom(*container.cache2());
		_cacheM->CopyFrom(*container.cacheM());
	}

	void TransferStateToHost(WeightSyncContainer& container)
	{
		_weight->CopyTo(*container.weight());
		_gradient->CopyTo(*container.gradient());
		_cache1->CopyTo(*container.cache1());
		_cache2->CopyTo(*container.cache2());
		_cacheM->CopyTo(*container.cacheM());
	}

protected:
	std::unique_ptr<T>	_weight;
	std::unique_ptr<T>	_gradient;
	std::unique_ptr<DeviceMatrix>	_cache1;
	std::unique_ptr<DeviceMatrix>	_cache2;
	std::unique_ptr<DeviceMatrix>	_cacheM;
};

class NeuroWeight : public NeuroWeightBase<DeviceMatrix>
{
public:
	NeuroWeight(int rows, int columns = 1, int seqLength = 1)
		: NeuroWeightBase<DeviceMatrix>(rows, columns, seqLength)
	{
		_weight = std::make_unique<DeviceMatrix>(rows, columns, seqLength);
		_gradient = std::make_unique<DeviceMatrix>(rows, columns, seqLength);
	}	
};

class NeuroWeightPtr : public NeuroWeightBase<DeviceMatrixPtr>
{
public:
	NeuroWeightPtr(int rows, int columns = 1, int seqLength = 1, float* weightPtr, float* gradPtr)
		: NeuroWeightBase<DeviceMatrixPtr>(rows, columns, seqLength)
	{
		_weight = std::make_unique<DeviceMatrixPtr>(rows, columns, seqLength, weightPtr);
		_gradient = std::make_unique<DeviceMatrixPtr>(rows, columns, seqLength, gradPtr);
	}
};