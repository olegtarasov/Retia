#pragma once

#include <memory>
#include "Matrix.h"

class NeuroWeigth
{
public:
	NeuroWeigth(int rows, int columns = 1, int seqLength = 1)
	{
		_weight = std::make_unique<DeviceMatrix>(rows, columns, seqLength);
		_gradient = std::make_unique<DeviceMatrix>(rows, columns, seqLength);
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

	void ClearCache();
	void ClearGradient();

private:
	std::unique_ptr<DeviceMatrix>	_weight;
	std::unique_ptr<DeviceMatrix>	_gradient;
	std::unique_ptr<DeviceMatrix>	_cache1;
	std::unique_ptr<DeviceMatrix>	_cache2;
	std::unique_ptr<DeviceMatrix>	_cacheM;
};
