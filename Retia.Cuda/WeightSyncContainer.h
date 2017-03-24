#pragma once

#include <memory>
#include "Matrix.h"

class WeightSyncContainer
{
public:
	WeightSyncContainer(HostMatrixPtr* weight, HostMatrixPtr* gradient, HostMatrixPtr* cache1, HostMatrixPtr* cache2, HostMatrixPtr* cacheM)
		: _weight(weight),
		_gradient(gradient),
		_cache1(cache1),
		_cache2(cache2),
		_cacheM(cacheM)
	{
		if (_weight == nullptr
			|| _gradient == nullptr
			|| _cache1 == nullptr
			|| _cache2 == nullptr
			|| _cacheM == nullptr)
		{
			throw RetiaException("All matrices must exist on host!");
		}
	}

	std::unique_ptr<HostMatrixPtr> weight() const
	{
		return _weight;
	}

	std::unique_ptr<HostMatrixPtr> gradient() const
	{
		return _gradient;
	}

	std::unique_ptr<HostMatrixPtr> cache1() const
	{
		return _cache1;
	}

	std::unique_ptr<HostMatrixPtr> cache2() const
	{
		return _cache2;
	}

	std::unique_ptr<HostMatrixPtr> cacheM() const
	{
		return _cacheM;
	}

private:
	std::unique_ptr<HostMatrixPtr> _weight, _gradient, _cache1, _cache2, _cacheM;
};
