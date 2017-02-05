#pragma once

#include <memory>
#include <cuda_runtime_api.h>
#include "Exceptions.h"

struct CudaMemoryDeleter
{
	void operator()(void* p) const {
		cudaFree(p);
	};
};

template <class T> class CudaTypedMemoryBlock
{
public:
	CudaTypedMemoryBlock(size_t size) : _size(size)
	{
		void* ptr;
		auto result = cudaMalloc(&ptr, _size);
		if (result != cudaSuccess)
		{
			throw CudaException(result);
		}

		_ptr = std::unique_ptr<T, CudaMemoryDeleter>((T*)ptr);

		result = cudaMemset(_ptr.get(), 0, _size);
		if (result != cudaSuccess)
		{
			throw CudaException(result);
		}
	}

	T* device_ptr() const
	{
		return _ptr.get();
	}

	size_t size() const
	{
		return _size;
	}

private:
	size_t _size = 0;
	std::unique_ptr<T, CudaMemoryDeleter> _ptr = nullptr;
};

class CudaMemoryBlock : public CudaTypedMemoryBlock<void>
{
public:
	explicit CudaMemoryBlock(size_t size)
		: CudaTypedMemoryBlock<void>(size)
	{
	}
};