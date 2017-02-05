#pragma once

#include <array>
#include <cudnn.h>
#include "CudaMemoryBlock.h"
#include <vector>

template <class T> class CuDnnTensorBase
{
public:
	CuDnnTensorBase(int dim1, int dim2, int dim3)
	{
		_dims.resize(3);
		_dims[0] = dim1;
		_dims[1] = dim2;
		_dims[2] = dim3;
		FillStrides();
		_len = dim1 * dim2 * dim3;
	}

	virtual ~CuDnnTensorBase()
	{
	}

	float* device_ptr() const
	{
		if (_gpuMem == nullptr)
		{
			throw RetiaException("Device memory not allocated!");
		}

		return _gpuMem->device_ptr();
	}

	virtual void CopyToDevice(float* source) const
	{
		if (_gpuMem == nullptr)
		{
			throw RetiaException("Device memory not allocated!");
		}

		auto result = cudaMemcpy(_gpuMem->device_ptr(), source, _len * sizeof(float), cudaMemcpyHostToDevice);
		if (result != cudaSuccess)
		{
			throw CudaException(result);
		}
	}

	virtual void ZeroMemory()
	{
		if (_gpuMem == nullptr)
		{
			throw RetiaException("Device memory not allocated!");
		}

		auto result = cudaMemset(_gpuMem->device_ptr(), 0, _len * sizeof(float));
		if (result != cudaSuccess)
		{
			throw CudaException(result);
		}
	}

	operator T()
	{
		return _tensorDesc;
	}

protected:
	std::vector<int>	_dims;
	std::vector<int>	_strides;
	int					_len = 0;

	std::unique_ptr<CudaTypedMemoryBlock<float>> _gpuMem = nullptr;

	T _tensorDesc;

	void AllocateMemory(int size)
	{
		_gpuMem = std::make_unique<CudaTypedMemoryBlock<float>>(size);
	}

	void FillStrides()
	{
		int cnt = (int)_dims.size();
		_strides.resize(cnt);

		for (int i = cnt - 1; i >= 0; i--)
		{
			int stride = 1;
			for (int j = 0; j < cnt - i - 1; j++)
			{
				stride *= _dims[cnt - j - 1];
			}

			_strides[i] = stride;
		}
	}	
};

class CuDnnNdTensor : public CuDnnTensorBase<cudnnTensorDescriptor_t>
{
public:

	CuDnnNdTensor(int dim1, int dim2, int dim3, bool allocate = true)
		: CuDnnTensorBase<cudnnTensorDescriptor_t>(dim1, dim2, dim3)
	{
		auto result = cudnnCreateTensorDescriptor(&_tensorDesc);
		if (result != CUDNN_STATUS_SUCCESS)
		{
			throw CuDnnException(result);
		}

		result = cudnnSetTensorNdDescriptor(_tensorDesc, CUDNN_DATA_FLOAT, (int)_dims.size(), _dims.data(), _strides.data());
		if (result != CUDNN_STATUS_SUCCESS)
		{
			throw CuDnnException(result);
		}

		if (allocate)
		{
			AllocateMemory(_len * sizeof(float));
		}
	}

	~CuDnnNdTensor()
	{
		cudnnDestroyTensorDescriptor(_tensorDesc);
	}
};

class CuDnnFilter : public CuDnnTensorBase<cudnnFilterDescriptor_t>
{
public:


	CuDnnFilter(int dim1, int dim2, int dim3, bool allocate = true)
		: CuDnnTensorBase<cudnnFilterDescriptor_t>(dim1, dim2, dim3)
	{
		auto result = cudnnCreateFilterDescriptor(&_tensorDesc);
		if (result != CUDNN_STATUS_SUCCESS)
		{
			throw CuDnnException(result);
		}

		result = cudnnSetFilterNdDescriptor(_tensorDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, (int)_dims.size(), _dims.data());
		if (result != CUDNN_STATUS_SUCCESS)
		{
			throw CuDnnException(result);
		}

		if (allocate)
		{
			AllocateMemory(_len * sizeof(float));
		}
	}

	~CuDnnFilter()
	{
		cudnnDestroyFilterDescriptor(_tensorDesc);
	}
};

class CuDnnNdTensorArray : public CuDnnTensorBase<cudnnTensorDescriptor_t*>
{
public:

	CuDnnNdTensorArray(int dim1, int dim2, int dim3, int count, bool allocate = true)
		: CuDnnTensorBase<cudnnTensorDescriptor_t*>(dim1, dim2, dim3)
	{
		_count = count;
		_tensorDesc = new cudnnTensorDescriptor_t[count];

		cudnnStatus_t result;

		for (int i = 0; i < _count; ++i)
		{
			result = cudnnCreateTensorDescriptor(&_tensorDesc[i]);
			if (result != CUDNN_STATUS_SUCCESS)
			{
				throw CuDnnException(result);
			}

			result = cudnnSetTensorNdDescriptor(_tensorDesc[i], CUDNN_DATA_FLOAT, 3, _dims.data(), _strides.data());
			if (result != CUDNN_STATUS_SUCCESS)
			{
				throw CuDnnException(result);
			}
		}

		if (allocate)
		{
			AllocateMemory(_len * _count * sizeof(float));
		}
	}

	~CuDnnNdTensorArray()
	{
		for (int i = 0; i < _count; ++i)
		{
			cudnnDestroyTensorDescriptor(_tensorDesc[i]);
		}

		delete[] _tensorDesc;
	}

	void CopyToDevice(float* source) const override
	{
		if (_gpuMem == nullptr)
		{
			throw RetiaException("Device memory not allocated!");
		}

		auto result = cudaMemcpy(_gpuMem->device_ptr(), source, _len * _count * sizeof(float), cudaMemcpyHostToDevice);
		if (result != cudaSuccess)
		{
			throw CudaException(result);
		}
	}

	void ZeroMemory() override
	{
		if (_gpuMem == nullptr)
		{
			throw RetiaException("Device memory not allocated!");
		}

		auto result = cudaMemset(_gpuMem->device_ptr(), 0, _len * _count * sizeof(float));
		if (result != cudaSuccess)
		{
			throw CudaException(result);
		}
	}
private:
	int _count = 0;
};
