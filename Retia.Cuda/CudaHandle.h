#pragma once

#include <cudnn.h>
#include <cublas.h>
#include "Exceptions.h"

template <class THandle, class TStatus> 
class CudaHandle
{
public:

	CudaHandle()
	{		
	}

	CudaHandle(TStatus(* creator)(THandle* ptr), TStatus(* destructor)(THandle desc))
		: _creator(creator),
		_destructor(destructor)
	{
	}

	~CudaHandle()
	{
		if (_descriptor != nullptr)
		{
			_destructor(_descriptor);
		}
	}

	void Create()
	{
		TStatus result = _creator(&_descriptor);
		if (result != 0)
		{
			throw CudaExceptionBase<TStatus>(result);
		}
	}

	void CheckCreated()
	{
		if (_descriptor == nullptr)
		{
			Create();
		}
	}

	operator THandle()
	{
		return _descriptor;
	}
private:
	THandle	_descriptor = nullptr;

	TStatus(*_creator)(THandle* ptr) = nullptr;
	TStatus(*_destructor)(THandle desc) = nullptr;
};

class CuDnnRnnDescriptor : public CudaHandle<cudnnRNNDescriptor_t, cudnnStatus_t>
{
public:
	CuDnnRnnDescriptor()
		: CudaHandle<cudnnRNNDescriptor_t, cudnnStatus_t>(cudnnCreateRNNDescriptor, cudnnDestroyRNNDescriptor)
	{
	}
};

class CuDnnDropoutDescriptor : public CudaHandle<cudnnDropoutDescriptor_t, cudnnStatus_t>
{
public:
	CuDnnDropoutDescriptor()
		: CudaHandle<cudnnDropoutDescriptor_t, cudnnStatus_t>(cudnnCreateDropoutDescriptor, cudnnDestroyDropoutDescriptor)
	{
	}
};

class CuDnnHandle : public CudaHandle<cudnnHandle_t, cudnnStatus_t>
{
public:
	CuDnnHandle()
		: CudaHandle<cudnnHandle_t, cudnnStatus_t>(cudnnCreate, cudnnDestroy)
	{
	}
};

class CuBlasHandle : public CudaHandle<cublasHandle_t, cublasStatus_t>
{
public:
	CuBlasHandle()
		: CudaHandle<cublasHandle_t, cublasStatus_t>(cublasCreate_v2, cublasDestroy_v2)
	{
	}
};