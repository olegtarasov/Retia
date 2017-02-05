#pragma once
#include <cudnn.h>
#include <cublas.h>
#include <string>

template <class T>
class CudaExceptionBase
{
public:

	explicit CudaExceptionBase(T status)
		: _status(status)
	{
	}

	T status() const
	{
		return _status;
	}

private:
	T	_status;
};

class CudaException : public CudaExceptionBase<cudaError_t>
{
public:
	explicit CudaException(cudaError_t status)
		: CudaExceptionBase<cudaError_t>(status)
	{
	}
};

class CuDnnException : public CudaExceptionBase<cudnnStatus_t>
{
public:
	explicit CuDnnException(cudnnStatus_t status)
		: CudaExceptionBase<cudnnStatus_t>(status)
	{
	}
};

class CuBlasException : public CudaExceptionBase<cublasStatus_t>
{
public:
	explicit CuBlasException(cublasStatus_t status)
		: CudaExceptionBase<cublasStatus_t>(status)
	{
	}
};

class RetiaException
{
public:
	explicit RetiaException(const std::string& message)
		: _message(message)
	{
	}


	std::string message() const
	{
		return _message;
	}

private:
	std::string	_message;
};