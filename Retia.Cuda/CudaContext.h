#pragma once
#include <memory>
#include "CudaHandle.h"

class CudaContext
{
public:

	static cudnnHandle_t cudnnHandle();
	static cublasHandle_t cublasHandle();

private:
	static thread_local std::unique_ptr<CuDnnHandle>	_cudnnHandle;
	static thread_local std::unique_ptr<CuBlasHandle>	_cuBlasHandle;

	template <class T> 
	static void CheckHandleCreated(std::unique_ptr<T> *ptr);
};
