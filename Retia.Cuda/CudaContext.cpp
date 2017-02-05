#include "CudaContext.h"

thread_local std::unique_ptr<CuDnnHandle> CudaContext::_cudnnHandle;
thread_local std::unique_ptr<CuBlasHandle> CudaContext::_cuBlasHandle;

template <class T>
void CudaContext::CheckHandleCreated(std::unique_ptr<T>* ptr)
{
	if (*ptr == nullptr)
	{
		*ptr = std::make_unique<T>();
	}
}

cudnnHandle_t CudaContext::cudnnHandle()
{
	CheckHandleCreated(&_cudnnHandle);
	_cudnnHandle->CheckCreated();
	return *_cudnnHandle;
}

cublasHandle_t CudaContext::cublasHandle()
{
	CheckHandleCreated(&_cuBlasHandle);
	_cuBlasHandle->CheckCreated();
	return *_cuBlasHandle;
}
