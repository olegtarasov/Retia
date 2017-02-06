#pragma once
#include "Matrix.h"
#include <thrust/fill.h>
#include <thrust/copy.h>
#include "Exceptions.h"
#include "CudaContext.h"

template <class TIterator, class TMatPtr>
Matrix<TIterator, TMatPtr>::Matrix(int rows, int columns, int seqLength) : MatrixBase<TIterator>(rows, columns, seqLength)
{
}

template <class TIterator, class TMatPtr>
template <class TSource>
void Matrix<TIterator, TMatPtr>::CopyFrom(TSource& src)
{
	if (src.rows() != this->_rows || src.columns() != this->_columns || src.seqLength() != this->_seqLength)
		throw RetiaException("Matrix dimensions don't aggree!");

	thrust::copy(src.begin(), src.end(), this->begin());
}

template <class TIterator, class TMatPtr>
template <class TDest>
void Matrix<TIterator, TMatPtr>::CopyTo(TDest& dest)
{
	if (dest.rows() != this->_rows || dest.columns() != this->_columns || dest.seqLength() != this->_seqLength)
		throw RetiaException("Matrix dimensions don't aggree!");

	thrust::copy(this->begin(), this->end(), dest.begin());
}

template <class TIterator, class TMatPtr>
template <class TSource>
void Matrix<TIterator, TMatPtr>::CopyFromLoose(TSource& src)
{
	if (src.length() != this->_length)
		throw RetiaException("Matrix lengths don't aggree!");

	thrust::copy(src.begin(), src.end(), this->begin());
}

template <class TIterator, class TMatPtr>
template <class TDest>
void Matrix<TIterator, TMatPtr>::CopyToLoose(TDest& dest)
{
	if (dest.length() != this->_length)
		throw RetiaException("Matrix lengths don't aggree!");

	thrust::copy(this->begin(), this->end(), dest.begin());
}

template <class TIterator, class TMatPtr>
template <class TSource>
void Matrix<TIterator, TMatPtr>::TileFrom(TSource& src)
{
	if (src.columns() > 1) throw RetiaException("Tiling more than 1 column is not supported");
	if (src.rows() != this->_rows) throw RetiaException("Target matrix rows mismatch");

	for (int i = 0; i < this->_columns * this->_seqLength; ++i)
	{
		thrust::copy(src.begin(), src.end(), this->begin() + this->_rows * i);
	}
}

template <class TIterator, class TMatPtr>
void Matrix<TIterator, TMatPtr>::ZeroMemory()
{
	thrust::fill(this->begin(), this->end(), 0.0f);
}

template <class TIterator, class TMatPtr>
void Matrix<TIterator, TMatPtr>::Fill(float value)
{
	thrust::fill(this->begin(), this->end(), value);
}

template <class TIterator, class TMatPtr>
TMatPtr Matrix<TIterator, TMatPtr>::GetSequenceElement(int idx)
{
	return TMatPtr(this->_rows, this->_columns, 1, this->raw_ptr() + this->_rows * this->_columns * idx);
}

template <class TIterator, class TMatPtr>
float Matrix<TIterator, TMatPtr>::get_slow(int row, int col, int seqElement)
{
	return this->begin()[this->_rows * this->_columns * seqElement + this->_rows * col + row];
}

template <class TIterator>
DeviceMatrixBase<TIterator>::DeviceMatrixBase(int rows, int columns, int seqLength): Matrix<TIterator, DeviceMatrixPtr>(rows, columns, seqLength)
{
}

template <class TIterator>
template <class TA, class TB>
void DeviceMatrixBase<TIterator>::Accumulate(TA& A, TB& B, float beta, float alpha, cublasOperation_t transposeA, cublasOperation_t transposeB)
{
	if ((B.columns() > 1 && transposeB == CUBLAS_OP_N) || (B.rows() > 1 && transposeB == CUBLAS_OP_T))
	{
		DotMatrix(A, B, *this, beta, alpha, transposeA, transposeB);
	}
	else
	{
		if (A.columns() > 1)
		{
			DotVec(A, B, *this, beta, alpha, transposeA);
		}
		else
		{
			UpdMatFromVec(A, B, *this, alpha);
		}
	}
}

template <class TIterator>
template <class TA>
void DeviceMatrixBase<TIterator>::Accumulate(TA& A)
{
	if (this->_rows != A.rows() || this->_columns != A.columns() || this->_seqLength != A.seqLength()) throw RetiaException("Matrix dimensions don't agree!");

	auto thisPtr = this->begin();
	auto thatPtr = A.begin();

	transform(thisPtr, thisPtr + this->_length, thatPtr, thisPtr, thrust::plus<float>());
}

template <class TIterator>
template <class TA, class TB, class TC>
void DeviceMatrixBase<TIterator>::DotMatrix(TA& A, TB& B, TC& C, float beta, float alpha, cublasOperation_t transposeA, cublasOperation_t transponseB)
{
	int m = C.rows();
	int n = C.columns() * C.seqLength();
	int k = (transposeA == CUBLAS_OP_N) ? A.columns() * A.seqLength() : A.rows();
	int bRows = (transponseB == CUBLAS_OP_N) ? B.rows() : B.columns() * B.seqLength();

	if (k != bRows)
	{
		throw CuBlasException(CUBLAS_STATUS_INVALID_VALUE);
	}

	auto result = cublasSgemm_v2(CudaContext::cublasHandle(), transposeA, transponseB, m, n, k, &alpha, A.raw_ptr(), A.rows(), B.raw_ptr(), B.rows(), &beta, C.raw_ptr(), C.rows());
	if (result != CUBLAS_STATUS_SUCCESS)
	{
		throw CuBlasException(result);
	}
}

template <class TIterator>
template <class TA, class Tx, class Ty>
void DeviceMatrixBase<TIterator>::DotVec(TA& A, Tx& x, Ty& y, float beta, float alpha, cublasOperation_t transposeA)
{
	int aCols = transposeA == CUBLAS_OP_N ? A.columns() * A.seqLength() : A.rows();

	if (aCols != x.rows())
	{
		throw CuBlasException(CUBLAS_STATUS_INVALID_VALUE);
	}

	auto result = cublasSgemv_v2(CudaContext::cublasHandle(), transposeA, A.rows(), A.columns() * A.seqLength(), &alpha, A.raw_ptr(), A.rows(), x.raw_ptr(), 1, &beta, y.raw_ptr(), 1);
	if (result != CUBLAS_STATUS_SUCCESS)
	{
		throw CuBlasException(result);
	}
}

template <class TIterator>
template <class Tx, class Ty, class TA>
void DeviceMatrixBase<TIterator>::UpdMatFromVec(Tx& x, Ty& y, TA& A, float alpha)
{
	if (x.rows() != y.columns() * y.seqLength())
	{
		throw CuBlasException(CUBLAS_STATUS_INVALID_VALUE);
	}

	auto result = cublasSger_v2(CudaContext::cublasHandle(), x.length(), y.length(), &alpha, x.raw_ptr(), 1, y.raw_ptr(), 1, A.raw_ptr(), A.rows());
	if (result != CUBLAS_STATUS_SUCCESS)
	{
		throw CuBlasException(result);
	}
}

inline DeviceMatrixPtr::DeviceMatrixPtr(int rows, int columns, int seqLength, float* rawPtr): DeviceMatrixBase<thrust::device_ptr<float>>(rows, columns, seqLength)
{
	_ptr = thrust::device_pointer_cast(rawPtr);
}

inline DeviceMatrixPtr::DeviceMatrixPtr(cudnnFilterDescriptor_t desc, float* rawPtr): DeviceMatrixBase<thrust::device_ptr<float>>(0, 0, 0)
{
	cudnnDataType_t dataType;
	cudnnTensorFormat_t format;
	int nbDims;
	int dims[3];

	auto result = cudnnGetFilterNdDescriptor(desc, 3, &dataType, &format, &nbDims, dims);
	if (result != CUDNN_STATUS_SUCCESS)
	{
		throw CuDnnException(result);
	}

	_rows = dims[1];
	_columns = dims[0];
	_seqLength = dims[2];

	_length = _rows * _columns * _seqLength;
	_ptr = thrust::device_pointer_cast(rawPtr);
}

inline DeviceMatrix::DeviceMatrix(int rows, int columns, int seqLength): DeviceMatrixBase<thrust::device_vector<float>::iterator>(rows, columns, seqLength)
{
	_storage = thrust::device_vector<float>(_length);
}

inline HostMatrixPtr::HostMatrixPtr(int rows, int columns, int seqLength, float* rawPtr): Matrix<float*, HostMatrixPtr>(rows, columns, seqLength)
{
	_ptr = rawPtr;
}

inline HostMatrix::HostMatrix(int rows, int columns, int seqLength): Matrix<thrust::host_vector<float>::iterator, HostMatrixPtr>(rows, columns, seqLength)
{
	_storage = thrust::host_vector<float>(_length);
}
