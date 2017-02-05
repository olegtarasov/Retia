#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cublas.h>
#include <cudnn.h>
#include "MatrixBase.h"

class DeviceMatrixPtr;

template <class TIterator, class TMatPtr>
class Matrix : public MatrixBase<TIterator>
{
public:

	Matrix(int rows, int columns, int seqLength);

	virtual ~Matrix() = default;

	template <class TSource>
	void CopyFrom(TSource& src);

	template <class TDest>
	void CopyTo(TDest& dest);

	template <class TSource>
	void CopyFromLoose(TSource& src);

	template <class TDest>
	void CopyToLoose(TDest& dest);

	template <class TSource>
	void TileFrom(TSource& src);

	void ZeroMemory();

	TMatPtr GetSequenceElement(int idx);

	float get_slow(int row, int col, int seqElement);
};

template <class TIterator>
class DeviceMatrixBase : public Matrix<TIterator, DeviceMatrixPtr>
{
public:
	DeviceMatrixBase(int rows, int columns, int seqLength);

	template <class TA, class TB>
	void Accumulate(TA& A, TB& B, float beta = 0.0f, float alpha = 1.0f, cublasOperation_t transposeA = CUBLAS_OP_N, cublasOperation_t transposeB = CUBLAS_OP_N);

	template <class TA>
	void Accumulate(TA& A);

private:
	/// <summary>
	///     C = alpha*AB + beta*C
	/// </summary>
	template <class TA, class TB, class TC>
	static void DotMatrix(TA& A, TB& B, TC& C, float beta = 0.0f, float alpha = 1.0f, cublasOperation_t transposeA = CUBLAS_OP_N, cublasOperation_t transponseB = CUBLAS_OP_N);

	/// <summary>
	///     y = beta*y + alpha*Ax;
	/// </summary>
	template <class TA, class Tx, class Ty>
	static void DotVec(TA& A, Tx& x, Ty& y, float beta, float alpha, cublasOperation_t transposeA);

	/// <summary>
	///     A = alpha*xyT + A
	/// </summary>
	template <class Tx, class Ty, class TA>
	static void UpdMatFromVec(Tx& x, Ty& y, TA& A, float alpha = 1.0f);
};

class DeviceMatrixPtr : public DeviceMatrixBase<thrust::device_ptr<float>>
{
public:
	DeviceMatrixPtr(int rows, int columns, int seqLength, float* rawPtr);


	DeviceMatrixPtr(cudnnFilterDescriptor_t desc, float* rawPtr);

	thrust::device_ptr<float> begin() override { return _ptr; }
	thrust::device_ptr<float> end() override { return _ptr + this->_length; }
	float* raw_ptr() override { return thrust::raw_pointer_cast(_ptr); }

private:
	thrust::device_ptr<float> _ptr;
};

class DeviceMatrix : public DeviceMatrixBase<thrust::device_vector<float>::iterator>
{
public:
	DeviceMatrix(int rows, int columns, int seqLength);


	thrust::device_vector<float>::iterator begin() override { return _storage.begin(); }
	thrust::device_vector<float>::iterator end() override { return _storage.end(); }
	float* raw_ptr() override { return thrust::raw_pointer_cast(_storage.data()); }
private:
	thrust::device_vector<float> _storage;
};

class HostMatrixPtr : public Matrix<float*, HostMatrixPtr>
{
public:
	HostMatrixPtr(int rows, int columns, int seqLength, float* rawPtr);


	float* begin() override { return _ptr; }
	float* end() override { return _ptr + this->_length; }
	float* raw_ptr() override { return _ptr; }
private:
	float* _ptr;
};

class HostMatrix : public Matrix<thrust::host_vector<float>::iterator, HostMatrixPtr>
{
public:
	HostMatrix(int rows, int columns, int seqLength);


	thrust::host_vector<float>::iterator begin() override { return _storage.begin(); }
	thrust::host_vector<float>::iterator end() override { return _storage.end(); }
	float* raw_ptr() override { return _storage.data(); }
private:
	thrust::host_vector<float> _storage;
};

#include "Matrix.inl"