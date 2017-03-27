#include "TestApi.h"
#include "Algorithms.h"
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>

double TestCrossEntropyErrorCpu(MatrixDefinition m1, MatrixDefinition m2)
{
	auto mat1 = std::make_unique<HostMatrixPtr>(m1.Rows, m1.Columns, m1.SeqLength, m1.Pointer);
	auto mat2 = std::make_unique<HostMatrixPtr>(m2.Rows, m2.Columns, m2.SeqLength, m2.Pointer);

	return Algorithms::CrossEntropyError(*mat1, *mat2);
}

double TestCrossEntropyErrorGpu(MatrixDefinition m1, MatrixDefinition m2)
{
	auto mat1 = std::make_unique<HostMatrixPtr>(m1.Rows, m1.Columns, m1.SeqLength, m1.Pointer);
	auto mat2 = std::make_unique<HostMatrixPtr>(m2.Rows, m2.Columns, m2.SeqLength, m2.Pointer);

	auto gpum1 = std::make_unique<DeviceMatrix>(m1.Rows, m1.Columns, m1.SeqLength);
	auto gpum2 = std::make_unique<DeviceMatrix>(m2.Rows, m2.Columns, m2.SeqLength);

	mat1->CopyTo(*gpum1);
	mat2->CopyTo(*gpum2);

	return Algorithms::CrossEntropyError(*gpum1, *gpum2);
}

void TestCrossEntropyBackpropCpu(MatrixDefinition m1, MatrixDefinition m2, MatrixDefinition result)
{
	auto mat1 = std::make_unique<HostMatrixPtr>(m1.Rows, m1.Columns, m1.SeqLength, m1.Pointer);
	auto mat2 = std::make_unique<HostMatrixPtr>(m2.Rows, m2.Columns, m2.SeqLength, m2.Pointer);
	auto mResult = std::make_unique<HostMatrixPtr>(result.Rows, result.Columns, result.SeqLength, result.Pointer);

	Algorithms::BackpropagateCrossEntropyError(*mat1, *mat2, *mResult);
}

void TestCrossEntropyBackpropGpu(MatrixDefinition m1, MatrixDefinition m2, MatrixDefinition result)
{
	auto mat1 = std::make_unique<HostMatrixPtr>(m1.Rows, m1.Columns, m1.SeqLength, m1.Pointer);
	auto mat2 = std::make_unique<HostMatrixPtr>(m2.Rows, m2.Columns, m2.SeqLength, m2.Pointer);
	auto mResult = std::make_unique<HostMatrixPtr>(result.Rows, result.Columns, result.SeqLength, result.Pointer);

	auto gpum1 = std::make_unique<DeviceMatrix>(m1.Rows, m1.Columns, m1.SeqLength);
	auto gpum2 = std::make_unique<DeviceMatrix>(m2.Rows, m2.Columns, m2.SeqLength);
	auto gpumRes = std::make_unique<DeviceMatrix>(result.Rows, result.Columns, result.SeqLength);

	mat1->CopyTo(*gpum1);
	mat2->CopyTo(*gpum2);
	mResult->CopyTo(*gpumRes);

	Algorithms::BackpropagateCrossEntropyError(*gpum1, *gpum2, *gpumRes);

	mResult->CopyFrom(*gpumRes);
}

void TestRMSPropUpdateCpu(MatrixDefinition weight, MatrixDefinition grad, MatrixDefinition cache1, MatrixDefinition cache2, MatrixDefinition cacheM, float learningRate, float decayRate, float momentum, float weightDecay)
{
	auto mWeight = std::make_unique<HostMatrixPtr>(weight.Rows, weight.Columns, weight.SeqLength, weight.Pointer);
	auto mGrad = std::make_unique<HostMatrixPtr>(grad.Rows, grad.Columns, grad.SeqLength, grad.Pointer);
	auto mCache1 = std::make_unique<HostMatrixPtr>(cache1.Rows, cache1.Columns, cache1.SeqLength, cache1.Pointer);
	auto mCache2 = std::make_unique<HostMatrixPtr>(cache2.Rows, cache2.Columns, cache2.SeqLength, cache2.Pointer);
	auto mCacheM = std::make_unique<HostMatrixPtr>(cacheM.Rows, cacheM.Columns, cacheM.SeqLength, cacheM.Pointer);

	Algorithms::RMSPropOptimize(*mWeight, *mGrad, *mCache1, *mCache2, *mCacheM, learningRate, decayRate, momentum, weightDecay);
}

void TestRMSPropUpdateGpu(MatrixDefinition weight, MatrixDefinition grad, MatrixDefinition cache1, MatrixDefinition cache2, MatrixDefinition cacheM, float learningRate, float decayRate, float momentum, float weightDecay)
{
	auto mWeight = std::make_unique<HostMatrixPtr>(weight.Rows, weight.Columns, weight.SeqLength, weight.Pointer);
	auto mGrad = std::make_unique<HostMatrixPtr>(grad.Rows, grad.Columns, grad.SeqLength, grad.Pointer);
	auto mCache1 = std::make_unique<HostMatrixPtr>(cache1.Rows, cache1.Columns, cache1.SeqLength, cache1.Pointer);
	auto mCache2 = std::make_unique<HostMatrixPtr>(cache2.Rows, cache2.Columns, cache2.SeqLength, cache2.Pointer);
	auto mCacheM = std::make_unique<HostMatrixPtr>(cacheM.Rows, cacheM.Columns, cacheM.SeqLength, cacheM.Pointer);

	auto gmWeight = std::make_unique<DeviceMatrix>(weight.Rows, weight.Columns, weight.SeqLength);
	auto gmGrad = std::make_unique<DeviceMatrix>(grad.Rows, grad.Columns, grad.SeqLength);
	auto gmCache1 = std::make_unique<DeviceMatrix>(cache1.Rows, cache1.Columns, cache1.SeqLength);
	auto gmCache2 = std::make_unique<DeviceMatrix>(cache2.Rows, cache2.Columns, cache2.SeqLength);
	auto gmCacheM = std::make_unique<DeviceMatrix>(cacheM.Rows, cacheM.Columns, cacheM.SeqLength);

	mWeight->CopyTo(*gmWeight);
	mGrad->CopyTo(*gmGrad);
	mCache1->CopyTo(*gmCache1);
	mCache2->CopyTo(*gmCache2);
	mCacheM->CopyTo(*gmCacheM);

	Algorithms::RMSPropOptimize(*gmWeight, *gmGrad, *gmCache1, *gmCache2, *gmCacheM, learningRate, decayRate, momentum, weightDecay);

	mWeight->CopyFrom(*gmWeight);
	mCache1->CopyFrom(*gmCache1);
	mCache2->CopyFrom(*gmCache2);
	mCacheM->CopyFrom(*gmCacheM);
}

void TestClampMatrixCpu(MatrixDefinition matrix, float threshold)
{
	auto mat = std::make_unique<HostMatrixPtr>(matrix.Rows, matrix.Columns, matrix.SeqLength, matrix.Pointer);

	Algorithms::Clamp(*mat, threshold);
}

void TestClampMatrixGpu(MatrixDefinition matrix, float threshold)
{
	auto mat = std::make_unique<HostMatrixPtr>(matrix.Rows, matrix.Columns, matrix.SeqLength, matrix.Pointer);
	auto gMat = std::make_unique<DeviceMatrix>(matrix.Rows, matrix.Columns, matrix.SeqLength);

	mat->CopyTo(*gMat);

	Algorithms::Clamp(*gMat, threshold);

	mat->CopyFrom(*gMat);
}

struct test_mutator
{
	template <typename Tuple>
	__host__ __device__
		void operator()(Tuple t)
	{
		thrust::get<0>(t) = thrust::get<0>(t) + thrust::get<1>(t);
	}
};

template <class T>
void MutateVector(int rows, int cols, T& vector)
{
	auto data = vector.data();

	for (int col = 0; col < cols; ++col)
	{
		for (int row = 0; row < rows; ++row)
		{
			vector[rows * col + row] = row - col;
		}
	}
}

template <class T>
void MutateVectorRowMajor(int rows, int cols, T& vector)
{
	for (int row = 0; row < rows; ++row)
	{
		for (int col = 0; col < cols; ++col)
		{
			vector[cols * row + col] = row - col;
		}
	}
}

void TestHostMatrix(int rows, int cols, float* ptr)
{
	auto mat = std::make_unique<HostMatrixPtr>(rows, cols, 1, ptr);
	auto coeff = thrust::host_vector<float>(mat->length());

	MutateVector(rows, cols, coeff);

	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(mat->begin(), coeff.begin())),
		thrust::make_zip_iterator(thrust::make_tuple(mat->end(), coeff.end())),
		test_mutator());
}

void TestDeviceMatrix(int rows, int cols, float* ptr)
{
	auto mat = std::make_unique<HostMatrixPtr>(rows, cols, 1, ptr);
	auto gMat = std::make_unique<DeviceMatrix>(rows, cols, 1);
	auto coeff = thrust::device_vector<float>(mat->length());

	MutateVector(rows, cols, coeff);

	mat->CopyTo(*gMat);

	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(gMat->begin(), coeff.begin())),
		thrust::make_zip_iterator(thrust::make_tuple(gMat->end(), coeff.end())),
		test_mutator());

	mat->CopyFrom(*gMat);
}

void TestHostMatrixRowMajor(int rows, int cols, float* ptr)
{
	auto mat = std::make_unique<HostMatrixPtr>(rows, cols, 1, ptr);
	auto coeff = thrust::host_vector<float>(mat->length());

	MutateVectorRowMajor(rows, cols, coeff);

	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(mat->begin(), coeff.begin())),
		thrust::make_zip_iterator(thrust::make_tuple(mat->end(), coeff.end())),
		test_mutator());
}

void TestDeviceMatrixRowMajor(int rows, int cols, float* ptr)
{
	auto mat = std::make_unique<HostMatrixPtr>(rows, cols, 1, ptr);
	auto gMat = std::make_unique<DeviceMatrix>(rows, cols, 1);
	auto coeff = thrust::device_vector<float>(mat->length());

	MutateVectorRowMajor(rows, cols, coeff);

	mat->CopyTo(*gMat);

	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(gMat->begin(), coeff.begin())),
		thrust::make_zip_iterator(thrust::make_tuple(gMat->end(), coeff.end())),
		test_mutator());

	mat->CopyFrom(*gMat);
}

void TestMatrixTransferCpu(MatrixDefinition matrix)
{
	TestHostMatrix(matrix.Rows, matrix.Columns, matrix.Pointer);
}

void TestMatrixTransferGpu(MatrixDefinition matrix)
{
	TestDeviceMatrix(matrix.Rows, matrix.Columns, matrix.Pointer);
}

void TestMatrixTransferRowMajorCpu(MatrixDefinition matrix)
{
	TestHostMatrixRowMajor(matrix.Rows, matrix.Columns, matrix.Pointer);
}

void TestMatrixTransferRowMajorGpu(MatrixDefinition matrix)
{
	TestDeviceMatrixRowMajor(matrix.Rows, matrix.Columns, matrix.Pointer);
}

void TestWeightTransferCpu(WeightDefinition weight)
{
	TestHostMatrix(weight.Rows, weight.Columns, weight.WeightPtr);
	TestHostMatrix(weight.Rows, weight.Columns, weight.GradPtr);
	TestHostMatrix(weight.Rows, weight.Columns, weight.Cache1Ptr);
	TestHostMatrix(weight.Rows, weight.Columns, weight.Cache2Ptr);
	TestHostMatrix(weight.Rows, weight.Columns, weight.CacheMPtr);
}

void TestWeightTransferGpu(WeightDefinition weight)
{
	TestDeviceMatrix(weight.Rows, weight.Columns, weight.WeightPtr);
	TestDeviceMatrix(weight.Rows, weight.Columns, weight.GradPtr);
	TestDeviceMatrix(weight.Rows, weight.Columns, weight.Cache1Ptr);
	TestDeviceMatrix(weight.Rows, weight.Columns, weight.Cache2Ptr);
	TestDeviceMatrix(weight.Rows, weight.Columns, weight.CacheMPtr);
}

void TestWeightTransferRowMajorCpu(WeightDefinition weight)
{
	TestHostMatrixRowMajor(weight.Rows, weight.Columns, weight.WeightPtr);
	TestHostMatrixRowMajor(weight.Rows, weight.Columns, weight.GradPtr);
	TestHostMatrixRowMajor(weight.Rows, weight.Columns, weight.Cache1Ptr);
	TestHostMatrixRowMajor(weight.Rows, weight.Columns, weight.Cache2Ptr);
	TestHostMatrixRowMajor(weight.Rows, weight.Columns, weight.CacheMPtr);
}

void TestWeightTransferRowMajorGpu(WeightDefinition weight)
{
	TestDeviceMatrixRowMajor(weight.Rows, weight.Columns, weight.WeightPtr);
	TestDeviceMatrixRowMajor(weight.Rows, weight.Columns, weight.GradPtr);
	TestDeviceMatrixRowMajor(weight.Rows, weight.Columns, weight.Cache1Ptr);
	TestDeviceMatrixRowMajor(weight.Rows, weight.Columns, weight.Cache2Ptr);
	TestDeviceMatrixRowMajor(weight.Rows, weight.Columns, weight.CacheMPtr);
}

void TestComplexWeightTransfer(WeightDefinition weight)
{
	auto gWeight = std::make_unique<NeuroWeight>(weight.Rows, weight.Columns, weight.SeqLength);
	auto container = std::make_unique<WeightSyncContainer>(weight.Rows, weight.Columns, weight.SeqLength,
		weight.WeightPtr, weight.GradPtr, weight.Cache1Ptr, weight.Cache2Ptr, weight.CacheMPtr);
	auto coeff = thrust::device_vector<float>(container->weight()->length());

	MutateVector(weight.Rows, weight.Columns, coeff);

	gWeight->TransferStateToDevice(*container);

	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(gWeight->weight().begin(), coeff.begin())),
		thrust::make_zip_iterator(thrust::make_tuple(gWeight->weight().end(), coeff.end())),
		test_mutator());
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(gWeight->gradient().begin(), coeff.begin())),
		thrust::make_zip_iterator(thrust::make_tuple(gWeight->gradient().end(), coeff.end())),
		test_mutator());
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(gWeight->cache1().begin(), coeff.begin())),
		thrust::make_zip_iterator(thrust::make_tuple(gWeight->cache1().end(), coeff.end())),
		test_mutator());
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(gWeight->cache2().begin(), coeff.begin())),
		thrust::make_zip_iterator(thrust::make_tuple(gWeight->cache2().end(), coeff.end())),
		test_mutator());
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(gWeight->cache_m().begin(), coeff.begin())),
		thrust::make_zip_iterator(thrust::make_tuple(gWeight->cache_m().end(), coeff.end())),
		test_mutator());
	
	gWeight->TransferStateToHost(*container);
}

void TestComplexWeightTransferRowMajor(WeightDefinition weight)
{
	auto gWeight = std::make_unique<NeuroWeight>(weight.Rows, weight.Columns, weight.SeqLength);
	auto container = std::make_unique<WeightSyncContainer>(weight.Rows, weight.Columns, weight.SeqLength,
		weight.WeightPtr, weight.GradPtr, weight.Cache1Ptr, weight.Cache2Ptr, weight.CacheMPtr);
	auto coeff = thrust::device_vector<float>(container->weight()->length());

	MutateVectorRowMajor(weight.Rows, weight.Columns, coeff);

	gWeight->TransferStateToDevice(*container);

	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(gWeight->weight().begin(), coeff.begin())),
		thrust::make_zip_iterator(thrust::make_tuple(gWeight->weight().end(), coeff.end())),
		test_mutator());
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(gWeight->gradient().begin(), coeff.begin())),
		thrust::make_zip_iterator(thrust::make_tuple(gWeight->gradient().end(), coeff.end())),
		test_mutator());
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(gWeight->cache1().begin(), coeff.begin())),
		thrust::make_zip_iterator(thrust::make_tuple(gWeight->cache1().end(), coeff.end())),
		test_mutator());
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(gWeight->cache2().begin(), coeff.begin())),
		thrust::make_zip_iterator(thrust::make_tuple(gWeight->cache2().end(), coeff.end())),
		test_mutator());
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(gWeight->cache_m().begin(), coeff.begin())),
		thrust::make_zip_iterator(thrust::make_tuple(gWeight->cache_m().end(), coeff.end())),
		test_mutator());

	gWeight->TransferStateToHost(*container);
}

