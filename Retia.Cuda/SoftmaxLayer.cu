#include "SoftmaxLayer.h"

#include <thrust/transform.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <cudnn.h>
#include "CudaContext.h"
#include "Algorithms.h"
#include <iostream>
#include "Helpers.h"

using std::cout;
using std::endl;

using thrust::get;

SoftmaxLayer::SoftmaxLayer(int inSize, int batchSize, int seqLen)
	: LayerBase(inSize, inSize, batchSize, seqLen)
{
	_xTensor = std::make_unique<CuDnnNdTensor>(_batchSize * _seqLen, _inputSize, 1, false);
	_output = std::make_unique<DeviceMatrix>(_inputSize, _batchSize, _seqLen);
	_sensitivity = std::make_unique<DeviceMatrix>(_inputSize, _batchSize, _seqLen);
}

void SoftmaxLayer::TransferStatesFromHost(std::vector<RawMatrixPtr*>& states)
{
	// Nothing to do here.
}

void SoftmaxLayer::TransferStatesToHost(std::vector<RawMatrixPtr*>& states)
{
	// Nothing to do here.
}

void SoftmaxLayer::ForwardSequence(DeviceMatrix& input)
{
	/*cout << "Softmax input" << endl;
	PrintMatrix(input);*/

	float alpha = 1.0f;
	float beta = 0.0f;
	auto result = cudnnSoftmaxForward(CudaContext::cudnnHandle(), CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha, *_xTensor, input.raw_ptr(), &beta, *_xTensor, _output->raw_ptr());
	if (result != CUDNN_STATUS_SUCCESS)
	{
		throw CuDnnException(result);
	}

	/*cout << "Softmax output" << endl;
	PrintMatrix(*_output);*/
}

void SoftmaxLayer::BackpropSequence(DeviceMatrix& input, DeviceMatrix& outSens)
{
	throw RetiaException("Can't propagate through softmax yet. Use ErrorPropagate.");
}

void SoftmaxLayer::Optimize(OptimizerBase& optimizer)
{
}

void SoftmaxLayer::ErrorPropagate(DeviceMatrix& target)
{
	Algorithms::PropagateError(*_output, target, *_sensitivity);
}

double SoftmaxLayer::LayerError(DeviceMatrix& target)
{
	return Algorithms::CrossEntropyError(*_output, target);
}

void SoftmaxLayer::ResetMemory()
{
}

void SoftmaxLayer::ResetOptimizerCache()
{
}
