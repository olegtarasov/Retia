#include "GruLayer.h"
#include "CudaContext.h"
#include "Algorithms.h"
#include "Helpers.h"
#include <iostream>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>

using std::make_unique;
using std::cout;
using std::endl;

GruLayer::GruLayer(int inSize, int hSize, int layers, int batchSize, int seqLength)
	: LayerBase(inSize, hSize, batchSize, seqLength), 
	_hSize(hSize),
	_layers(layers)
{
	InitLayers();
}

/*
* States indexes for each layer:
* 0  - Wxr
* 1  - Wxz
* 2  - Wxh
*
* 3  - Whr
* 4  - Whz
* 5  - Whh
*
* 6  - bxr
* 7  - bxz
* 8  - bxh
*
* 9  - bhr
* 10 - bhz
* 11 - bhh
*/
void GruLayer::TransferStatesToDevice(std::vector<WeightSyncContainer*>& states)
{
	if (states.size() != _weights.size()) throw RetiaException("There should be exactly 12 state vectors for each layer");

	for (int i = 0; i < states.size(); ++i)
	{
		_weights[i]->TransferStateToDeviceLoose(*states[i]);
	}
}

void GruLayer::TransferStatesToHost(std::vector<WeightSyncContainer*>& states)
{
	if (states.size() != _weights.size()) throw RetiaException("There should be exactly 12 state vectors for each layer");

	for (int i = 0; i < states.size(); ++i)
	{
		_weights[i]->TransferStateToHostLoose(*states[i]);
	}
}

void GruLayer::ForwardSequence(DeviceMatrix& input)
{
	/*cout << "GRU input" << endl;
	PrintMatrix(input);*/

	// Forward through RNN
	auto result = cudnnRNNForwardTraining(CudaContext::cudnnHandle(), _rnnDesc, _seqLen,
		*_xTensor, input.raw_ptr(),
		*_hxTensor, _hxTensor->device_ptr(),
		*_cxTensor, nullptr,
		*_wFilter, _w->weight().raw_ptr(),
		*_yTensor, _output->raw_ptr(),
		*_hyTensor, _hyTensor->device_ptr(),
		*_cyTensor, nullptr,
		_workspace->device_ptr(), _workspace->size(),
		_reserve->device_ptr(), _reserve->size());
	if (result != CUDNN_STATUS_SUCCESS)
	{
		throw CuDnnException(result);
	}

	/*cout << "GRU output" << endl;
	PrintMatrix(*_output);*/
}

void GruLayer::BackpropSequence(DeviceMatrix& input, DeviceMatrix& outSens)
{
	// RNN backprop
	auto result = cudnnRNNBackwardData(CudaContext::cudnnHandle(), _rnnDesc, _seqLen,
		*_yTensor, _output->raw_ptr(),
		*_dyTensor, outSens.raw_ptr(),
		*_dhyTensor, _dhyTensor->device_ptr(),
		*_dcyTensor, nullptr,
		*_wFilter, _w->weight().raw_ptr(),
		*_hxTensor, _hxTensor->device_ptr(),
		*_cxTensor, nullptr,
		*_dxTensor, _sensitivity->raw_ptr(),
		*_dhxTensor, _dhxTensor->device_ptr(),
		*_dcxTensor, nullptr,
		_workspace->device_ptr(), _workspace->size(),
		_reserve->device_ptr(), _reserve->size());
	if (result != CUDNN_STATUS_SUCCESS)
	{
		throw CuDnnException(result);
	}

	_w->ClearGradient();
	result = cudnnRNNBackwardWeights(CudaContext::cudnnHandle(), _rnnDesc, _seqLen,
		*_xTensor, input.raw_ptr(),
		*_hxTensor, _hxTensor->device_ptr(),
		*_yTensor, _output->raw_ptr(),
		_workspace->device_ptr(), _workspace->size(),
		*_dwFilter, _w->gradient().raw_ptr(),
		_reserve->device_ptr(), _reserve->size());
	if (result != CUDNN_STATUS_SUCCESS)
	{
		throw CuDnnException(result);
	}

	Algorithms::Clamp(_w->gradient(), 5.0f);
}

void GruLayer::Optimize(OptimizerBase& optimizer)
{
	optimizer.Optimize(*_w);
}

void GruLayer::ResetMemory()
{
	_hxTensor->ZeroMemory();
	_hyTensor->ZeroMemory();
	_dhxTensor->ZeroMemory();
	_dhyTensor->ZeroMemory();
}

void GruLayer::ResetOptimizerCache()
{
	_w->ClearCache();
}

void GruLayer::InitLayers()
{
	// Allocate RNN in/out tensors and GPU memory
	_xTensor = make_unique<CuDnnNdTensorArray>(_batchSize, _inputSize, 1, _seqLen, false);
	_dxTensor = make_unique<CuDnnNdTensorArray>(_batchSize, _inputSize, 1, _seqLen, false);
	_yTensor = make_unique<CuDnnNdTensorArray>(_batchSize, _hSize, 1, _seqLen, false);
	_dyTensor = make_unique<CuDnnNdTensorArray>(_batchSize, _hSize, 1, _seqLen, false);

	_output = make_unique<DeviceMatrix>(_hSize, _batchSize, _seqLen);
	_sensitivity = make_unique<DeviceMatrix>(_inputSize, _batchSize, _seqLen);
	_output->ZeroMemory();
	_sensitivity->ZeroMemory();

	// Allocate RNN hidden state tensors and GPU memory
	_hxTensor = make_unique<CuDnnNdTensor>(_layers, _batchSize, _hSize);
	_hyTensor = make_unique<CuDnnNdTensor>(_layers, _batchSize, _hSize);
	_dhxTensor = make_unique<CuDnnNdTensor>(_layers, _batchSize, _hSize);
	_dhyTensor = make_unique<CuDnnNdTensor>(_layers, _batchSize, _hSize);
	_cxTensor = make_unique<CuDnnNdTensor>(_layers, _batchSize, _hSize, false);
	_cyTensor = make_unique<CuDnnNdTensor>(_layers, _batchSize, _hSize, false);
	_dcxTensor = make_unique<CuDnnNdTensor>(_layers, _batchSize, _hSize, false);
	_dcyTensor = make_unique<CuDnnNdTensor>(_layers, _batchSize, _hSize, false);

	// Allocate dropout and RNN descriptors
	_dropoutDesc.Create();

	size_t dropoutSz;
	auto result = cudnnDropoutGetStatesSize(CudaContext::cudnnHandle(), &dropoutSz);
	if (result != CUDNN_STATUS_SUCCESS)
	{
		throw CuDnnException(result);
	}

	_dropoutStates = make_unique<CudaMemoryBlock>(dropoutSz);
	result = cudnnSetDropoutDescriptor(_dropoutDesc, CudaContext::cudnnHandle(), 0.0f, _dropoutStates->device_ptr(), dropoutSz, 1337ull);
	if (result != CUDNN_STATUS_SUCCESS)
	{
		throw CuDnnException(result);
	}

	// Create RNN descriptor
	_rnnDesc.Create();
	result = cudnnSetRNNDescriptor(_rnnDesc, _hSize, _layers, _dropoutDesc, CUDNN_LINEAR_INPUT, CUDNN_UNIDIRECTIONAL, CUDNN_GRU, CUDNN_DATA_FLOAT);
	if (result != CUDNN_STATUS_SUCCESS)
	{
		throw CuDnnException(result);
	}

	// Allocate RNN parameters
	size_t weightsSize;
	result = cudnnGetRNNParamsSize(CudaContext::cudnnHandle(), _rnnDesc, (*_xTensor)[0], &weightsSize, CUDNN_DATA_FLOAT);
	if (result != CUDNN_STATUS_SUCCESS)
	{
		throw CuDnnException(result);
	}

	int wLen = (int)(weightsSize / sizeof(float));
	_wFilter = make_unique<CuDnnFilter>(wLen, 1, 1, false);
	_dwFilter = make_unique<CuDnnFilter>(wLen, 1, 1, false);
	_w = make_unique<NeuroWeight>(wLen, 1, 1);
	InitWeights();

	// Allocate workspace and reserve
	size_t wsSize, reserveSize;
	result = cudnnGetRNNWorkspaceSize(CudaContext::cudnnHandle(), _rnnDesc, _seqLen, *_xTensor, &wsSize);
	if (result != CUDNN_STATUS_SUCCESS)
	{
		throw CuDnnException(result);
	}

	result = cudnnGetRNNTrainingReserveSize(CudaContext::cudnnHandle(), _rnnDesc, _seqLen, *_xTensor, &reserveSize);
	if (result != CUDNN_STATUS_SUCCESS)
	{
		throw CuDnnException(result);
	}

	_workspace = make_unique<CudaMemoryBlock>(wsSize);
	_reserve = make_unique<CudaMemoryBlock>(reserveSize);
}

void GruLayer::InitWeights()
{
	for (int i = 0; i < _layers * 12; ++i)
	{
		_weights.push_back(std::unique_ptr<NeuroWeightPtr>());
	}

	cudnnStatus_t result;
	for (int layer = 0; layer < _layers; layer++)
	{
		for (int linLayerID = 0; linLayerID < 6; linLayerID++) { // 6 matrices for GRU
			int matIdx = layer * 12 + linLayerID;
			int bIdx = matIdx + 6;

			cudnnFilterDescriptor_t filterDesc;
			result = cudnnCreateFilterDescriptor(&filterDesc);
			if (result != CUDNN_STATUS_SUCCESS)
			{
				throw CuDnnException(result);
			}

			float *filterMemPtr;

			// Weight matrix
			result = cudnnGetRNNLinLayerMatrixParams(CudaContext::cudnnHandle(), _rnnDesc, layer, (*_xTensor)[0], *_wFilter, _w->weight().raw_ptr(), linLayerID, filterDesc, (void**)&filterMemPtr);
			if (result != CUDNN_STATUS_SUCCESS)
			{
				throw CuDnnException(result);
			}

			_weights[matIdx].reset(GetWeightPtr(filterDesc, filterMemPtr));

			result = cudnnDestroyFilterDescriptor(filterDesc);
			if (result != CUDNN_STATUS_SUCCESS)
			{
				throw CuDnnException(result);
			}

			// Bias vector
			result = cudnnCreateFilterDescriptor(&filterDesc);
			if (result != CUDNN_STATUS_SUCCESS)
			{
				throw CuDnnException(result);
			}

			result = cudnnGetRNNLinLayerBiasParams(CudaContext::cudnnHandle(), _rnnDesc, layer, (*_xTensor)[0], *_wFilter, _w->weight().raw_ptr(), linLayerID, filterDesc, (void**)&filterMemPtr);
			if (result != CUDNN_STATUS_SUCCESS)
			{
				throw CuDnnException(result);
			}

			_weights[bIdx].reset(GetWeightPtr(filterDesc, filterMemPtr));

			result = cudnnDestroyFilterDescriptor(filterDesc);
			if (result != CUDNN_STATUS_SUCCESS)
			{
				throw CuDnnException(result);
			}
		}
	}
}

std::tuple<int, int, int> GruLayer::GetTensorDims(cudnnFilterDescriptor_t desc)
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

	return std::make_tuple(dims[1], dims[0], dims[2]);
}

NeuroWeightPtr* GruLayer::GetWeightPtr(cudnnFilterDescriptor_t tensor, float* weightPtr)
{
	auto dims = GetTensorDims(tensor);
	auto delta = weightPtr - _w->weight().raw_ptr(); // We assume that gradients have the same offset as weights
	float* gradPtr = _w->gradient().raw_ptr() + delta;
	float* cache1Ptr = _w->cache1().raw_ptr() + delta;
	float* cache2Ptr = _w->cache2().raw_ptr() + delta;
	float* cacheMPtr = _w->cache_m().raw_ptr() + delta;
	return new NeuroWeightPtr(std::get<0>(dims), std::get<1>(dims), std::get<2>(dims), weightPtr, gradPtr, cache1Ptr, cache2Ptr, cacheMPtr);
}
