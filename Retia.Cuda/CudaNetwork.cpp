#include "CudaNetwork.h"

#include <random>

using std::make_unique;
using std::unique_ptr;

thread_local unique_ptr<CuDnnHandle> CudaNetwork::_cudnnHandle;
thread_local unique_ptr<CuBlasHandle> CudaNetwork::_cuBlasHandle;

CudaNetwork::CudaNetwork(int inputSize, int hSize, int outSize, int seqLen, int batchSize)
{
	_inputSize = inputSize;
	_hSize = hSize;
	_outSize = outSize;
	_seqLen = seqLen;
	_batchSize = batchSize;
}

void CudaNetwork::ForwardPass(float* input, float* output, float* linW)
{
	CheckCudnnCreated();
	CheckBlasCreated();

	_xTensor->CopyToDevice(input);
	_linW->set_host_ptr(linW);
	_linW->CopyToDevice();

	// We store tiled bias vector in _linOutDev, where final forward result will be placed.
	_linOut->set_host_ptr(output);
	_linOut->CopyToDevice();
	_yTensor->ZeroMemory();

	// Forward through RNN
	auto result = cudnnRNNForwardTraining(*_cudnnHandle, _rnnDesc, _seqLen,
		*_xTensor, _xTensor->device_ptr(),
		*_hxTensor, _hxTensor->device_ptr(),
		*_cxTensor, _cxTensor->device_ptr(),
		*_wFilter, _wFilter->device_ptr(),
		*_yTensor, _yTensor->device_ptr(),
		*_hyTensor, _hyTensor->device_ptr(),
		*_cyTensor, _cyTensor->device_ptr(),
		_workspace->device_ptr(), _workspace->size(),
		_reserve->device_ptr(), _reserve->size());
	if (result == CUDNN_STATUS_SUCCESS)
	{
		throw CuDnnException(result);
	}
	
	auto syncResut = cudaDeviceSynchronize();
	if (syncResut != cudaSuccess)
	{
		throw CudaException(syncResut);
	}

	// Forward through lin layer
	auto yMat = _yTensor->AsGpuMatrix(1, 0);
	_linOut->AccumulateOnDevice(*_linW, yMat, 1.0f);
	cudaDeviceSynchronize();

	// Apply softmax
	float alpha = 1.0f;
	float beta = 0.0f;
	for (int i = 0; i < _seqLen; i++)
	{
		auto curOut = _linOut->GetSequenceElement(i)->device_ptr();
		result = cudnnSoftmaxForward(*_cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha, *_outTensor, curOut, &beta, *_outTensor, curOut);
		if (result == CUDNN_STATUS_SUCCESS)
		{
			throw CuDnnException(result);
		}
	}

	// Copy output to host
	_linOut->CopyToHost();
}

void CudaNetwork::Backpropagate(float* ySens)
{
	CheckCudnnCreated();
	CheckBlasCreated();

	// We assume x and y's have not changed since the forward pass.
	// Backprop through lin layer
	_linBGrad->ZeroMemory();
	_linWGrad->ZeroMemory();
	_dxTensor->ZeroMemory();
	_dyTensor->ZeroMemory();
	_dhxTensor->ZeroMemory();
	_dhyTensor->ZeroMemory();

	_linOut->set_host_ptr(ySens);
	_linOut->CopyToDevice();

	auto y = _yTensor->AsGpuMatrix(1, 0);
	auto dy = _dyTensor->AsGpuMatrix(1, 0);
	for (int i = _seqLen - 1; i >= 0; i--)
	{
		auto curSens = _linOut->GetSequenceElement(i);
		auto curInput = y.GetSequenceElement(i);
		auto curDy = dy.GetSequenceElement(i);

		_linWGrad->AccumulateOnDevice(*curSens, *curInput, 1.0f, 1.0f, CUBLAS_OP_N, CUBLAS_OP_T);
		if (_batchSize > 1)
		{
			_linBGrad->AccumulateOnDevice(*curSens, *_linYIdentity, 1.0f);
		}
		else
		{
			// TODO: Fix
			throw RetiaException("Batch size 1 not supported.");
			//_linBGrad.AsMatrix().Accumulate(sens[i]); // Ok, we are doing it in host memory right away
		}

		curDy->AccumulateOnDevice(*_linW, *curSens, 1.0f, 1.0f, CUBLAS_OP_T);
	}

	// TODO: Copy lin layer grads to host

	// RNN backprop
	auto result = cudnnRNNBackwardData(*_cudnnHandle, _rnnDesc, _seqLen,
		*_yTensor, _yTensor->device_ptr(),
		*_dyTensor, _dyTensor->device_ptr(),
		*_dhyTensor, _dhyTensor->device_ptr(),
		*_dcyTensor, _dcyTensor->device_ptr(),
		*_wFilter, _wFilter->device_ptr(),
		*_hxTensor, _hxTensor->device_ptr(),
		*_cxTensor, _cxTensor->device_ptr(),
		*_dxTensor, _dxTensor->device_ptr(),
		*_dhxTensor, _dhxTensor->device_ptr(),
		*_dcxTensor, _dcxTensor->device_ptr(),
		_workspace->device_ptr(), _workspace->size(),
		_reserve->device_ptr(), _reserve->size());
	if (result != CUDNN_STATUS_SUCCESS)
	{
		throw CuDnnException(result);
	}

	auto cudaResult = cudaDeviceSynchronize();
	if (cudaResult != cudaSuccess)
	{
		throw CudaException(cudaResult);
	}

	_dwFilter->ZeroMemory();
	result = cudnnRNNBackwardWeights(*_cudnnHandle, _rnnDesc, _seqLen,
		*_xTensor, _xTensor->device_ptr(),
		*_hxTensor, _hxTensor->device_ptr(),
		*_yTensor, _yTensor->device_ptr(),
		_workspace->device_ptr(), _workspace->size(),
		*_dwFilter, _dwFilter->device_ptr(),
		_reserve->device_ptr(), _reserve->size());
	if (result != CUDNN_STATUS_SUCCESS)
	{
		throw CuDnnException(result);
	}

	cudaResult = cudaDeviceSynchronize();
	if (cudaResult != cudaSuccess)
	{
		throw CudaException(cudaResult);
	}
}

void CudaNetwork::Initialize()
{
	CheckCudnnCreated();

	// Allocate RNN in/out tensors and GPU memory
	_xTensor = make_unique<CuDnnNdTensorArray>(_batchSize, _inputSize, 1, _seqLen);
	_dxTensor = make_unique<CuDnnNdTensorArray>(_batchSize, _inputSize, 1, _seqLen);
	_yTensor = make_unique<CuDnnNdTensorArray>(_batchSize, _hSize, 1, _seqLen);
	_dyTensor = make_unique<CuDnnNdTensorArray>(_batchSize, _hSize, 1, _seqLen);
	
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
	auto result = cudnnDropoutGetStatesSize(*_cudnnHandle, &dropoutSz);
	if (result != CUDNN_STATUS_SUCCESS)
	{
		throw CuDnnException(result);
	}
	
	_dropoutStates = make_unique<CudaMemoryBlock>(dropoutSz);
	result = cudnnSetDropoutDescriptor(_dropoutDesc, *_cudnnHandle, 0.0f, _dropoutStates->device_ptr(), dropoutSz, 1337ull);
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
	result = cudnnGetRNNParamsSize(*_cudnnHandle, _rnnDesc, (*_xTensor)[0], &weightsSize, CUDNN_DATA_FLOAT);
	if (result != CUDNN_STATUS_SUCCESS)
	{
		throw CuDnnException(result);
	}

	_wFilter = make_unique<CuDnnFilter>((int)(weightsSize / sizeof(float)), 1, 1, true);
	_dwFilter = make_unique<CuDnnFilter>((int)(weightsSize / sizeof(float)), 1, 1, true);

	// Allocate workspace and reserve
	size_t wsSize, reserveSize;
	result = cudnnGetRNNWorkspaceSize(*_cudnnHandle, _rnnDesc, _seqLen, *_xTensor, &wsSize);
	if (result != CUDNN_STATUS_SUCCESS)
	{
		throw CuDnnException(result);
	}
	
	result = cudnnGetRNNTrainingReserveSize(*_cudnnHandle, _rnnDesc, _seqLen, *_xTensor, &reserveSize);
	if (result != CUDNN_STATUS_SUCCESS)
	{
		throw CuDnnException(result);
	}

	_workspace = make_unique<CudaMemoryBlock>(wsSize);
	_reserve = make_unique<CudaMemoryBlock>(reserveSize);

	// Init weights one matrix at a time
	for (int layer = 0; layer < _layers; layer++) {
		for (int linLayerID = 0; linLayerID < 6; linLayerID++) { // 6 matrices for GRU
			cudnnFilterDescriptor_t filterDesc;
			result = cudnnCreateFilterDescriptor(&filterDesc);
			if (result != CUDNN_STATUS_SUCCESS)
			{
				throw CuDnnException(result);
			}

			float *filterMemPtr;

			// Weight matrix
			result = cudnnGetRNNLinLayerMatrixParams(*_cudnnHandle, _rnnDesc, layer, (*_xTensor)[0], *_wFilter, _wFilter->device_ptr(), linLayerID, filterDesc, (void**)&filterMemPtr);
			if (result != CUDNN_STATUS_SUCCESS)
			{
				throw CuDnnException(result);
			}
			
			// Fill with random
			auto matrix = GpuMatrix(filterDesc);
			matrix.AllocateOnHost();
			matrix.set_device_ptr(filterMemPtr);
			FillWithRandom(matrix.host_ptr(), matrix.length());
			matrix.CopyToHost();
					
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
			
			result = cudnnGetRNNLinLayerBiasParams(*_cudnnHandle, _rnnDesc, layer, (*_xTensor)[0], *_wFilter, _wFilter->device_ptr(), linLayerID, filterDesc, (void**)&filterMemPtr);
			if (result != CUDNN_STATUS_SUCCESS)
			{
				throw CuDnnException(result);
			}

			// Fill with random
			matrix = GpuMatrix(filterDesc);
			matrix.AllocateOnHost();
			matrix.set_device_ptr(filterMemPtr);
			FillWithRandom(matrix.host_ptr(), matrix.length());
			matrix.CopyToHost();

			result = cudnnDestroyFilterDescriptor(filterDesc);
			if (result != CUDNN_STATUS_SUCCESS)
			{
				throw CuDnnException(result);
			}
		}
	}

	// Init linear layer
	_linW = make_unique<GpuMatrix>(_outSize, _hSize, 1, false);
	_linOut = make_unique<GpuMatrix>(_outSize, _batchSize, _seqLen, false);
	
	// Single output matrix
	_outTensor = make_unique<CuDnnNdTensor>(_batchSize, _outSize, 1, false);

	// Init backprop
	_linWGrad = make_unique<GpuMatrix>(_outSize, _hSize);
	_linBGrad = make_unique<GpuMatrix>(_outSize, 1);
	_linYIdentity = make_unique<GpuMatrix>(_batchSize, 1);

	auto linYPtr = _linYIdentity->host_ptr();
	for (int i = 0; i < _linYIdentity->length(); ++i)
	{
		linYPtr[i] = 1.0f;
	}

	_linYIdentity->CopyToDevice();
}

void CudaNetwork::FillWithRandom(float arr[], int len)
{
	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_real_distribution<float> dist(-5e-2f, 5e-2f);

	for (int i = 0; i < len; i++)
	{
		arr[i] = dist(mt);
	}
}

void CudaNetwork::FillStrides(int dims[], int strides[], int cnt)
{
	for (int i = cnt - 1; i >= 0; i--)
	{
		int stride = 1;
		for (int j = 0; j < cnt - i - 1; j++)
		{
			stride *= dims[cnt - j - 1];
		}

		strides[i] = stride;
	}
}

void CudaNetwork::CheckCudnnCreated()
{
	CheckHandleCreated(&_cudnnHandle);
	_cudnnHandle->CheckCreated();
}

void CudaNetwork::CheckBlasCreated()
{
	CheckHandleCreated(&_cuBlasHandle);
	_cuBlasHandle->CheckCreated();
}


template <class T>
void CudaNetwork::CheckHandleCreated(unique_ptr<T>* ptr)
{
	if (*ptr == nullptr)
	{
		*ptr = make_unique<T>();
	}
}
