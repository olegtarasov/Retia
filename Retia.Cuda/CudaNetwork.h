#pragma once

#include <memory>

#include "CuDnnTensor.h"
#include "GpuMatrix.h"
#include "CudaMemoryBlock.h"
#include "CudaHandle.h"

class CudaNetwork
{
public:
	CudaNetwork(int inputSize, int hSize, int outSize, int seqLen, int batchSize);

	void Initialize();
	void ForwardPass(float* input, float* output, float* linW);
	void Backpropagate(float* ySens);
private:
	int _inputSize;
	int _hSize;
	int _outSize;
	int _seqLen;
	int _batchSize;
	int _layers = 1;

	// In/out tensors
	std::unique_ptr<CuDnnNdTensorArray>		_xTensor, _yTensor, 
											_dxTensor, _dyTensor;

	// State tensors
	std::unique_ptr<CuDnnNdTensor>			_hxTensor, _cxTensor,
											_hyTensor, _cyTensor, 
											_dhxTensor, _dcxTensor, 
											_dhyTensor, _dcyTensor;
	
	// RNN Weigths
	std::unique_ptr<CuDnnFilter>			_wFilter, _dwFilter;

	// Linear weigths
	std::unique_ptr<GpuMatrix>				_linW;

	// Out matrix
	std::unique_ptr<GpuMatrix>				_linOut;
	std::unique_ptr<CuDnnNdTensor>			_outTensor;

	// Backprop
	std::unique_ptr<GpuMatrix>				_linWGrad, _linBGrad, _linYIdentity;
	
	// Dropout
	std::unique_ptr<CudaMemoryBlock>	_dropoutStates;
	CuDnnDropoutDescriptor				_dropoutDesc;

	// RNN desc
	CuDnnRnnDescriptor					_rnnDesc;

	// Workspace GPU memory
	std::unique_ptr<CudaMemoryBlock>	_workspace, _reserve;

	// Handles
	static thread_local std::unique_ptr<CuDnnHandle>	_cudnnHandle;
	static thread_local std::unique_ptr<CuBlasHandle>	_cuBlasHandle;

	void FillWithRandom(float arr[], int len);
	void FillStrides(int dims[], int strides[], int cnt);

	template <class T> static void CheckHandleCreated(std::unique_ptr<T> *ptr);
	static void CheckCudnnCreated();
	static void CheckBlasCreated();
};

