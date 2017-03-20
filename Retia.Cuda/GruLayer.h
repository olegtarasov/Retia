#pragma once
#include "LayerBase.h"
#include "CuDnnTensor.h"
#include "CudaHandle.h"
#include "RawMatrixPtr.h"

class GruLayer : public LayerBase
{
public:
	GruLayer(int inputSize, int hSize, int layers, int batchSize, int seqLen);


	void TransferStatesFromHost(std::vector<RawMatrixPtr*>& states) override;
	void TransferStatesToHost(std::vector<RawMatrixPtr*>& states) override;
	void ForwardSequence(DeviceMatrix& input) override;
	void BackpropSequence(DeviceMatrix& input, DeviceMatrix& outSens) override;
	void Optimize(OptimizerBase& optimizer) override;
	void ResetMemory() override;
	void ResetOptimizerCache() override;

private:
	int _hSize;
	int _layers;

	// In/out tensors
	std::unique_ptr<CuDnnNdTensorArray>		_xTensor, _yTensor,
											_dxTensor, _dyTensor;

	// State tensors
	std::unique_ptr<CuDnnNdTensor>			_hxTensor, _cxTensor,
		_hyTensor, _cyTensor,
		_dhxTensor, _dcxTensor,
		_dhyTensor, _dcyTensor;

	// RNN Weigths
	std::unique_ptr<CuDnnFilter>		_wFilter, _dwFilter;
	std::unique_ptr<NeuroWeigth>		_w;

	// Dropout
	std::unique_ptr<CudaMemoryBlock>	_dropoutStates;
	CuDnnDropoutDescriptor				_dropoutDesc;

	// RNN desc
	CuDnnRnnDescriptor					_rnnDesc;

	// Workspace GPU memory
	std::unique_ptr<CudaMemoryBlock>	_workspace, _reserve;

	void InitLayers();

	/*
	* States indexes:
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
	void TransferState(std::vector<RawMatrixPtr*>& states, bool hostToDevice);
};
