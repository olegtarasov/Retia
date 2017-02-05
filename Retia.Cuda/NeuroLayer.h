#pragma once

#include <memory>
#include <vector>
#include "Matrix.h"
#include "OptimizerBase.h"
#include "Exceptions.h"
#include "RawMatrixPtr.h"

class NeuroLayer
{
public:

	NeuroLayer(int inSize, int outSize, int batchSize, int seqLen)
		: _inputSize(inSize),
		  _outputSize(outSize),
		  _batchSize(batchSize), 
		  _seqLen(seqLen)
	{
	}

	virtual ~NeuroLayer() = default;


	int inputSize() const
	{
		return _inputSize;
	}

	int outputSize() const
	{
		return _outputSize;
	}

	int batchSize() const
	{
		return _batchSize;
	}

	int seqLen() const
	{
		return _seqLen;
	}

	void TransferOutputToHost(RawMatrixPtr* output) const
	{
		_output->CopyTo(*output);
	}

	virtual void TransferStatesFromHost(std::vector<RawMatrixPtr*>& states) = 0;
	virtual void TransferStatesToHost(std::vector<RawMatrixPtr*>& states) = 0;
	virtual void ForwardSequence(DeviceMatrix& input) = 0;
	virtual void BackpropSequence(DeviceMatrix& input, DeviceMatrix& outSens) = 0;
	virtual void Optimize(OptimizerBase& optimizer) = 0;
	virtual void ErrorPropagate(DeviceMatrix& target) { throw RetiaException("This layer can't propagate the error"); }
	virtual double LayerError(DeviceMatrix& target) { throw RetiaException("This layer can't calculate errors!"); }
	virtual void ResetMemory() = 0;
	virtual void ResetOptimizerCache() = 0;
	
	DeviceMatrix& output() const
	{
		return *_output;
	}
	
	DeviceMatrix& sensitivity() const
	{
		return *_sensitivity;
	}

protected:
	std::unique_ptr<DeviceMatrix>	_output;
	std::unique_ptr<DeviceMatrix>	_sensitivity;
	
	int		_inputSize;
	int		_outputSize;
	int		_batchSize;
	int		_seqLen;
};
