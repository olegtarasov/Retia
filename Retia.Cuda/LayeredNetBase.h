#pragma once

#include <memory>
#include "RawMatrixPtr.h"

class LayeredNetBase
{
public:
	LayeredNetBase(int input_size, int output_size, int batch_size, int seqLen)
		: _inputSize(input_size),
		_outputSize(output_size),
		_batchSize(batch_size),
		_seqLen(seqLen)
	{
	}

	virtual ~LayeredNetBase() = default;


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

	virtual double TrainSequence(std::vector<RawMatrixPtr*>& inputs, std::vector<RawMatrixPtr*>& targets) = 0;
	virtual void Opimize() = 0;
	virtual void ResetMemory() = 0;
	virtual void ResetOptimizerCache() = 0;

protected:
	int	_inputSize, _outputSize, _batchSize, _seqLen;
};
