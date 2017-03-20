#include "LayeredNet.h"

#include <algorithm>

void LayeredNet::UpdateLearningRate(float learningRate)
{
	_optimizer->setLearningRate(learningRate);
}

void LayeredNet::TransferStatesToHost(int layer, std::vector<RawMatrixPtr*>& states)
{
	if (layer >= _layers.size())
		throw RetiaException("Index out of range");

	_layers[layer]->TransferStatesToHost(states);
}

double LayeredNet::TrainSequence(std::vector<RawMatrixPtr*>& inputs, std::vector<RawMatrixPtr*>& targets)
{
	if (inputs.size() != _seqLen || targets.size() != _seqLen) throw RetiaException("Wrong number of matrices!");

	auto devInput = DeviceMatrix(_inputSize, _batchSize, _seqLen);
	auto devTarget = DeviceMatrix(_outputSize, _batchSize, _seqLen);

	for (int i = 0; i < inputs.size(); ++i)
	{
		auto inElement = devInput.GetSequenceElement(i);
		auto targElement = devTarget.GetSequenceElement(i);

		inElement.CopyFrom(*inputs[i]);
		targElement.CopyFrom(*targets[i]);
	}

	return TrainSequence(devInput, devTarget);
}

void LayeredNet::AddLayer(LayerBase* layer)
{
	auto inSize = _layers.size() > 0 ? _layers[_layers.size() - 1]->outputSize() : _inputSize;
	if (layer->inputSize() != inSize)
		throw RetiaException("Conflicting layer size!");
	
	_layers.push_back(std::unique_ptr<LayerBase>(layer));
}

double LayeredNet::TrainSequence(DeviceMatrix& input, DeviceMatrix& target)
{
	if (_layers.size() == 0) 
		throw RetiaException("No layers!");
	if (_layers[_layers.size() - 1]->outputSize() != _outputSize)
		throw RetiaException("Last layer out size doesn't match network out size!");
	if (input.rows() != _inputSize || input.columns() != _batchSize || input.seqLength() != _seqLen)
		throw RetiaException("Input matrix dimensions are wrong!");
	if (target.rows() != _outputSize || target.columns() != _batchSize || target.seqLength() != _seqLen)
		throw RetiaException("Output matrix dimensions are wrong!");

	auto curInput = input;
	for (int i = 0; i < _layers.size(); ++i)
	{
		auto layer = _layers[i].get();
		layer->ForwardSequence(curInput);
		curInput = layer->output();
	}

	auto lastLayer = _layers[_layers.size() - 1].get();
	auto error = lastLayer->LayerError(target);
	lastLayer->ErrorPropagate(target);
	auto curSens = lastLayer->sensitivity();

	if (_layers.size() == 1)
		return error;

	for (int i = _layers.size() - 2; i >= 0; --i)
	{
		auto layer = _layers[i].get();
		auto curIn = i > 0 ? _layers[i - 1]->output() : input;
		layer->BackpropSequence(curIn, curSens);
		curSens = layer->sensitivity();
	}

	return error;
}

void LayeredNet::Opimize()
{
	for (int i = 0; i < _layers.size(); ++i)
	{
		_layers[i]->Optimize(*_optimizer);
	}
}

void LayeredNet::ResetMemory()
{
	for (int i = 0; i < _layers.size(); ++i)
	{
		_layers[i]->ResetMemory();
	}
}

void LayeredNet::ResetOptimizerCache()
{
	for (int i = 0; i < _layers.size(); ++i)
	{
		_layers[i]->ResetOptimizerCache();
	}
}