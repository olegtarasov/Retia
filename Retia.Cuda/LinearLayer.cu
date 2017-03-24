#include "LinearLayer.h"
#include "CudaContext.h"
#include <iostream>
#include "Helpers.h"

using std::cout;
using std::endl;

LinearLayer::LinearLayer(int inSize, int outSize, int batchSize, int seqLength): LayerBase(inSize, outSize, batchSize, seqLength)
{
	_w = std::make_unique<NeuroWeight>(outSize, inSize);
	_b = std::make_unique<NeuroWeight>(outSize, 1, 1);

	_output = std::make_unique<DeviceMatrix>(outSize, batchSize, seqLength);
	_sensitivity = std::make_unique<DeviceMatrix>(inSize, batchSize, seqLength);
	_identity = std::make_unique<DeviceMatrix>(batchSize, 1, 1);
}

void LinearLayer::TransferStatesToDevice(std::vector<WeightSyncContainer*>& states)
{
	if (states.size() != 2) throw RetiaException("State vector should have the length of exactly 2");

	_w->TransferStateToDevice(*states[0]);
	_b->TransferStateToDevice(*states[1]);
}

void LinearLayer::TransferStatesToHost(std::vector<WeightSyncContainer*>& states)
{
	if (states.size() != 2) throw RetiaException("State vector should have the length of exactly 2");

	_w->TransferStateToHost(*states[0]);
	_b->TransferStateToHost(*states[1]);
}

void LinearLayer::ForwardSequence(DeviceMatrix& input)
{
	/*cout << "Linear input" << endl;
	PrintMatrix(input);*/

	_output->TileFrom(_b->weight());
	_output->Accumulate(_w->weight(), input, 1.0f);

	/*cout << "Linear output" << endl;
	PrintMatrix(*_output);*/
}

void LinearLayer::BackpropSequence(DeviceMatrix& input, DeviceMatrix& outSens)
{
	_w->gradient().ZeroMemory();
	_b->gradient().ZeroMemory();
	_sensitivity->ZeroMemory();

	for (int i = _seqLen - 1; i >= 0; --i)
	{
		auto curOutSens = outSens.GetSequenceElement(i);
		auto curInput = input.GetSequenceElement(i);
		auto curInSens = _sensitivity->GetSequenceElement(i);
		

		_w->gradient().Accumulate(curOutSens, curInput, 1.0f, 1.0f, CUBLAS_OP_N, CUBLAS_OP_T);
		_b->gradient().Accumulate(curOutSens, *_identity, 1.0f);
	}

	_sensitivity->Accumulate(_w->weight(), outSens, 0.0f, 1.0f, CUBLAS_OP_T);
}

void LinearLayer::Optimize(OptimizerBase& optimizer)
{
	optimizer.Optimize(*_w);
	optimizer.Optimize(*_b);
}

void LinearLayer::ResetMemory()
{
}

void LinearLayer::ResetOptimizerCache()
{
	_w->ClearCache();
	_b->ClearCache();
}
