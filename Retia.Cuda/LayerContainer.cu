#include "LayerContainer.h"
#include "GruLayer.h"
#include "Helpers.h"
#include <iostream>

using std::cout;
using std::endl;

void LayerContainer::ForwardSequence(std::vector<RawMatrixPtr*>& input)
{
	auto deviceInput = DeviceMatrix(input[0]->rows(), input[0]->columns(), input.size());

	for (int i = 0; i < input.size(); i++)
	{
		auto element = deviceInput.GetSequenceElement(i);

		element.CopyFrom(*input[i]);
	}

	/*cout << ">> gpu input" << endl;
	PrintMatrix(deviceInput);
	cout << "<<" << endl;*/

	_layer->ForwardSequence(deviceInput);
}

void LayerContainer::TransferOutputToHost(std::vector<RawMatrixPtr*>& output)
{
	if (output.size() != _layer->seqLen()) throw RetiaException("Wrong number of matrices for layer sequence length!");

	/*cout << ">> gpu output" << endl;
	PrintMatrix(_layer->output());
	cout << "<<" << endl;*/

	for (int i = 0; i < output.size(); i++)
	{
		auto element = _layer->output().GetSequenceElement(i);
		element.CopyTo(*output[i]);
	}
}
