#include "LayeredNetFactoryBase.h"
#include "LayeredNetFactory.h"

std::unique_ptr<LayeredNetFactoryBase> LayeredNetFactoryBase::Create(int inputSize, int outputSize, int batchSize, int seqLen)
{
	return std::unique_ptr<LayeredNetFactoryBase>(static_cast<LayeredNetFactoryBase*>(new LayeredNetFactory(inputSize, outputSize, batchSize, seqLen)));
}
