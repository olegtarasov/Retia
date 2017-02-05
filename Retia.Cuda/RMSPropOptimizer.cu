#include "RMSPropOptimizer.h"

#include <thrust/transform.h>
#include <thrust/for_each.h>
#include "Algorithms.h"

using namespace thrust;

void RMSPropOptimizer::Optimize(NeuroWeigth& weigth)
{
	Algorithms::RMSPropOptimize(weigth.weight(), weigth.gradient(), weigth.cache1(), weigth.cache2(), weigth.cache_m(), _learningRate, _decayRate, _momentum, _weightDecay);
}
