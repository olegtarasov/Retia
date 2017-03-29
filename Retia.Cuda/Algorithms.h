#pragma once

#include <thrust/transform.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

#define WEIGHT(t) thrust::get<0>(t)
#define GRAD(t) thrust::get<1>(t)
#define CACHE1(t) thrust::get<2>(t)
#define CACHE2(t) thrust::get<3>(t)
#define CACHEM(t) thrust::get<4>(t)

struct CrossEntropyPropagationFunctor : public thrust::binary_function<float, float, float>
{
	const int _batchSize;

	CrossEntropyPropagationFunctor(int batchSize)
		: _batchSize(batchSize)
	{
	}

	__host__ __device__
	float operator()(const float& y, const float& t) const
	{
		return (y - t) / _batchSize;
	}
};

struct CrossEntropyErrorFunctor
{
	template <typename Tuple>
	__host__ __device__
	float operator()(Tuple t)
	{
		return log(thrust::get<0>(t)) * thrust::get<1>(t);
	}

};

struct RMSPropFunctor
{
	const float _learningRate, _decayRate, _momentum, _weightDecay;

	RMSPropFunctor(float learningRate, float decayRate, float momentum, float weightDecay)
		: _learningRate(learningRate),
		_decayRate(decayRate),
		_momentum(momentum),
		_weightDecay(weightDecay)
	{
	}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		const float e = 1e-4f;

		// cache2 = _decayRate * cache2 + (1 - _decayRate) * grad^2
		CACHE2(t) = _decayRate * CACHE2(t) + (1 - _decayRate) * GRAD(t) * GRAD(t);

		// cache1 = _decayRate * cache1 + (1 - _decayRate) * grad
		CACHE1(t) = _decayRate * CACHE1(t) + (1 - _decayRate) * GRAD(t);

		// cacheM = _momentum * cacheM - _learningRate * grad / sqrt(cache2 - cache1^2 + e)
		CACHEM(t) = _momentum * CACHEM(t) - _learningRate * GRAD(t) / sqrt(CACHE2(t) - CACHE1(t) * CACHE1(t) + e);

		// weight = weight + cacheM - _learningRate * _weightDecay * weight
		WEIGHT(t) = WEIGHT(t) + CACHEM(t) - _learningRate * _weightDecay * WEIGHT(t);
	}

};

struct ClampFunctor : public thrust::unary_function<float, float>
{
	const float _clamp;

	ClampFunctor(float clamp) : _clamp(clamp)
	{
	}

	__host__ __device__
	float operator()(float value)
	{
		if (value < -_clamp)
			return -_clamp;
		if (value > _clamp)
			return _clamp;

		return value;
	}

};

class Algorithms
{
public:
	template <class TMatrix>
	static void BackpropagateCrossEntropyError(TMatrix& output, TMatrix& target, TMatrix& result)
	{
		thrust::transform(output.begin(), output.end(), target.begin(), result.begin(), CrossEntropyPropagationFunctor(target.columns()));
	}

	template <class TMatrix> 
	static double CrossEntropyError(TMatrix& output, TMatrix& target)
	{
		double err = thrust::transform_reduce(
			thrust::make_zip_iterator(thrust::make_tuple(output.begin(), target.begin())),
			thrust::make_zip_iterator(thrust::make_tuple(output.end(), target.end())),
			CrossEntropyErrorFunctor(),
			0.0,
			thrust::plus<double>());

		return -err / (output.columns() * output.seqLength());
	}

	template <class TMatrix>
	static void RMSPropOptimize(TMatrix& weight, TMatrix& gradient, TMatrix& cache1, TMatrix& cache2, TMatrix& cacheM,
								float learningRate, float decayRate, float momentum, float weightDecay)
	{
		thrust::for_each(
			thrust::make_zip_iterator(thrust::make_tuple(weight.begin(), gradient.begin(), cache1.begin(), cache2.begin(), cacheM.begin())),
			thrust::make_zip_iterator(thrust::make_tuple(weight.end(), gradient.end(), cache1.end(), cache2.end(), cacheM.end())),
			RMSPropFunctor(learningRate, decayRate, momentum, weightDecay));
	}

	template <class TMatrix>
	static void Clamp(TMatrix& matrix, float value)
	{
		thrust::transform(matrix.begin(), matrix.end(), matrix.begin(), ClampFunctor(value));
	}
};