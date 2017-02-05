#include <iostream>
#include <gtest/gtest.h>
#include "Matrix.h"
#include "Helpers.h"
#include "Algorithms.h"

using std::cout;
using std::endl;

TEST(AlgorithmsTests, CanPropagateError)
{
	auto output = HostMatrix(2, 2, 2);
	auto target = HostMatrix(2, 2, 2);
	auto result = HostMatrix(2, 2, 2);

	InitMatrix(output, new float[8] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f });
	InitMatrix(target, new float[8]{ 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f });

	cout << "Output:" << endl;
	PrintMatrix(output);

	cout << "Target:" << endl;
	PrintMatrix(target);
	
	Algorithms::PropagateError(output, target, result);

	cout << "Result" << endl;
	PrintMatrix(result);

	auto ptr = result.raw_ptr();
	ASSERT_EQ(ptr[0], -3.5f);
	ASSERT_EQ(ptr[1], -2.5f);
	ASSERT_EQ(ptr[2], -1.5f);
	ASSERT_EQ(ptr[3], -0.5f);
	ASSERT_EQ(ptr[4], 0.5f);
	ASSERT_EQ(ptr[5], 1.5f);
	ASSERT_EQ(ptr[6], 2.5f);
	ASSERT_EQ(ptr[7], 3.5f);
}

TEST(AlgorithmsTests, CanCalculateCrossEntropyError)
{
	auto output = HostMatrix(2, 2, 2);
	auto target = HostMatrix(2, 2, 2);
	
	InitMatrix(output, new float[8]{ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f });
	InitMatrix(target, new float[8]{ 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f });

	cout << "Output:" << endl;
	PrintMatrix(output);

	cout << "Target:" << endl;
	PrintMatrix(target);

	auto result = Algorithms::CrossEntropyError(output, target);
	auto ref = (
		(-(log(1.0f) * 8.0f) / 2) +
		(-(log(2.0f) * 7.0f) / 2) +
		(-(log(3.0f) * 6.0f) / 2) +
		(-(log(4.0f) * 5.0f) / 2) +
		(-(log(5.0f) * 4.0f) / 2) +
		(-(log(6.0f) * 3.0f) / 2) +
		(-(log(7.0f) * 2.0f) / 2) +
		(-(log(8.0f) * 1.0f) / 2)
		) / 2;

	cout << "Result: " << result << endl;
	cout << "Ref: " << ref;

	ASSERT_EQ(result, ref);
}