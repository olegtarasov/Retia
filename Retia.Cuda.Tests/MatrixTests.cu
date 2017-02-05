#include <gtest/gtest.h>
#include "Matrix.h"

TEST(MatrixTests, CanZeroMemory)
{
	auto matrix = HostMatrix(2, 2, 2);
	auto ptr = matrix.raw_ptr();

	ASSERT_EQ(matrix.length(), 8);

	for (int i = 0; i < matrix.length(); ++i)
	{
		ptr[i] = static_cast<float>(i);
	}

	matrix.ZeroMemory();

	for (int i = 0; i < matrix.length(); ++i)
	{
		ASSERT_EQ(ptr[i], 0.0f);
	}
}

TEST(MatrixTests, CanCopy)
{
	auto matrix1 = HostMatrix(2, 2, 2);
	auto matrix2 = HostMatrix(2, 2, 2);
	auto ptr1 = matrix1.raw_ptr();
	auto ptr2 = matrix2.raw_ptr();

	for (int i = 0; i < matrix1.length(); ++i)
	{
		ptr1[i] = static_cast<float>(i);
	}

	matrix2.CopyFrom(matrix1);

	for (int i = 0; i < matrix2.length(); ++i)
	{
		ASSERT_EQ(ptr2[i], static_cast<float>(i));
	}
}

TEST(MatrixTests, CanCopyLoose)
{
	auto matrix1 = HostMatrix(2, 2, 2);
	auto matrix2 = HostMatrix(4, 2, 1);
	auto ptr1 = matrix1.raw_ptr();
	auto ptr2 = matrix2.raw_ptr();

	for (int i = 0; i < matrix1.length(); ++i)
	{
		ptr1[i] = static_cast<float>(i);
	}

	matrix2.CopyFromLoose(matrix1);

	for (int i = 0; i < matrix2.length(); ++i)
	{
		ASSERT_EQ(ptr2[i], static_cast<float>(i));
	}
}

TEST(MatrixTests, CanGetSequenceElement)
{
	auto matrix = HostMatrix(2, 2, 2);
	auto ptr = matrix.raw_ptr();

	ASSERT_EQ(matrix.length(), 8);

	for (int i = 0; i < matrix.length(); ++i)
	{
		ptr[i] = static_cast<float>(i);
	}

	auto el1 = matrix.GetSequenceElement(0);
	
	ASSERT_EQ(el1.length(), 4);
	ASSERT_EQ(el1.seqLength(), 1);
	
	ptr = el1.raw_ptr();

	for (int i = 0; i < el1.length(); ++i)
	{
		ASSERT_EQ(ptr[i], static_cast<float>(i));
	}

	auto el2 = matrix.GetSequenceElement(1);

	ASSERT_EQ(el2.length(), 4);
	ASSERT_EQ(el2.seqLength(), 1);

	ptr = el2.raw_ptr();

	for (int i = 0; i < el2.length(); ++i)
	{
		ASSERT_EQ(ptr[i], static_cast<float>(i + 4));
	}
}

TEST(MatrixTests, CanGetAndWriteSequenceElement)
{
	auto matrix = HostMatrix(2, 2, 2);
	auto ptr = matrix.raw_ptr();

	matrix.ZeroMemory();

	auto el1 = matrix.GetSequenceElement(0);

	ASSERT_EQ(el1.length(), 4);
	ASSERT_EQ(el1.seqLength(), 1);

	ptr = el1.raw_ptr();

	for (int i = 0; i < el1.length(); ++i)
	{
		ptr[i] = static_cast<float>(i);
	}

	auto el2 = matrix.GetSequenceElement(1);

	ASSERT_EQ(el2.length(), 4);
	ASSERT_EQ(el2.seqLength(), 1);

	ptr = el2.raw_ptr();

	for (int i = 0; i < el1.length(); ++i)
	{
		ptr[i] = static_cast<float>(i);
	}

	ptr = matrix.raw_ptr();
	ASSERT_EQ(ptr[0], 0);
	ASSERT_EQ(ptr[1], 1); 
	ASSERT_EQ(ptr[2], 2);
	ASSERT_EQ(ptr[3], 3);
	ASSERT_EQ(ptr[4], 0);
	ASSERT_EQ(ptr[5], 1);
	ASSERT_EQ(ptr[6], 2);
	ASSERT_EQ(ptr[7], 3);
}

TEST(MatrixTests, CanTileTo)
{
	auto column = HostMatrix(2, 1, 1);
	auto matrix = HostMatrix(2, 2, 2);
	auto ptr = column.raw_ptr();

	for (int i = 0; i < column.length(); ++i)
	{
		ptr[i] = static_cast<float>(i);
	}

	matrix.TileFrom(column);

	ASSERT_EQ(matrix.seqLength(), 2);

	for (int seq = 0; seq < matrix.seqLength(); ++seq)
	{
		auto el = matrix.GetSequenceElement(seq);

		ASSERT_EQ(el.columns(), 2);

		ptr = el.raw_ptr();
		for (int col = 0; col < el.columns(); ++col)
		{
			for (int row = 0; row < el.rows(); ++row)
			{
				ASSERT_EQ(ptr[col * el.rows() + row], static_cast<float>(row));
			}
		}
	}
}