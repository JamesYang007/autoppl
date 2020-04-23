#include <cmath>
#include <array>
#include "gtest/gtest.h"
#include <autoppl/expression/variable/binop.hpp>
#include <testutil/mock_types.hpp>

namespace ppl {
namespace expr {

//////////////////////////////////////////////////////
// Model with one RV TESTS
//////////////////////////////////////////////////////

/*
 * Mock binary operation node for testing purposes.
 */
struct MockBinaryOp
{
	// mock operation -- returns the sum 
	static double evaluate(double x, double y) {
		return x + y;
	}

};

struct binop_fixture : ::testing::Test
{
protected:
	MockVarExpr x = 0;
	MockVarExpr y = 0;

	using binop_result_t = double;
	
	using binop_node_t = BinaryOpNode<MockBinaryOp, MockVarExpr, MockVarExpr>;
	
	void reconfigureX(double val)
	{ x.set_value(val); }

	void reconfigureY(double val)
	{ y.set_value(val); }

};

TEST_F(binop_fixture, add)
{
	double val1 = 3.5;
	double val2 = 4.5;
	
	double addDouble = AddOp::evaluate(val1, val2);
	EXPECT_DOUBLE_EQ(addDouble, 8.0);

	int addInt = AddOp::evaluate(3, 4);
	EXPECT_EQ(addInt, 7);
}

TEST_F(binop_fixture, sub)
{
	double val1 = 3.5;
	double val2 = 4.5;
	
	double subDouble = SubOp::evaluate(val1, val2);
	EXPECT_DOUBLE_EQ(subDouble, -1.0);

	int subInt = SubOp::evaluate(3, 4);
	EXPECT_EQ(subInt, -1);
}


TEST_F(binop_fixture, mult)
{
	double val1 = 3.5;
	double val2 = 4.5;
	
	double multDouble = MultOp::evaluate(val1, val2);
	EXPECT_DOUBLE_EQ(multDouble, 15.75);

	int multInt = MultOp::evaluate(3, 4);
	EXPECT_EQ(multInt, 12);
}

TEST_F(binop_fixture, div)
{
	double val1 = 4.5;
	double val2 = 0.75;
	
	double divDouble = DivOp::evaluate(val1, val2);
	EXPECT_DOUBLE_EQ(divDouble, 6.0);

	int divInt = DivOp::evaluate(12, 3);
	EXPECT_EQ(divInt, 4);
}

TEST_F(binop_fixture, binop_node)
{
	reconfigureX(3);
	reconfigureY(4);

	binop_node_t addNode = {x, y};
	double res = addNode.get_value();

	EXPECT_EQ(res, 7);

}

} // namespace expr
} // namespace ppl
