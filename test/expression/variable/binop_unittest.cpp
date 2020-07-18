#include <cmath>
#include <array>
#include "gtest/gtest.h"
#include <autoppl/expression/variable/binop.hpp>
#include <autoppl/util/traits/mock_types.hpp>

namespace ppl {
namespace expr {

//////////////////////////////////////////////////////
// Model with one RV TESTS
//////////////////////////////////////////////////////

struct binop_fixture : ::testing::Test
{
protected:
	using addop_node_t = BinaryOpNode<AddOp, MockVarExpr, MockVarExpr>;
};

//////////////////////////////////////////////////////
// Functor TESTS
//////////////////////////////////////////////////////

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

//////////////////////////////////////////////////////
// Binop Node TESTS
//////////////////////////////////////////////////////

TEST_F(binop_fixture, binop_node_value)
{
    addop_node_t node(MockVarExpr(3), MockVarExpr(4));
    // first parameter is always ignored
    // second parameter is ignored because MockVarExprs are scalars
	EXPECT_DOUBLE_EQ(node.value(0, 0), 7);
	EXPECT_DOUBLE_EQ(node.value(0, 1), 7);
}

TEST_F(binop_fixture, binop_node_size)
{
    addop_node_t node(MockVarExpr(0), MockVarExpr(1));
	EXPECT_EQ(node.size(), 1ul);

    addop_node_t node2(MockVarExpr(3), MockVarExpr(1));
	EXPECT_EQ(node2.size(), 3ul);
}

TEST_F(binop_fixture, binop_node_to_ad)
{
    addop_node_t node(MockVarExpr(2), MockVarExpr(4));
    // all parameters are ignored in this case by MockVarExpr
    auto expr = node.to_ad(0,0,0);
	EXPECT_DOUBLE_EQ(ad::evaluate(expr), 6.0);
}

} // namespace expr
} // namespace ppl
