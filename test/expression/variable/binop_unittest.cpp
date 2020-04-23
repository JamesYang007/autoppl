#include <cmath>
#include <array>
#include "gtest/gtest.h"
#include <autoppl/expression/variable/binop.hpp>

namespace ppl {
namespace expr {

//////////////////////////////////////////////////////
// Model with one RV TESTS
//////////////////////////////////////////////////////

/*
 * Mock var object for testing purposes.
 * Must meet some of the requirements of actual var types.
 */
struct MockVar 
{
    using value_t = double;
    using pointer_t = double*;
    using state_t = void;
	using binop_result_t = double;

    void set_value(double val) { value_ = val; }  
    double get_value() const { return value_; } 

	/* BinOpNode<MockVar, MockVar, MockBinaryOp> operator+(const MockVar& b) const
	{
		return new BinOpNode<MockVar, MockVar, MockBinaryOp>(this, b);
	} */

private:
    double value_;
};

/*
 * Mock binary operation node for testing purposes.
 */
struct MockBinaryOp
{
	using binop_result_t = double;

	// mock operation -- returns 1
	double evaluate(MockVar x, MockVar y) {
		double xv = x.get_value();
		double yv = y.get_value();
		return xv + yv;
	}

};

struct binop_fixture : ::testing::Test
{
protected:
	MockVar x;
	MockVar y;

	using binop_result_t = double;
	/*AddOp<double, double> addNode();
	MultOp<double, double> multNode();
	*/
	/* BinaryOpNode<MockVar, MockVar, AddOp<MockVar, MockVar> > addNode = x + y;
	BinaryOpNode<MockVar, MockVar, MultOp<MockVar, MockVar> > multNode = x * y;
	*/
	
	void reconfigureX(double val)
	{ x.set_value(val); }

	void reconfigureY(double val)
	{ y.set_value(val); }

};

TEST_F(binop_fixture, add)
{
	double val1 = 3;
	double val2 = 4;
	
	double addResult = AddOp::evaluate(val1, val2);
	double multResult = MultOp::evaluate(val1, val2);
	
	EXPECT_EQ(addResult, 7);
	EXPECT_EQ(multResult, 12);
}

} // namespace expr
} // namespace ppl
