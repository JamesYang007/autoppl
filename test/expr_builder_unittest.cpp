#include "gtest/gtest.h"
#include <autoppl/expr_builder.hpp>
#include <testutil/mock_types.hpp>

namespace ppl {

struct expr_builder_fixture : ::testing::Test
{
protected:
};

TEST_F(expr_builder_fixture, convert_to_cont_param_var)
{
    using namespace details;
    static_assert(std::is_same_v<MockVar, std::decay_t<MockVar>>);
    static_assert(util::is_var_v<MockVar>);
    static_assert(!std::is_same_v<MockVar, util::cont_param_t>);
    static_assert(!util::is_var_expr_v<MockVar>);
    static_assert(std::is_same_v<
            convert_to_cont_param_t<MockVar>,
            expr::VariableViewer<MockVar>
            >);
}

TEST_F(expr_builder_fixture, convert_to_cont_param_raw)
{
    using namespace details;
    using data_t = util::cont_param_t;
    static_assert(std::is_same_v<data_t, std::decay_t<data_t>>);
    static_assert(!util::is_var_v<data_t>);
    static_assert(std::is_same_v<data_t, util::cont_param_t>);
    static_assert(!util::is_var_expr_v<data_t>);
    static_assert(std::is_same_v<
            convert_to_cont_param_t<data_t>,
            expr::Constant<data_t>
            >);
}

TEST_F(expr_builder_fixture, convert_to_cont_param_var_expr)
{
    using namespace details;
    static_assert(!util::is_var_v<MockVarExpr>);
    static_assert(!std::is_same_v<MockVarExpr, util::cont_param_t>);
    static_assert(util::is_var_expr_v<MockVarExpr>);
    static_assert(std::is_same_v<
            convert_to_cont_param_t<MockVarExpr&>,
            MockVarExpr&
            >);
    static_assert(std::is_same_v<
            convert_to_cont_param_t<MockVarExpr&&>,
            MockVarExpr&&
            >);
}

struct binop_overload_fixture : ::testing::Test {

	MockVarExpr x = 0;
	MockVarExpr y = 0;

};

TEST_F(binop_overload_fixture, op_plus)
{
	static_assert(
			std::is_same_v<expr::BinaryOpNode<expr::AddOp, MockVarExpr, MockVarExpr>,
			std::decay_t<decltype(x + y)> >);
}

TEST_F(binop_overload_fixture, op_minus)
{
	static_assert(
			std::is_same_v<expr::BinaryOpNode<expr::SubOp, MockVarExpr, MockVarExpr>,
			std::decay_t<decltype(x - y)> >);
}

TEST_F(binop_overload_fixture, op_times)
{
	static_assert(
			std::is_same_v<expr::BinaryOpNode<expr::MultOp, MockVarExpr, MockVarExpr>,
			std::decay_t<decltype(x * y)> >);
}
TEST_F(binop_overload_fixture, op_div)
{
	static_assert(
			std::is_same_v<expr::BinaryOpNode<expr::DivOp, MockVarExpr, MockVarExpr>,
			std::decay_t<decltype(x / y)> >);
}
} // namespace ppl
