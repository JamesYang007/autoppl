#include "gtest/gtest.h"
#include <autoppl/expr_builder.hpp>
#include <testutil/mock_types.hpp>

namespace ppl {

struct expr_builder_fixture : ::testing::Test
{
protected:
    MockVarExpr x;
    MockVarExpr y;
    MockParam v;
    double d;
    long int i;
};

TEST_F(expr_builder_fixture, convert_to_param_var)
{
    using namespace details;
    static_assert(std::is_same_v<MockParam, std::decay_t<MockParam>>);
#if __cplusplus <= 201703L
    static_assert(util::is_var_v<MockParam>);
#else
    static_assert(util::var<MockParam>);
#endif
    static_assert(!std::is_same_v<MockParam, util::cont_param_t>);
#if __cplusplus <= 201703L
    static_assert(!util::is_var_expr_v<MockParam>);
#else
    static_assert(!util::var_expr<MockParam>);
#endif
    static_assert(std::is_same_v<
            convert_to_param_t<MockParam>,
            expr::VariableViewer<MockParam>
            >);
}

TEST_F(expr_builder_fixture, convert_to_param_raw)
{
    using namespace details;
    using data_t = util::cont_param_t;
    static_assert(std::is_same_v<data_t, std::decay_t<data_t>>);
#if __cplusplus <= 201703L
    static_assert(!util::is_var_v<data_t>);
#else
    static_assert(!util::var<data_t>);
#endif
    static_assert(std::is_same_v<data_t, util::cont_param_t>);
#if __cplusplus <= 201703L
    static_assert(!util::is_var_expr_v<data_t>);
#else
    static_assert(!util::var_expr<data_t>);
#endif
    static_assert(std::is_same_v<
            convert_to_param_t<data_t>,
            expr::Constant<data_t>
            >);
}

TEST_F(expr_builder_fixture, convert_to_param_var_expr)
{
    using namespace details;
#if __cplusplus <= 201703L
    static_assert(!util::is_var_v<MockVarExpr>);
#else
    static_assert(!util::var<MockVarExpr>);
#endif
    static_assert(!std::is_same_v<MockVarExpr, util::cont_param_t>);
#if __cplusplus <= 201703L
    static_assert(util::is_var_expr_v<MockVarExpr>);
#else
    static_assert(util::var_expr<MockVarExpr>);
#endif
    static_assert(std::is_same_v<
            convert_to_param_t<MockVarExpr&>,
            MockVarExpr&
            >);
    static_assert(std::is_same_v<
            convert_to_param_t<MockVarExpr&&>,
            MockVarExpr&&
            >);
}

TEST_F(expr_builder_fixture, op_plus)
{
    // sanity-check: both literals lead to choosing default operator+
    static_assert(std::is_same_v<int, std::decay_t<decltype(3 + 4)> >);

    // MockVarExpr, [MockVarExpr, double, long int, MockVar]
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::AddOp, MockVarExpr, MockVarExpr>,
            std::decay_t<decltype(x + y)> >);
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::AddOp, MockVarExpr, expr::Constant<double>>,
            std::decay_t<decltype(x + 3.)> >);
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::AddOp, MockVarExpr, expr::Constant<long int>>,
            std::decay_t<decltype(x + 3l)> >);
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::AddOp, MockVarExpr, expr::VariableViewer<MockParam>>,
            std::decay_t<decltype(x + v)> >);
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::AddOp, MockVarExpr, MockVarExpr>,
            std::decay_t<decltype(MockVarExpr()+ y)> >);
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::AddOp, MockVarExpr, expr::Constant<double>>,
            std::decay_t<decltype(MockVarExpr()+ 3.)> >);
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::AddOp, MockVarExpr, expr::Constant<long int>>,
            std::decay_t<decltype(MockVarExpr()+ 3l)> >);
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::AddOp, MockVarExpr, expr::VariableViewer<MockParam>>,
            std::decay_t<decltype(MockVarExpr()+ v)> >);

    // double, [MockVarExpr, double, long int, MockVar]
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::AddOp, expr::Constant<double>, MockVarExpr>,
            std::decay_t<decltype(d + y)> >);
    static_assert(std::is_same_v<
            double,
            std::decay_t<decltype(d + 3.)> >);
    static_assert(std::is_same_v<
            double,
            std::decay_t<decltype(d + 3l)> >);
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::AddOp, expr::Constant<double>, expr::VariableViewer<MockParam>>,
            std::decay_t<decltype(d + v)> >);
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::AddOp, expr::Constant<double>, MockVarExpr>,
            std::decay_t<decltype(3. + y)> >);
    static_assert(std::is_same_v<
            double,
            std::decay_t<decltype(3. + 3.)> >);
    static_assert(std::is_same_v<
            double,
            std::decay_t<decltype(3. + 3l)> >);
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::AddOp, expr::Constant<double>, expr::VariableViewer<MockParam>>,
            std::decay_t<decltype(3. + v)> >);

    // long int, [MockVarExpr, double, long int, MockVar]
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::AddOp, expr::Constant<long>, MockVarExpr>,
            std::decay_t<decltype(i + y)> >);
    static_assert(std::is_same_v<
            double,
            std::decay_t<decltype(i + 3.)> >);
    static_assert(std::is_same_v<
            long,
            std::decay_t<decltype(i + 3l)> >);
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::AddOp, expr::Constant<long>, expr::VariableViewer<MockParam>>,
            std::decay_t<decltype(i + v)> >);
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::AddOp, expr::Constant<long>, MockVarExpr>,
            std::decay_t<decltype(3l + y)> >);
    static_assert(std::is_same_v<
            double,
            std::decay_t<decltype(3l + 3.)> >);
    static_assert(std::is_same_v<
            long,
            std::decay_t<decltype(3l + 3l)> >);
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::AddOp, expr::Constant<long>, expr::VariableViewer<MockParam>>,
            std::decay_t<decltype(3l + v)> >);

    // MockVar, [MockVarExpr, double, long int, MockVar]
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::AddOp, expr::VariableViewer<MockParam>, MockVarExpr>,
            std::decay_t<decltype(v + y)> >);
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::AddOp, expr::VariableViewer<MockParam>, expr::Constant<double>>,
            std::decay_t<decltype(v + 3.)> >);
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::AddOp, expr::VariableViewer<MockParam>, expr::Constant<long>>,
            std::decay_t<decltype(v + 3l)> >);
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::AddOp, expr::VariableViewer<MockParam>, expr::VariableViewer<MockParam>>,
            std::decay_t<decltype(v + v)> >);
}

TEST_F(expr_builder_fixture, op_minus)
{
    // MockVarExpr, [MockVarExpr, double, long int, MockVar]
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::SubOp, MockVarExpr, MockVarExpr>,
            std::decay_t<decltype(x - y)> >);
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::SubOp, MockVarExpr, expr::Constant<double>>,
            std::decay_t<decltype(x - 3.)> >);
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::SubOp, MockVarExpr, expr::Constant<long int>>,
            std::decay_t<decltype(x - 3l)> >);
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::SubOp, MockVarExpr, expr::VariableViewer<MockParam>>,
            std::decay_t<decltype(x - v)> >);
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::SubOp, MockVarExpr, MockVarExpr>,
            std::decay_t<decltype(MockVarExpr()- y)> >);
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::SubOp, MockVarExpr, expr::Constant<double>>,
            std::decay_t<decltype(MockVarExpr()- 3.)> >);
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::SubOp, MockVarExpr, expr::Constant<long int>>,
            std::decay_t<decltype(MockVarExpr()- 3l)> >);
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::SubOp, MockVarExpr, expr::VariableViewer<MockParam>>,
            std::decay_t<decltype(MockVarExpr()- v)> >);

    // double, [MockVarExpr, double, long int, MockVar]
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::SubOp, expr::Constant<double>, MockVarExpr>,
            std::decay_t<decltype(d - y)> >);
    static_assert(std::is_same_v<
            double,
            std::decay_t<decltype(d - 3.)> >);
    static_assert(std::is_same_v<
            double,
            std::decay_t<decltype(d - 3l)> >);
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::SubOp, expr::Constant<double>, expr::VariableViewer<MockParam>>,
            std::decay_t<decltype(d - v)> >);
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::SubOp, expr::Constant<double>, MockVarExpr>,
            std::decay_t<decltype(3. - y)> >);
    static_assert(std::is_same_v<
            double,
            std::decay_t<decltype(3. - 3.)> >);
    static_assert(std::is_same_v<
            double,
            std::decay_t<decltype(3. - 3l)> >);
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::SubOp, expr::Constant<double>, expr::VariableViewer<MockParam>>,
            std::decay_t<decltype(3. - v)> >);

    // long int, [MockVarExpr, double, long int, MockVar]
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::SubOp, expr::Constant<long>, MockVarExpr>,
            std::decay_t<decltype(i - y)> >);
    static_assert(std::is_same_v<
            double,
            std::decay_t<decltype(i - 3.)> >);
    static_assert(std::is_same_v<
            long,
            std::decay_t<decltype(i - 3l)> >);
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::SubOp, expr::Constant<long>, expr::VariableViewer<MockParam>>,
            std::decay_t<decltype(i - v)> >);
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::SubOp, expr::Constant<long>, MockVarExpr>,
            std::decay_t<decltype(3l - y)> >);
    static_assert(std::is_same_v<
            double,
            std::decay_t<decltype(3l - 3.)> >);
    static_assert(std::is_same_v<
            long,
            std::decay_t<decltype(3l - 3l)> >);
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::SubOp, expr::Constant<long>, expr::VariableViewer<MockParam>>,
            std::decay_t<decltype(3l - v)> >);

    // MockVar, [MockVarExpr, double, long int, MockVar]
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::SubOp, expr::VariableViewer<MockParam>, MockVarExpr>,
            std::decay_t<decltype(v - y)> >);
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::SubOp, expr::VariableViewer<MockParam>, expr::Constant<double>>,
            std::decay_t<decltype(v - 3.)> >);
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::SubOp, expr::VariableViewer<MockParam>, expr::Constant<long>>,
            std::decay_t<decltype(v - 3l)> >);
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::SubOp, expr::VariableViewer<MockParam>, expr::VariableViewer<MockParam>>,
            std::decay_t<decltype(v - v)> >);
}

TEST_F(expr_builder_fixture, op_times)
{
    // MockVarExpr, [MockVarExpr, double, long int, MockVar]
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::MultOp, MockVarExpr, MockVarExpr>,
            std::decay_t<decltype(x * y)> >);
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::MultOp, MockVarExpr, expr::Constant<double>>,
            std::decay_t<decltype(x * 3.)> >);
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::MultOp, MockVarExpr, expr::Constant<long int>>,
            std::decay_t<decltype(x * 3l)> >);
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::MultOp, MockVarExpr, expr::VariableViewer<MockParam>>,
            std::decay_t<decltype(x * v)> >);
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::MultOp, MockVarExpr, MockVarExpr>,
            std::decay_t<decltype(MockVarExpr()* y)> >);
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::MultOp, MockVarExpr, expr::Constant<double>>,
            std::decay_t<decltype(MockVarExpr()* 3.)> >);
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::MultOp, MockVarExpr, expr::Constant<long int>>,
            std::decay_t<decltype(MockVarExpr()* 3l)> >);
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::MultOp, MockVarExpr, expr::VariableViewer<MockParam>>,
            std::decay_t<decltype(MockVarExpr()* v)> >);

    // double, [MockVarExpr, double, long int, MockVar]
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::MultOp, expr::Constant<double>, MockVarExpr>,
            std::decay_t<decltype(d * y)> >);
    static_assert(std::is_same_v<
            double,
            std::decay_t<decltype(d * 3.)> >);
    static_assert(std::is_same_v<
            double,
            std::decay_t<decltype(d * 3l)> >);
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::MultOp, expr::Constant<double>, expr::VariableViewer<MockParam>>,
            std::decay_t<decltype(d * v)> >);
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::MultOp, expr::Constant<double>, MockVarExpr>,
            std::decay_t<decltype(3. * y)> >);
    static_assert(std::is_same_v<
            double,
            std::decay_t<decltype(3. * 3.)> >);
    static_assert(std::is_same_v<
            double,
            std::decay_t<decltype(3. * 3l)> >);
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::MultOp, expr::Constant<double>, expr::VariableViewer<MockParam>>,
            std::decay_t<decltype(3. * v)> >);

    // long int, [MockVarExpr, double, long int, MockVar]
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::MultOp, expr::Constant<long>, MockVarExpr>,
            std::decay_t<decltype(i * y)> >);
    static_assert(std::is_same_v<
            double,
            std::decay_t<decltype(i * 3.)> >);
    static_assert(std::is_same_v<
            long,
            std::decay_t<decltype(i * 3l)> >);
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::MultOp, expr::Constant<long>, expr::VariableViewer<MockParam>>,
            std::decay_t<decltype(i * v)> >);
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::MultOp, expr::Constant<long>, MockVarExpr>,
            std::decay_t<decltype(3l * y)> >);
    static_assert(std::is_same_v<
            double,
            std::decay_t<decltype(3l * 3.)> >);
    static_assert(std::is_same_v<
            long,
            std::decay_t<decltype(3l * 3l)> >);
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::MultOp, expr::Constant<long>, expr::VariableViewer<MockParam>>,
            std::decay_t<decltype(3l * v)> >);

    // MockVar, [MockVarExpr, double, long int, MockVar]
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::MultOp, expr::VariableViewer<MockParam>, MockVarExpr>,
            std::decay_t<decltype(v * y)> >);
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::MultOp, expr::VariableViewer<MockParam>, expr::Constant<double>>,
            std::decay_t<decltype(v * 3.)> >);
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::MultOp, expr::VariableViewer<MockParam>, expr::Constant<long>>,
            std::decay_t<decltype(v * 3l)> >);
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::MultOp, expr::VariableViewer<MockParam>, expr::VariableViewer<MockParam>>,
            std::decay_t<decltype(v * v)> >);
}

TEST_F(expr_builder_fixture, op_div)
{
    // MockVarExpr, [MockVarExpr, double, long int, MockVar]
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::DivOp, MockVarExpr, MockVarExpr>,
            std::decay_t<decltype(x / y)> >);
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::DivOp, MockVarExpr, expr::Constant<double>>,
            std::decay_t<decltype(x / 3.)> >);
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::DivOp, MockVarExpr, expr::Constant<long int>>,
            std::decay_t<decltype(x / 3l)> >);
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::DivOp, MockVarExpr, expr::VariableViewer<MockParam>>,
            std::decay_t<decltype(x / v)> >);
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::DivOp, MockVarExpr, MockVarExpr>,
            std::decay_t<decltype(MockVarExpr()/ y)> >);
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::DivOp, MockVarExpr, expr::Constant<double>>,
            std::decay_t<decltype(MockVarExpr()/ 3.)> >);
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::DivOp, MockVarExpr, expr::Constant<long int>>,
            std::decay_t<decltype(MockVarExpr()/ 3l)> >);
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::DivOp, MockVarExpr, expr::VariableViewer<MockParam>>,
            std::decay_t<decltype(MockVarExpr()/ v)> >);

    // double, [MockVarExpr, double, long int, MockVar]
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::DivOp, expr::Constant<double>, MockVarExpr>,
            std::decay_t<decltype(d / y)> >);
    static_assert(std::is_same_v<
            double,
            std::decay_t<decltype(d / 3.)> >);
    static_assert(std::is_same_v<
            double,
            std::decay_t<decltype(d / 3l)> >);
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::DivOp, expr::Constant<double>, expr::VariableViewer<MockParam>>,
            std::decay_t<decltype(d / v)> >);
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::DivOp, expr::Constant<double>, MockVarExpr>,
            std::decay_t<decltype(3. / y)> >);
    static_assert(std::is_same_v<
            double,
            std::decay_t<decltype(3. / 3.)> >);
    static_assert(std::is_same_v<
            double,
            std::decay_t<decltype(3. / 3l)> >);
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::DivOp, expr::Constant<double>, expr::VariableViewer<MockParam>>,
            std::decay_t<decltype(3. / v)> >);

    // long int, [MockVarExpr, double, long int, MockVar]
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::DivOp, expr::Constant<long>, MockVarExpr>,
            std::decay_t<decltype(i / y)> >);
    static_assert(std::is_same_v<
            double,
            std::decay_t<decltype(i / 3.)> >);
    static_assert(std::is_same_v<
            long,
            std::decay_t<decltype(i / 3l)> >);
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::DivOp, expr::Constant<long>, expr::VariableViewer<MockParam>>,
            std::decay_t<decltype(i / v)> >);
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::DivOp, expr::Constant<long>, MockVarExpr>,
            std::decay_t<decltype(3l / y)> >);
    static_assert(std::is_same_v<
            double,
            std::decay_t<decltype(3l / 3.)> >);
    static_assert(std::is_same_v<
            long,
            std::decay_t<decltype(3l / 3l)> >);
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::DivOp, expr::Constant<long>, expr::VariableViewer<MockParam>>,
            std::decay_t<decltype(3l / v)> >);

    // MockVar, [MockVarExpr, double, long int, MockVar]
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::DivOp, expr::VariableViewer<MockParam>, MockVarExpr>,
            std::decay_t<decltype(v / y)> >);
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::DivOp, expr::VariableViewer<MockParam>, expr::Constant<double>>,
            std::decay_t<decltype(v / 3.)> >);
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::DivOp, expr::VariableViewer<MockParam>, expr::Constant<long>>,
            std::decay_t<decltype(v / 3l)> >);
    static_assert(std::is_same_v<
            expr::BinaryOpNode<expr::DivOp, expr::VariableViewer<MockParam>, expr::VariableViewer<MockParam>>,
            std::decay_t<decltype(v / v)> >);
}
} // namespace ppl
