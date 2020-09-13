#include "gtest/gtest.h"
#include <testutil/base_fixture.hpp>
#include <autoppl/util/traits/var_expr_traits.hpp>
#include <autoppl/expression/variable/param.hpp>
#include <autoppl/expression/variable/data.hpp>
#include <autoppl/expression/variable/constant.hpp>
#include <autoppl/expression/variable/binary.hpp>
#include <autoppl/expression/variable/dot.hpp>
#include <autoppl/expression/distribution/uniform.hpp>
#include <autoppl/expression/model/bar_eq.hpp>
#include <autoppl/expression/model/glue.hpp>
#include <autoppl/expression/op_overloads.hpp>

namespace ppl {

struct expr_builder_fixture: 
    base_fixture<double>,
    ::testing::Test
{
protected:
    scl_pv_t x;
    scl_pv_t y;
    scl_p_t v;
    double d;
    long int i;

    info_pack_t info;

    expr_builder_fixture()
        : x(&info)
        , y(nullptr)
        , v()
    {}
};

TEST_F(expr_builder_fixture, convert_to_param_var)
{
    using namespace details;
    static_assert(std::is_same_v<scl_p_t, std::decay_t<scl_p_t>>);
    static_assert(util::is_var_v<scl_p_t>);
    static_assert(!std::is_same_v<scl_p_t, util::cont_param_t>);
    static_assert(std::is_same_v<
            util::convert_to_param_t<scl_p_t>,
            scl_pv_t
            >);
}

TEST_F(expr_builder_fixture, convert_to_param_raw)
{
    using namespace details;
    using data_t = util::cont_param_t;
    static_assert(std::is_same_v<data_t, std::decay_t<data_t>>);
    static_assert(!util::is_var_v<data_t>);
    static_assert(std::is_same_v<data_t, util::cont_param_t>);
    static_assert(std::is_same_v<
            util::convert_to_param_t<data_t>,
            expr::var::Constant<data_t>
            >);
}

TEST_F(expr_builder_fixture, convert_to_param_var_expr)
{
    using namespace details;
    static_assert(!std::is_same_v<scl_pv_t, util::cont_param_t>);
    static_assert(util::is_var_expr_v<scl_pv_t>);
    static_assert(std::is_same_v<
            util::convert_to_param_t<scl_pv_t&>,
            scl_pv_t
            >);
    static_assert(std::is_same_v<
            util::convert_to_param_t<scl_pv_t&&>,
            scl_pv_t
            >);
}

TEST_F(expr_builder_fixture, op_plus)
{
    // sanity-check: both literals lead to choosing default operator+
    static_assert(std::is_same_v<int, std::decay_t<decltype(3 + 4)> >);

    // scl_pv_t, [scl_pv_t, double, long int, MockVar]
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Add, scl_pv_t, scl_pv_t>,
            std::decay_t<decltype(x + y)> >);
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Add, scl_pv_t, expr::var::Constant<double>>,
            std::decay_t<decltype(x + 3.)> >);
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Add, scl_pv_t, expr::var::Constant<long int>>,
            std::decay_t<decltype(x + 3l)> >);
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Add, scl_pv_t, scl_pv_t>,
            std::decay_t<decltype(x + v)> >);
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Add, scl_pv_t, scl_pv_t>,
            std::decay_t<decltype(scl_pv_t(&info)+ y)> >);
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Add, scl_pv_t, expr::var::Constant<double>>,
            std::decay_t<decltype(scl_pv_t(&info)+ 3.)> >);
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Add, scl_pv_t, expr::var::Constant<long int>>,
            std::decay_t<decltype(scl_pv_t(&info)+ 3l)> >);
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Add, scl_pv_t, scl_pv_t>,
            std::decay_t<decltype(scl_pv_t(&info)+ v)> >);

    // double, [scl_pv_t, double, long int, MockVar]
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Add, expr::var::Constant<double>, scl_pv_t>,
            std::decay_t<decltype(d + y)> >);
    static_assert(std::is_same_v<
            double,
            std::decay_t<decltype(d + 3.)> >);
    static_assert(std::is_same_v<
            double,
            std::decay_t<decltype(d + 3l)> >);
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Add, expr::var::Constant<double>, scl_pv_t>,
            std::decay_t<decltype(d + v)> >);
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Add, expr::var::Constant<double>, scl_pv_t>,
            std::decay_t<decltype(3. + y)> >);
    static_assert(std::is_same_v<
            double,
            std::decay_t<decltype(3. + 3.)> >);
    static_assert(std::is_same_v<
            double,
            std::decay_t<decltype(3. + 3l)> >);
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Add, expr::var::Constant<double>, scl_pv_t>,
            std::decay_t<decltype(3. + v)> >);

    // long int, [scl_pv_t, double, long int, MockVar]
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Add, expr::var::Constant<long>, scl_pv_t>,
            std::decay_t<decltype(i + y)> >);
    static_assert(std::is_same_v<
            double,
            std::decay_t<decltype(i + 3.)> >);
    static_assert(std::is_same_v<
            long,
            std::decay_t<decltype(i + 3l)> >);
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Add, expr::var::Constant<long>, scl_pv_t>,
            std::decay_t<decltype(i + v)> >);
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Add, expr::var::Constant<long>, scl_pv_t>,
            std::decay_t<decltype(3l + y)> >);
    static_assert(std::is_same_v<
            double,
            std::decay_t<decltype(3l + 3.)> >);
    static_assert(std::is_same_v<
            long,
            std::decay_t<decltype(3l + 3l)> >);
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Add, expr::var::Constant<long>, scl_pv_t>,
            std::decay_t<decltype(3l + v)> >);

    // MockVar, [scl_pv_t, double, long int, MockVar]
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Add, scl_pv_t, scl_pv_t>,
            std::decay_t<decltype(v + y)> >);
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Add, scl_pv_t, expr::var::Constant<double>>,
            std::decay_t<decltype(v + 3.)> >);
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Add, scl_pv_t, expr::var::Constant<long>>,
            std::decay_t<decltype(v + 3l)> >);
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Add, scl_pv_t, scl_pv_t>,
            std::decay_t<decltype(v + v)> >);
}

TEST_F(expr_builder_fixture, op_minus)
{
    // scl_pv_t, [scl_pv_t, double, long int, MockVar]
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Sub, scl_pv_t, scl_pv_t>,
            std::decay_t<decltype(x - y)> >);
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Sub, scl_pv_t, expr::var::Constant<double>>,
            std::decay_t<decltype(x - 3.)> >);
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Sub, scl_pv_t, expr::var::Constant<long int>>,
            std::decay_t<decltype(x - 3l)> >);
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Sub, scl_pv_t, scl_pv_t>,
            std::decay_t<decltype(x - v)> >);
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Sub, scl_pv_t, scl_pv_t>,
            std::decay_t<decltype(scl_pv_t(&info)- y)> >);
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Sub, scl_pv_t, expr::var::Constant<double>>,
            std::decay_t<decltype(scl_pv_t(&info)- 3.)> >);
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Sub, scl_pv_t, expr::var::Constant<long int>>,
            std::decay_t<decltype(scl_pv_t(&info)- 3l)> >);
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Sub, scl_pv_t, scl_pv_t>,
            std::decay_t<decltype(scl_pv_t(&info)- v)> >);

    // double, [scl_pv_t, double, long int, MockVar]
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Sub, expr::var::Constant<double>, scl_pv_t>,
            std::decay_t<decltype(d - y)> >);
    static_assert(std::is_same_v<
            double,
            std::decay_t<decltype(d - 3.)> >);
    static_assert(std::is_same_v<
            double,
            std::decay_t<decltype(d - 3l)> >);
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Sub, expr::var::Constant<double>, scl_pv_t>,
            std::decay_t<decltype(d - v)> >);
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Sub, expr::var::Constant<double>, scl_pv_t>,
            std::decay_t<decltype(3. - y)> >);
    static_assert(std::is_same_v<
            double,
            std::decay_t<decltype(3. - 3.)> >);
    static_assert(std::is_same_v<
            double,
            std::decay_t<decltype(3. - 3l)> >);
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Sub, expr::var::Constant<double>, scl_pv_t>,
            std::decay_t<decltype(3. - v)> >);

    // long int, [scl_pv_t, double, long int, MockVar]
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Sub, expr::var::Constant<long>, scl_pv_t>,
            std::decay_t<decltype(i - y)> >);
    static_assert(std::is_same_v<
            double,
            std::decay_t<decltype(i - 3.)> >);
    static_assert(std::is_same_v<
            long,
            std::decay_t<decltype(i - 3l)> >);
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Sub, expr::var::Constant<long>, scl_pv_t>,
            std::decay_t<decltype(i - v)> >);
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Sub, expr::var::Constant<long>, scl_pv_t>,
            std::decay_t<decltype(3l - y)> >);
    static_assert(std::is_same_v<
            double,
            std::decay_t<decltype(3l - 3.)> >);
    static_assert(std::is_same_v<
            long,
            std::decay_t<decltype(3l - 3l)> >);
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Sub, expr::var::Constant<long>, scl_pv_t>,
            std::decay_t<decltype(3l - v)> >);

    // MockVar, [scl_pv_t, double, long int, MockVar]
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Sub, scl_pv_t, scl_pv_t>,
            std::decay_t<decltype(v - y)> >);
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Sub, scl_pv_t, expr::var::Constant<double>>,
            std::decay_t<decltype(v - 3.)> >);
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Sub, scl_pv_t, expr::var::Constant<long>>,
            std::decay_t<decltype(v - 3l)> >);
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Sub, scl_pv_t, scl_pv_t>,
            std::decay_t<decltype(v - v)> >);
}

TEST_F(expr_builder_fixture, op_times)
{
    // scl_pv_t, [scl_pv_t, double, long int, MockVar]
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Mul, scl_pv_t, scl_pv_t>,
            std::decay_t<decltype(x * y)> >);
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Mul, scl_pv_t, expr::var::Constant<double>>,
            std::decay_t<decltype(x * 3.)> >);
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Mul, scl_pv_t, expr::var::Constant<long int>>,
            std::decay_t<decltype(x * 3l)> >);
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Mul, scl_pv_t, scl_pv_t>,
            std::decay_t<decltype(x * v)> >);
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Mul, scl_pv_t, scl_pv_t>,
            std::decay_t<decltype(scl_pv_t(&info)* y)> >);
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Mul, scl_pv_t, expr::var::Constant<double>>,
            std::decay_t<decltype(scl_pv_t(&info)* 3.)> >);
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Mul, scl_pv_t, expr::var::Constant<long int>>,
            std::decay_t<decltype(scl_pv_t(&info)* 3l)> >);
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Mul, scl_pv_t, scl_pv_t>,
            std::decay_t<decltype(scl_pv_t(&info)* v)> >);

    // double, [scl_pv_t, double, long int, MockVar]
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Mul, expr::var::Constant<double>, scl_pv_t>,
            std::decay_t<decltype(d * y)> >);
    static_assert(std::is_same_v<
            double,
            std::decay_t<decltype(d * 3.)> >);
    static_assert(std::is_same_v<
            double,
            std::decay_t<decltype(d * 3l)> >);
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Mul, expr::var::Constant<double>, scl_pv_t>,
            std::decay_t<decltype(d * v)> >);
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Mul, expr::var::Constant<double>, scl_pv_t>,
            std::decay_t<decltype(3. * y)> >);
    static_assert(std::is_same_v<
            double,
            std::decay_t<decltype(3. * 3.)> >);
    static_assert(std::is_same_v<
            double,
            std::decay_t<decltype(3. * 3l)> >);
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Mul, expr::var::Constant<double>, scl_pv_t>,
            std::decay_t<decltype(3. * v)> >);

    // long int, [scl_pv_t, double, long int, MockVar]
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Mul, expr::var::Constant<long>, scl_pv_t>,
            std::decay_t<decltype(i * y)> >);
    static_assert(std::is_same_v<
            double,
            std::decay_t<decltype(i * 3.)> >);
    static_assert(std::is_same_v<
            long,
            std::decay_t<decltype(i * 3l)> >);
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Mul, expr::var::Constant<long>, scl_pv_t>,
            std::decay_t<decltype(i * v)> >);
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Mul, expr::var::Constant<long>, scl_pv_t>,
            std::decay_t<decltype(3l * y)> >);
    static_assert(std::is_same_v<
            double,
            std::decay_t<decltype(3l * 3.)> >);
    static_assert(std::is_same_v<
            long,
            std::decay_t<decltype(3l * 3l)> >);
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Mul, expr::var::Constant<long>, scl_pv_t>,
            std::decay_t<decltype(3l * v)> >);

    // MockVar, [scl_pv_t, double, long int, MockVar]
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Mul, scl_pv_t, scl_pv_t>,
            std::decay_t<decltype(v * y)> >);
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Mul, scl_pv_t, expr::var::Constant<double>>,
            std::decay_t<decltype(v * 3.)> >);
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Mul, scl_pv_t, expr::var::Constant<long>>,
            std::decay_t<decltype(v * 3l)> >);
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Mul, scl_pv_t, scl_pv_t>,
            std::decay_t<decltype(v * v)> >);
}

TEST_F(expr_builder_fixture, op_div)
{
    // scl_pv_t, [scl_pv_t, double, long int, MockVar]
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Div, scl_pv_t, scl_pv_t>,
            std::decay_t<decltype(x / y)> >);
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Div, scl_pv_t, expr::var::Constant<double>>,
            std::decay_t<decltype(x / 3.)> >);
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Div, scl_pv_t, expr::var::Constant<long int>>,
            std::decay_t<decltype(x / 3l)> >);
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Div, scl_pv_t, scl_pv_t>,
            std::decay_t<decltype(x / v)> >);
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Div, scl_pv_t, scl_pv_t>,
            std::decay_t<decltype(scl_pv_t(&info)/ y)> >);
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Div, scl_pv_t, expr::var::Constant<double>>,
            std::decay_t<decltype(scl_pv_t(&info)/ 3.)> >);
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Div, scl_pv_t, expr::var::Constant<long int>>,
            std::decay_t<decltype(scl_pv_t(&info)/ 3l)> >);
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Div, scl_pv_t, scl_pv_t>,
            std::decay_t<decltype(scl_pv_t(&info)/ v)> >);

    // double, [scl_pv_t, double, long int, MockVar]
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Div, expr::var::Constant<double>, scl_pv_t>,
            std::decay_t<decltype(d / y)> >);
    static_assert(std::is_same_v<
            double,
            std::decay_t<decltype(d / 3.)> >);
    static_assert(std::is_same_v<
            double,
            std::decay_t<decltype(d / 3l)> >);
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Div, expr::var::Constant<double>, scl_pv_t>,
            std::decay_t<decltype(d / v)> >);
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Div, expr::var::Constant<double>, scl_pv_t>,
            std::decay_t<decltype(3. / y)> >);
    static_assert(std::is_same_v<
            double,
            std::decay_t<decltype(3. / 3.)> >);
    static_assert(std::is_same_v<
            double,
            std::decay_t<decltype(3. / 3l)> >);
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Div, expr::var::Constant<double>, scl_pv_t>,
            std::decay_t<decltype(3. / v)> >);

    // long int, [scl_pv_t, double, long int, MockVar]
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Div, expr::var::Constant<long>, scl_pv_t>,
            std::decay_t<decltype(i / y)> >);
    static_assert(std::is_same_v<
            double,
            std::decay_t<decltype(i / 3.)> >);
    static_assert(std::is_same_v<
            long,
            std::decay_t<decltype(i / 3l)> >);
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Div, expr::var::Constant<long>, scl_pv_t>,
            std::decay_t<decltype(i / v)> >);
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Div, expr::var::Constant<long>, scl_pv_t>,
            std::decay_t<decltype(3l / y)> >);
    static_assert(std::is_same_v<
            double,
            std::decay_t<decltype(3l / 3.)> >);
    static_assert(std::is_same_v<
            long,
            std::decay_t<decltype(3l / 3l)> >);
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Div, expr::var::Constant<long>, scl_pv_t>,
            std::decay_t<decltype(3l / v)> >);

    // MockVar, [scl_pv_t, double, long int, MockVar]
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Div, scl_pv_t, scl_pv_t>,
            std::decay_t<decltype(v / y)> >);
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Div, scl_pv_t, expr::var::Constant<double>>,
            std::decay_t<decltype(v / 3.)> >);
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Div, scl_pv_t, expr::var::Constant<long>>,
            std::decay_t<decltype(v / 3l)> >);
    static_assert(std::is_same_v<
            expr::var::BinaryNode<ad::core::Div, scl_pv_t, scl_pv_t>,
            std::decay_t<decltype(v / v)> >);
}
} // namespace ppl
