#pragma once
#include <cmath>

namespace ppl {

/*
 * Mock state class for testing purposes.
 */
enum class MockState {
    data,
    parameter
};

/*
 * Mock Variable class that should meet the requirements
 * of is_var_v.
 */
struct MockVar
{
    using value_t = double;
    using pointer_t = double*;
    using const_pointer_t = const double*;
    using state_t = MockState;

    value_t get_value() const {return x_;}
    explicit operator value_t() const { return get_value(); }

    void set_value(value_t x) {x_ = x;}

    void set_storage(pointer_t ptr) {ptr_ = ptr;}

    void set_state(state_t state) {state_ = state;}
    state_t get_state() const {return state_;}

private:
    value_t x_ = 0.;
    pointer_t ptr_ = nullptr;
    state_t state_ = state_t::parameter;
};

/*
 * Mock variable classes that fulfill 
 * var_traits requirements, but do not fit the rest.
 */
struct MockVar_no_convertible
{
    using value_t = double;
    using pointer_t = double*;
    using const_pointer_t = const double*;
    using state_t = void;
};

/*
 * Mock Variable Expression class that should meet the requirements
 * of is_var_expr_v.
 */
struct MockVarExpr
{
    using value_t = double;
    value_t get_value() const { return x_; }
    explicit operator value_t() const { return get_value(); }

    /* not part of API */
    MockVarExpr(value_t x)
        : x_{x}
    {}

    void set_value(value_t x) {x_ = x;}
private:
    value_t x_ = 0.;
};

/*
 * Mock variable expression classes that fulfill 
 * var_expr_traits requirements, but do not fit the rest.
 */
struct MockVarExpr_no_convertible
{
    using value_t = double;
};

/*
 * Mock distribution expression class that should meet the requirements
 * of is_dist_expr_v.
 */
struct MockDistExpr
{
    using value_t = double;
    using dist_value_t = double;

    dist_value_t pdf(value_t x) const
    { return x; }

    dist_value_t log_pdf(value_t x) const
    { return std::log(x); }
};

/*
 * Mock distribution expression classes that fulfill 
 * dist_expr_traits requirements, but do not fit the rest.
 */
struct MockDistExpr_no_pdf : public MockDistExpr
{
private:
    using MockDistExpr::pdf;
};

struct MockDistExpr_no_log_pdf : public MockDistExpr
{
private:
    using MockDistExpr::log_pdf;
};

/*
 * TODO:
 * Mock model expression clases that should meet the 
 * requirements of is_model_expr_v.
 * Additionally, MockEqNode should satisfy is_eq_node_expr_v.
 */

} // namespace ppl
