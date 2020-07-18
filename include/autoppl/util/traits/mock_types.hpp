#pragma once
#include <cmath>
#include <fastad>
#include <autoppl/util/traits/traits.hpp>
#include <autoppl/util/functional.hpp>
#include <cassert>

namespace ppl {

/*
 * Mock state class for testing and concepts purposes.
 */
enum class MockState {
    data,
    parameter
};

struct MockParam: 
    util::VarExprBase<MockParam>,
    util::ParamBase<MockParam>
{
    using value_t = double;
    using pointer_t = double*;
    using const_pointer_t = const double*;
    using shape_t = ppl::scl;
    using index_t = uint32_t;
    using id_t = int;
    static constexpr bool has_param = true;
    static constexpr size_t fixed_size = 1;

    template <class PVecType
            , class F = util::identity>
    value_t value(const PVecType&,
                  size_t=0,
                  F f = F()) const 
    { return f(value_); }

    constexpr size_t size() const { return fixed_size; }
    const pointer_t& storage(size_t=0) const { return ptr_; }
    id_t id() const { return id_; }

    template <class VecADVarType>
    auto to_ad(const VecADVarType& vars,
               const VecADVarType&,
               size_t=0) const
    { return vars[0]; }

    index_t set_offset(index_t offset) { 
        offset_ = offset; 
        return offset + this->size();
    }

    index_t set_cache_offset(index_t offset) 
    { return offset; }

    /* Not part of API */
    MockParam(value_t value) : value_{value} {}
    MockParam() =default;

private:
    id_t id_ = 0;
    index_t offset_ = 0;
    value_t value_ = 0.0;
    pointer_t ptr_ = nullptr;
};

struct MockData: 
    util::VarExprBase<MockData>,
    util::DataBase<MockData>
{
    using value_t = double;
    using shape_t = ppl::scl;
    using index_t = uint32_t;
    using id_t = int;
    static constexpr bool has_param = true;
    static constexpr size_t fixed_size = 1;

    template <class PVecType
            , class F = util::identity>
    const value_t& value(const PVecType&,
                         size_t=0,
                         F = F()) const 
    { return value_; }

    constexpr size_t size() const { return 1ul; }
    id_t id() const { return id_; }

    template <class VecADVarType>
    auto to_ad(const VecADVarType&,
               const VecADVarType&,
               size_t=0) const
    { return ad::constant(value_); }

    index_t set_cache_offset(index_t offset) 
    { return offset; }

private:
    id_t id_ = 0;
    value_t value_ = 0.0;
};

/*
 * Mock param class that fits all but the "new" conditions of param.
 */
struct MockNotParam: 
    util::VarExprBase<MockNotParam>
{
    using value_t = double;
    using shape_t = ppl::scl;
    using index_t = uint32_t;
    static constexpr bool has_param = true;
    static constexpr size_t fixed_size = 1;

    template <class PVecType
            , class F = util::identity>
    const value_t& value(const PVecType&,
                         size_t=0,
                         F = F()) const 
    { return value_; }

    constexpr size_t size() const { return 1ul; }

    template <class VecADVarType>
    auto to_ad(const VecADVarType&,
               const VecADVarType&,
               size_t=0) const
    { return ad::constant(value_); }

    index_t set_cache_offset(index_t offset) 
    { return offset; }

private:
    value_t value_ = 0.0;
};

/*
 * Mock data class that fits all but the "new" conditions of data.
 */
struct MockNotData: 
    util::VarExprBase<MockNotData>
{
    using value_t = double;
    using shape_t = ppl::scl;
    using index_t = uint32_t;
    static constexpr bool has_param = true;
    static constexpr size_t fixed_size = 1;

    template <class PVecType
            , class F = util::identity>
    const value_t& value(const PVecType&,
                         size_t=0,
                         F = F()) const 
    { return value_; }

    constexpr size_t size() const { return 1ul; }

    template <class VecADVarType>
    auto to_ad(const VecADVarType&,
               const VecADVarType&,
               size_t=0) const
    { return ad::constant(value_); }

    index_t set_cache_offset(index_t offset) 
    { return offset; }

private:
    value_t value_ = 0.0;
};

/*
 * Mock variable expression class that fits all 
 * conditions of variable expression.
 */
struct MockVarExpr: 
    util::VarExprBase<MockVarExpr>
{
    using value_t = double;
    using shape_t = ppl::scl;
    using index_t = uint32_t;
    static constexpr bool has_param = true;
    static constexpr size_t fixed_size = 1;

    template <class PVecType
            , class F = util::identity>
    const value_t& value(const PVecType&,
                         size_t=0,
                         F = F()) const 
    { return x_; }

    size_t size() const { return x_; }

    template <class VecADVarType>
    auto to_ad(const VecADVarType&, 
               const VecADVarType&, 
               size_t=0) const { 
        return ad::constant(x_); 
    }

    index_t set_cache_offset(index_t offset) 
    { return offset; }

    /* not part of API */
    MockVarExpr(value_t x = 0.)
        : x_{x}
    {}

private:
    value_t x_;
};

/*
 * Mock variable expression class that fits all but the "new"
 * conditions of variable expression.
 */
struct MockNotVarExpr
{
    using shape_t = ppl::scl;
    constexpr size_t size() const { return 1ul; }
};

/*
 * Mock shaped class that fits all conditions of shape.
 */
struct MockScalar
{
    using shape_t = ppl::scl;
    constexpr size_t size() const { return 1ul; }
};

/*
 * Mock distribution expression class that fits all
 * conditions of is_dist_expr_v.
 */
struct MockDistExpr: util::DistExprBase<MockDistExpr>
{
private:
    using base_t = util::DistExprBase<MockDistExpr>;
public:
    using value_t = double;
    using dist_value_t = typename base_t::dist_value_t;
    using index_t = uint32_t;

    template <class PVecType
            , class F = util::identity>
    value_t min(const PVecType&,
                F = F()) const { return 0.; }

    template <class PVecType
            , class F = util::identity>
    value_t max(const PVecType&,
                F = F()) const { return 1.; }

    /* Not part of API */
    MockDistExpr(value_t p=0) : p_{p} {}
    
    template <class VarType
            , class PVecType
            , class F = util::identity>
    value_t pdf(const VarType& x,
                const PVecType& pvalues,
                F f = F()) const 
    { return x.value(pvalues, 0, f) * p_; }

    template <class VarType
            , class PVecType
            , class F = util::identity>
    value_t log_pdf(const VarType& x,
                    const PVecType& pvalues,
                    F f = F()) const 
    { return std::log(this->pdf(x, pvalues, f)); }

    template <class VarType
            , class VecADVarType>
    auto ad_log_pdf(const VarType&,
                    const VecADVarType&,
                    const VecADVarType&) const 
    { return ad::constant(p_); }

    index_t set_cache_offset(index_t offset) 
    { return offset; }

private:
    value_t p_;
};

/*
 * TODO:
 * Mock model expression clases that should meet the 
 * requirements of is_model_expr_v.
 * Additionally, MockEqNode should satisfy is_eq_node_expr_v.
 */

} // namespace ppl
