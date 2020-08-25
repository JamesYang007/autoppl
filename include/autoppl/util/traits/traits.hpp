#pragma once
#include <fastad_bits/util/type_traits.hpp>
#include <autoppl/util/traits/concept.hpp>
#include <autoppl/util/traits/type_traits.hpp>
#include <autoppl/util/traits/shape_traits.hpp>
#include <autoppl/util/traits/var_traits.hpp>
#include <autoppl/util/traits/var_expr_traits.hpp>
#include <autoppl/util/traits/dist_expr_traits.hpp>
#include <autoppl/util/traits/model_expr_traits.hpp>
#include <autoppl/util/traits/program_expr_traits.hpp>

namespace ppl {

// forward declarations
template <class ValueType
        , class ShapeType
        , class ConstraintType>
struct ParamView;

template <class ValueType
        , class ShapeType>
struct TParamView;

template <class ValueType
        , class ShapeType>
struct DataView;

namespace expr {
namespace var {

template <class ValueType
        , class ShapeType>
struct Constant;

} // namespace var

namespace prog {

template <class TupExprType, class>
struct ProgramNode;

} // namespace prog
} // namespace expr

namespace util {
namespace details {

/**
 * Converts T to a valid parameter type for building expressions.
 */
template <class T, class = void>
struct convert_to_param
{};

// Convert from param to param viewer
template <class T>
struct convert_to_param<T,
    std::enable_if_t<
        util::is_param_v<std::decay_t<T>>    
        > >
{
private:
    using raw_t = std::decay_t<T>;
    using value_t = typename 
        util::param_traits<raw_t>::value_t;
    using shape_t = typename 
        util::shape_traits<raw_t>::shape_t;
    using constraint_t = typename
        util::param_traits<raw_t>::constraint_t;
public:
    using type = ppl::ParamView<value_t, shape_t, constraint_t>;
};

// Convert from tparam to tparam viewer
template <class T>
struct convert_to_param<T,
    std::enable_if_t<
        util::is_tparam_v<std::decay_t<T>>    
        > >
{
private:
    using raw_t = std::decay_t<T>;
    using value_t = typename 
        util::tparam_traits<raw_t>::value_t;
    using shape_t = typename 
        util::shape_traits<raw_t>::shape_t;
public:
    using type = ppl::TParamView<value_t, shape_t>;
};

// Convert from data to data viewer
template <class T>
struct convert_to_param<T,
    std::enable_if_t<
        util::is_data_v<std::decay_t<T>>
        > >
{
private:
    using raw_t = std::decay_t<T>;
    using value_t = typename 
        util::data_traits<raw_t>::value_t;
    using shape_t = typename 
        util::shape_traits<raw_t>::shape_t;
public:
    using type = ppl::DataView<value_t, shape_t>;
};

// Convert arithmetic types to Constants
template <class T>
struct convert_to_param<T, 
    std::enable_if_t<std::is_arithmetic_v<std::decay_t<T>> 
        > >
{
    using type = expr::var::Constant<std::decay_t<T>, scl>;
};

// Convert Eigen types to Constants
template <class T>
struct convert_to_param<T, 
    std::enable_if_t<ad::util::is_eigen_v<std::decay_t<T>> 
        > >
{
private:
    using raw_t = std::decay_t<T>;
    using value_t = typename T::Scalar;
    using shape_t = std::conditional_t<
        T::ColsAtCompileTime == 1,
        ppl::vec, ppl::mat>;
public:
    using type = expr::var::Constant<value_t, shape_t>;
};

// Convert variable expressions (not variables) into itself (no change)
template <class T>
struct convert_to_param<T, 
    std::enable_if_t<
        util::is_var_expr_v<std::decay_t<T>> &&
        !util::is_var_v<std::decay_t<T>>
        > >
{
    using type = T;
};

} // namespace details

template <class T>
using convert_to_param_t = 
    typename details::convert_to_param<T>::type;

/**
 * Convert T to valid program expression type.
 */
namespace details {

template <class T, class = void>
struct convert_to_program;

// Convert T to ProgramNode if T is model expression
template <class T>
struct convert_to_program<T,
    std::enable_if_t<
        util::is_model_expr_v<std::decay_t<T>>
        > >
{
    using type = expr::prog::ProgramNode<std::tuple<std::decay_t<T>>, void>;
};

// Convert T to itself if it already is a program expression 
template <class T>
struct convert_to_program<T,
    std::enable_if_t<
        util::is_program_expr_v<std::decay_t<T>>
        > >
{
    using type = T;
};

} // namespace details

template <class T>
using convert_to_program_t = 
    typename details::convert_to_program<T>::type;

/**
 * Checks if valid distribution parameter:
 * - can be arithmetic
 * - if not arithmetic, must be variable or variable expression
 */
template <class T>
inline constexpr bool is_valid_dist_param_v =
    std::is_arithmetic_v<std::decay_t<T>> ||
    ad::util::is_eigen_v<std::decay_t<T>> ||
    util::is_var_v<std::decay_t<T>> ||
    util::is_var_expr_v<std::decay_t<T>>
    ;

/**
 * Checks if the decayed types of T1 and T2 
 * are not both arithmetic types.
 */
template <class T1, class T2>
inline constexpr bool is_not_both_arithmetic_v =
    !(std::is_arithmetic_v<std::decay_t<T1>> &&
      std::is_arithmetic_v<std::decay_t<T2>>)
    ;

template <class T>
inline constexpr bool is_valid_op_param_v =
    std::is_arithmetic_v<std::decay_t<T>> || 
    util::is_var_v<std::decay_t<T>> ||
    util::is_var_expr_v<std::decay_t<T>>
    ;

} // namespace util
} // namespace ppl
