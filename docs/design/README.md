# Design Overview

## Expression

The bulk of the work is building a systematic way of creating expressions.
We define three big concepts of expressions that will be powerful enough
to construct many examples such as linear regression and Bayesian network.

### Shape Traits

The recent version of AutoPPL incorporates shape information as part of the type.
This brings significant boost in performance since computation graphs can be
further optimized at compile-time.
Note that only the general shape must be known, __not__ the actual dimensions,
which are usually known at run-time.

We currently support only scalar, vector, and matrix shapes.
They have corresponding tags defined as:
```cpp
ppl::scl // scalar
ppl::vec // vector
ppl::mat // matrix
```

If any objects are "tagged" with one (and only one!) of these tags,
they are to satisfy the `shape` concept.
In more detail, the concepts are defined as the following:

```cpp
template <class T>
concept scl_c = 
    requires(const T cx) {
        typename T::shape_t;
        { cx.size() } -> std::same_as<size_t>;
    } &&
    std::same_as<typename T::shape_t, ppl::scl>
    ;

template <class T>
concept vec_c = 
    requires(const T cx) {
        typename T::shape_t;
        { cx.size() } -> std::same_as<size_t>;
    } &&
    std::same_as<typename T::shape_t, ppl::vec>
    ;

template <class T>
concept mat_c = 
    requires(const T cx) {
        typename T::shape_t;
        { cx.size() } -> std::same_as<size_t>;
    } &&
    std::same_as<typename T::shape_t, ppl::mat>
    ;

template <class T>
concept shape_c =
    scl_c<T> || 
    vec_c<T> ||
    mat_c<T>
    ;
```

- the user must define a member alias `shape_t` that refers to one of the three tags.
- const member function `size` must return the number of elements it represents.

### Variable Expression

A `variable expression` is heuristically one that 
consists of mathematical operations on variable names.
This definition is motivated by looking at examples such as:
```cpp
x + y * z - w / 2.
```

While we only support up to the four binary operations 
and matrix dot-product with a vector, a `variable expression` is
general enough to extend to other cases such as unary operations like:

```cpp
-x
sin(x)
sigmoid(x)
```

The concept is defined as the following:

```cpp
template <class T>
concept var_expr_c = 
    shape_c<T> &&
    var_expr_is_base_of_v<T> &&
    requires () {
        var_expr_traits<T>::has_param;
        var_expr_traits<T>::fixed_size;
        typename var_expr_traits<T>::value_t;
        typename var_expr_traits<T>::index_t;
    } &&
    requires(typename var_expr_traits<T>::index_t offset,
             T& x) {
       { x.set_cache_offset(offset) } -> std::same_as<
               typename var_expr_traits<T>::index_t 
               >;
    } &&
    (
        ( 
            !util::is_mat_v<T> &&
            requires (const MockVector<typename var_expr_traits<T>::value_t>& values,
                      const MockVector< ad::Var<
                        typename var_expr_traits<T>::value_t> >& ad_vars,
                      const T& cx, 
                      size_t i) {
                { cx.value(values, i) } -> std::convertible_to<
                    typename var_expr_traits<T>::value_t>;
                { cx.to_ad(ad_vars, ad_vars, i) } -> ad::is_ad_expr;
            }
        ) ||       
        ( 
            util::is_mat_v<T> &&
            requires (const MockVector<typename var_expr_traits<T>::value_t>& values,
                      const MockVector< ad::Var<
                        typename var_expr_traits<T>::value_t> >& ad_vars,
                      const T& cx, 
                      size_t i) {
                { cx.value(values, i, i) } -> std::convertible_to<
                    typename var_expr_traits<T>::value_t>;
                { cx.to_ad(ad_vars, ad_vars, i, i) } -> ad::is_ad_expr;
            }
        ) 
    )
    ;
```

- a variable expression is a `shape`

- must derive from `VarExprBase<Derived>` where `Derived` is the type of the expression 

- `has_param`: `static constexpr bool` member that indicates whether
the expression contains any references to a `parameter` (described in [Variable](#variable) section).

- `fixed_size`: `static constexpr size_t` member that indicates whether
the expression is of fixed size (known at compile-time). This may be used by expressions
which can optimize performance if `fixed_size > 0`. Expressions whose size is not
fixed must have it set to `0`.

- `value_t`: member alias that aliases the underlying data type (usually `double` or `int`).

- `index_t`: index type in order to access various types of vectors
(see below under `value`, `to_ad`, `set_cache_offset`). It is usually `uint32_t`.

- `set_cache_offset`: member function that may choose to assign itself a region of the 
AD variable cache vector (see `to_ad`). It must return the next offset.
Hence, if it does not need the cache, it must be the identity function,
i.e. simply returns the `offset` parameter. 
Otherwise, if it needs `n` cache variables, return `offset + n`.

- `value`: evaluates ith element of the scalar or vector expression 
or (i,j)th element of the matrix expression, using the values stored in `values`.
The parameter `values` should only be used by `variable` objects (see [Variable](#variable)).
All other variable expressions usually delegate the parameters to its children in the expression tree.

- `to_ad`: converts its expression into AD expression.
The first parameter is the vector of AD variables which you want the expression to build off of.
Later when we differentiate the AD expression, user will be interested in collecting the values
and adjoints of these variables.
The second parameter is a vector of AD variables used to cache any intermediate steps
if certain variable expressions find it necessary for performance boost.
For example, in `ppl::dot(X,w)`, the naive approach of getting the ith expression is something like
```cpp
ad::sum(begin, end, [](...) { return X.to_ad(i,j) * w.to_ad(j); });
```
Note that in general `X` and `w` could be complicated expressions.
Especially in these cases, we would copy such expressions for `w` `n` times where `n`
is the number of rows of `X`.
During the differentiation, we would evaluate the same thing `n` times.
Since `n` can get large in practice, this kills performance.
Ideally, since this `dot` node knows that `w` expression evaluations can get reused,
it should cache these results.
By using the second parameter to `to_ad`, since we know that the range [offset, offset+n) 
is uniquely reserved for this node from calling `set_cache_offset` before,
we can cache the results using this range of cache vector like:
```cpp
(
ad::for_each(offset, offset+n, [](){ return cache[j] = w.to_ad(j); }),
ad::sum(begin, end, [](...) { return X.to_ad(i,j) * w.to_ad(j); });
)
```
We cannot return this AD expression for every node i since
in that case, we would be evaluating the `for_each` `n` times,
not solving the problem we intended to solve.
But we can return this node only when `i==0` and for all other `i`,
simply return that expression but change the first expression to
```cpp
ad::for_each(offset, offset, [](){ return cache[j] = w.to_ad(j); }),
```
Note that this is a dummy `for_each` that doesn't do anything,
effectively not computing anything.
Benchmark shows that performance is really saved by large orders of magnitude.

#### Variable

A `variable` is a special case of `variable expression`.
They are like the leaves of the expression tree.
Specifically, objects representing a `parameter` or `data` are what we call `variable`.

The concepts are defined as follows:

```cpp

template <class T>
concept data_c =
    var_expr_c<T> &&
    data_is_base_of_v<T> &&
    requires (const T cx, size_t i) {
        typename var_traits<T>::id_t;
        { cx.id() } -> std::same_as<typename var_traits<T>::id_t>;
    }
    ;

template <class T>
concept param_c = 
    var_expr_c<T> &&
    param_is_base_of_v<T> &&
    requires () {
        typename var_traits<T>::id_t;
        typename param_traits<T>::pointer_t;
        typename param_traits<T>::const_pointer_t;
    } &&
    requires (T x, const T cx, size_t i,
              typename param_traits<T>::index_t offset) {
        { x.set_offset(offset) } -> std::same_as<
                typename var_traits<T>::index_t 
                >;
        { cx.storage(i) } -> std::convertible_to<typename param_traits<T>::pointer_t>;
        { cx.id() } -> std::same_as<typename var_traits<T>::id_t>;
    }
    ;

template <class T>
concept var_c = 
    data_c<T> ||
    param_c<T>
    ;   
```

- must be a `variable expression`

- derive from `DataBase<Derived>` or `ParamBase<Derived>`, respectively

- `id_t`: every variable has an ID that will mainly be used to check model construction.
It is one way to know when multiple objects refer to the same entity.

- `pointer_t`: underlying value pointer type. If value type is `double`
then `pointer_t` will likely be `double*`. 

- `const_pointer_t`: similar to `pointer_t` but one that has a notion that 
the pointee is read-only.

- `set_offset`: similar to `set_cache_offset` in [Variable Expression section](#variable-expression),
this sets the offset of the vector of AD variables that gets passed into `to_ad`.
The logic is exactly the same as `set_cache_offset`.
The only difference is that `set_offset` will only do something significant if the `variable`
is a `parameter`, since inference is w.r.t. `parameter`s and not `data`.

- `storage`: const member function that returns the ith storage pointer.
While the pointer itself is `const`, it can modify the pointee.
Later inference algorithms will walk through the model expression and store each sample
by dereferencing these storage pointers.

- `id`: const member function that returns the ID of the variable.

### Distribution Expression

`Distribution expression`s build on top of `variable expression`s.
In detail, the parameters to the distributions will be `variable expression`s.
For example,

```cpp
ppl::normal(x + w, s * s);
```

A `distribution expression` contains logic about how to evaluate its
pdf, log_pdf, and generate the corresponding AD expression for the log pdf.

```cpp
template <class T>
concept dist_expr_c = 
    dist_expr_is_base_of_v<T> &&
    requires () {
        typename dist_expr_traits<T>::value_t;
        typename dist_expr_traits<T>::dist_value_t;
        typename dist_expr_traits<T>::index_t;
    } &&
    requires(typename var_expr_traits<T>::index_t offset,
             T& x) {
       { x.set_cache_offset(offset) } -> std::same_as<
               typename dist_expr_traits<T>::index_t 
               >;
    } &&
    (
        requires (const ppl::Param<typename dist_expr_traits<T>::value_t, ppl::scl>& p,
                  const MockVector<typename dist_expr_traits<T>::value_t>& v,
                  const T& cx,
                  size_t i) {
            { cx.pdf(p, v) } -> std::same_as<typename dist_expr_traits<T>::dist_value_t>;
            { cx.log_pdf(p, v) } -> std::same_as<typename dist_expr_traits<T>::dist_value_t>;
            { cx.min(v, i) } -> std::same_as<typename dist_expr_traits<T>::value_t>;
            { cx.max(v, i) } -> std::same_as<typename dist_expr_traits<T>::value_t>;
        } ||
        requires (const ppl::Param<typename dist_expr_traits<T>::value_t, ppl::vec>& p,
                  const MockVector<typename dist_expr_traits<T>::value_t>& v,
                  const T& cx,
                  size_t i) {
            { cx.pdf(p, v) } -> std::same_as<typename dist_expr_traits<T>::dist_value_t>;
            { cx.log_pdf(p, v) } -> std::same_as<typename dist_expr_traits<T>::dist_value_t>;
            { cx.min(v, i) } -> std::same_as<typename dist_expr_traits<T>::value_t>;
            { cx.max(v, i) } -> std::same_as<typename dist_expr_traits<T>::value_t>;
        }
    )
    ;
```

- `pdf`: returns the value of the (joint) pdf calculated at
the point `p` and
the parameter values in `v`. This only makes sense when the distribution
represents independent variables. This API may have to change in the future.

- `log_pdf`: returns the value of the (joint) log pdf calculated at
the point `p` and
the parameter values in `v`. This only makes sense when the distribution
represents independent variables. This API may have to change in the future.

- ` min`: returns the minimum possible ith random variable value.
Such value may depend on the distribution parameters, and hence,
must supply vector of model parameter values.

- ` min`: returns the maximum possible ith random variable value.
Such value may depend on the distribution parameters, and hence,
must supply vector of model parameter values.

- `ad_log_pdf` (not listed above yet): returns the AD expression representing the log pdf
built using the first parameter as the "point at which to evaluate log pdf",
the second parameter as the vector of AD variables associated with the parameters in the model,
and the third parameter as the vector of AD cache variables that can be used by any
variable expressions intermediate.
For efficiency sake, the AD expression _does not_ have to represent
the entire log pdf - it can omit constants.
We expect that users will not be needing this AD expression to compute 
the log pdf, and that only inference algorithms will rely on this member function.
If user wishes to compute the actual log pdf, they do not have to deal with AD expressions
and can just directly compute it using `log_pdf`.

### Model Expression

A `model expression` is one that combines `distribution expression`s,
`variable`s, and `model expression`s.
They mainly delegate calls in a proper ordering,
but otherwise do not do much.
We expect that users will not have to make any other model expressions
than what we provide.

Nonetheless, we provide the concept:

```cpp
template <class T>
concept model_expr_c =
    model_expr_is_base_of_v<T> &&
    requires (const MockVector<double>& v,
              const MockVector<ad::Var<double>>& ad_vars,
              const T& cx) {
        typename model_expr_traits<T>::dist_value_t;
        { cx.pdf(v) } -> std::same_as<typename model_expr_traits<T>::dist_value_t>;
        { cx.log_pdf(v) } -> std::same_as<typename model_expr_traits<T>::dist_value_t>;
        { cx.ad_log_pdf(ad_vars, ad_vars) } -> ad::is_ad_expr;
    }
    ;
```

#### EqNode

An `EqNode` represents the assignment of a distribution to a variable.
It is syntactically written as:

```cpp
x |= distribution(parameters...)
```

Such node is created when using `operator|=` with a `variable`
on the left side with a `distribution expression` on the right.

When invoking the `pdf`, `log_pdf`, or `ad_log_pdf` calls,
they will simply delegate the respective calls to the underlying
distribution node by evaluating at whatever value `x` is -
this is precisely the first parameter of all three calls.

#### GlueNode

A `GlueNode` combines `EqNode`s to create the final model expression.

```cpp
w |= ppl::uniform(0., 1.),
x |= ppl::normal(2.*w, w * w)
```

It is created by combining model expressions with `operator,`.
