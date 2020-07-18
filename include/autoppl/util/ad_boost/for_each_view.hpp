// Disabled for the moment because not needed.

//#pragma once
//#include <iterator>
//#include <fastad_bits/node.hpp>
//
///*
// * ForEachView node is like ForEach node in FastAD
// * but doesn't allocate memory to save each expression.
// * Instead it assumes that the user provides iterators
// * that return a reference to existing expressions.
// * The user must be able to provide reverse iterators as well.
// */
//
//namespace ad {
//namespace core {
//namespace details {
//
//template <class Iter>
//using value_t = 
//    typename std::iterator_traits<Iter>::value_type::value_type;
//
//} // namespace details
//
//template <class FIter
//        , class RIter>
//struct ForEachView:
//    DualNum<details::value_t<FIter>>, 
//    ADNodeExpr<ForEachView<FIter, RIter>>
//{
//private:
//    using fvalue_t = typename 
//        std::iterator_traits<FIter>::value_type;
//    using rvalue_t = typename
//        std::iterator_traits<RIter>::value_type;
//    static_assert(std::is_same_v<fvalue_t, rvalue_t>);
//
//public:
//    using value_t = typename fvalue_t::value_type;
//    using data_t = DualNum<value_t>;
//
//    ForEachView(FIter fbegin, FIter fend,
//                RIter rbegin, RIter rend)
//        : data_t(0,0)
//        , fbegin_{fbegin}
//        , fend_{fend}
//        , rbegin_{rbegin}
//        , rend_{rend}
//    {}
//
//    value_t feval() 
//    {
//        if (fbegin_ == fend_) return 0;
//        auto last = std::prev(fend_);
//        std::for_each(fbegin_, last,
//                [](auto& expr) { expr.feval(); });
//        return this->set_value(last->feval());
//    }
//
//    void beval(value_t seed) 
//    {
//        if (rbegin_ == rend_) return;
//        this->set_adjoint(seed);
//        rbegin_->beval(seed);
//        std::for_each(std::next(rbegin_), rend_,
//                [](auto& expr) { expr.beval(); });
//    }
//
//private:
//    FIter fbegin_;
//    FIter fend_;
//    RIter rbegin_;
//    RIter rend_;
//};
//
//} // namespace core
//
//template <class FIter, class RIter>
//inline constexpr 
//auto for_each_view(FIter fbegin, FIter fend,
//                   RIter rbegin, RIter rend)
//{
//    return core::ForEachView<FIter, RIter>(
//            fbegin, fend, rbegin, rend);
//}
//
//} // namespace ad
