#include "gtest/gtest.h"
#include <autoppl/util/traits/concept.hpp>

namespace ppl {
namespace util {

struct MockType
{};

struct MockType2
{
    using value_t = double;
    using pointer_t = double*;

    void pdf() {};
    void log_pdf() {};
};

struct concept_fixture : ::testing::Test
{
protected:
};

TEST_F(concept_fixture, has_type_value_t_v_false)
{
    static_assert(!has_type_value_t_v<int>);
    static_assert(!has_type_value_t_v<const char>);
    static_assert(!has_type_value_t_v<void>);
    static_assert(!has_type_value_t_v<MockType>);
}

TEST_F(concept_fixture, has_type_value_t_v_true)
{
    static_assert(has_type_value_t_v<MockType2>);
}

TEST_F(concept_fixture, has_type_pointer_t_v_false)
{
    static_assert(!has_type_pointer_t_v<int>);
    static_assert(!has_type_pointer_t_v<const char>);
    static_assert(!has_type_pointer_t_v<void>);
    static_assert(!has_type_pointer_t_v<MockType>);
}

TEST_F(concept_fixture, has_type_pointer_t_v_true)
{
    static_assert(has_type_pointer_t_v<MockType2>);
}

//TEST_F(concept_fixture, has_func_pdf_v_false)
//{
//    static_assert(!has_func_pdf_v<int>);
//    static_assert(!has_func_pdf_v<const char>);
//    static_assert(!has_func_pdf_v<void>);
//    static_assert(!has_func_pdf_v<MockType>);
//}
//
//TEST_F(concept_fixture, has_func_pdf_v_true)
//{
//    static_assert(has_func_pdf_v<MockType2>);
//}
//
//TEST_F(concept_fixture, has_func_log_pdf_v_false)
//{
//    static_assert(!has_func_log_pdf_v<int>);
//    static_assert(!has_func_log_pdf_v<const char>);
//    static_assert(!has_func_log_pdf_v<void>);
//    static_assert(!has_func_log_pdf_v<MockType>);
//}
//
//TEST_F(concept_fixture, has_func_log_pdf_v_true)
//{
//    static_assert(has_func_log_pdf_v<MockType2>);
//}

} // namespace util
} // namespace ppl
