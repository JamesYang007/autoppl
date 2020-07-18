#include <gtest/gtest.h>
#include <autoppl/mcmc/hmc/var_adapter.hpp>

namespace ppl {
namespace mcmc {

struct var_adapter_fixture : ::testing::Test
{
protected:
    using diag_adapter_t = VarAdapter<diag_var>;
    arma::vec x = arma::zeros(1);
    arma::vec var = arma::zeros(1);

    size_t n_params = 1;

    void test_case_1(size_t warmup,
                     size_t init_buffer,
                     size_t term_buffer,
                     size_t window_base)
    {
        diag_adapter_t adapter(n_params, warmup, init_buffer,
                               term_buffer, window_base);

        bool res;
        for (size_t i = 0; i < warmup-1; ++i) {
            res = adapter.adapt(x, var);
            EXPECT_FALSE(res);
        }

        res = adapter.adapt(x, var);
        EXPECT_TRUE(res);
    }

    void test_case_2(size_t warmup,
                     size_t init_buffer,
                     size_t term_buffer,
                     size_t window_base)
    {
        diag_adapter_t adapter(n_params, warmup, init_buffer,
                               term_buffer, window_base);

        bool res;

        size_t new_init_buffer = 0.15 * warmup;
        size_t new_term_buffer = 0.1 * warmup;
        size_t new_window_base = warmup - new_init_buffer - new_term_buffer;

        // init buffer always returns false
        for (size_t i = 0; i < new_init_buffer; ++i) {
            res = adapter.adapt(x, var);
            EXPECT_FALSE(res);
        }

        // first window always returns false except at the very end
        for (size_t i = 0; i < new_window_base-1; ++i) {
            res = adapter.adapt(x, var);
            EXPECT_FALSE(res);
        }
        res = adapter.adapt(x, var);
        EXPECT_TRUE(res);

        // termination always returns false
        for (size_t i = 0; i < new_term_buffer; ++i) {
            res = adapter.adapt(x, var);
            EXPECT_FALSE(res);
        }
    }

    void test_case_3(size_t warmup,
                     size_t init_buffer,
                     size_t term_buffer,
                     size_t window_base)
    {
        diag_adapter_t adapter(n_params, warmup, init_buffer,
                               term_buffer, window_base);

        bool res;

        // init buffer always returns false
        for (size_t i = 0; i < init_buffer; ++i) {
            res = adapter.adapt(x, var);
            EXPECT_FALSE(res);
        }

        // Adapt for every window
        for (size_t i = init_buffer; 
                i < warmup - term_buffer; 
                window_base *= 2) {

            // check if at the last window that may have just been extended to term
            size_t window_end = (i + 3*window_base < warmup-term_buffer) ?
                init_buffer+window_base : warmup-term_buffer;

            // within window always returns false except at the very end
            for (; i < window_end - 1; ++i) {
                res = adapter.adapt(x, var);
                EXPECT_FALSE(res);
            }

            // reached last iteration of window - check that returns true
            res = adapter.adapt(x, var);
            EXPECT_TRUE(res);

            if (++i == warmup - term_buffer) break;
        }

        // termination always returns false
        for (size_t i = 0; i < term_buffer; ++i) {
            res = adapter.adapt(x, var);
            EXPECT_FALSE(res);
        }
    }
};

// Case 1: warmup <= 20
// Subcase 1: large term buffer
TEST_F(var_adapter_fixture, diag_ctor_case_11)
{
    size_t warmup = 10;
    size_t init_buffer = 1;
    size_t term_buffer = 13;
    size_t window_base = 4;
    test_case_1(warmup, init_buffer,
                term_buffer, window_base);
}

// Case 1: warmup <= 20
// Subcase 2: large init buffer
TEST_F(var_adapter_fixture, diag_ctor_case_12)
{
    size_t warmup = 10;
    size_t init_buffer = 9;
    size_t term_buffer = 0;
    size_t window_base = 5;
    test_case_1(warmup, init_buffer,
                term_buffer, window_base);
}

// Case 1: warmup <= 20
// Subcase 3: large window
TEST_F(var_adapter_fixture, diag_ctor_case_13)
{
    size_t warmup = 10;
    size_t init_buffer = 9;
    size_t term_buffer = 1;
    size_t window_base = 20;
    test_case_1(warmup, init_buffer,
                term_buffer, window_base);
}

// Case 2: 20 < warmup < init + window_base + term
// Subcase 1: large init buffer 
TEST_F(var_adapter_fixture, diag_ctor_case_21)
{
    size_t warmup = 100;
    size_t init_buffer = 110;
    size_t term_buffer = 10;
    size_t window_base = 10;
    test_case_2(warmup, init_buffer,
                term_buffer, window_base);
}

// Case 2: 20 < warmup < init + window_base + term
// Subcase 2: large init buffer 
TEST_F(var_adapter_fixture, diag_ctor_case_22)
{
    size_t warmup = 100;
    size_t init_buffer = 10;
    size_t term_buffer = 110;
    size_t window_base = 10;
    test_case_2(warmup, init_buffer,
                term_buffer, window_base);
}

// Case 2: 20 < warmup < init + window_base + term
// Subcase 3: large term buffer 
TEST_F(var_adapter_fixture, diag_ctor_case_23)
{
    size_t warmup = 100;
    size_t init_buffer = 50;
    size_t term_buffer = 10;
    size_t window_base = 110;
    test_case_2(warmup, init_buffer,
                term_buffer, window_base);
}

// Case 3: warmup >= init + window_base + term
// Subcase 1: large init buffer 
TEST_F(var_adapter_fixture, diag_ctor_case_31)
{
    size_t warmup = 100;
    size_t init_buffer = 50;
    size_t term_buffer = 10;
    size_t window_base = 30;
    test_case_3(warmup, init_buffer,
                term_buffer, window_base);
}

// Case 3: warmup >= init + window_base + term
// Subcase 2: large term buffer 
TEST_F(var_adapter_fixture, diag_ctor_case_32)
{
    size_t warmup = 100;
    size_t init_buffer = 5;
    size_t term_buffer = 80;
    size_t window_base = 10;
    test_case_3(warmup, init_buffer,
                term_buffer, window_base);

    term_buffer = 30;
    test_case_3(warmup, init_buffer,
                term_buffer, window_base);
}

// Case 3: warmup >= init + window_base + term
// Subcase 3: large window buffer 
TEST_F(var_adapter_fixture, diag_ctor_case_33)
{
    size_t warmup = 10031;
    size_t init_buffer = 63;
    size_t term_buffer = 59;
    size_t window_base = 1582;
    test_case_3(warmup, init_buffer,
                term_buffer, window_base);
}

} // namespace mcmc
} // namespace ppl
