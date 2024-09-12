#pragma once

#include <algorithm>
#include <cassert>
#include <vector>

#include "ContArray.hpp"

template <typename T>
class FastSubsetConvolutionForDP {
public:
  FastSubsetConvolutionForDP(unsigned n) : n(n), N(1ULL << n) {
    // Compute the subset sums for `K` = 0 and `K` = 1. Init the array.
    store.assign(n + 1, std::vector<T>(N, T()));

    // Set the empty subset. This will all values to `\infty`.
    // store[0][0] = T();
  }

  std::vector<T> bounded_convolution(const std::vector<T>& f, const unsigned K) {
    // Apply the Zeta transform.
    bounded_zeta(f, K - 1);

    // Compute the Moebius transform of the product.
    return bounded_moebius_of_prod(K);
  }

private:
  void bounded_zeta(const std::vector<T> &f, const unsigned dim) {
    // We exploit the fact that the DP-table for subsets of size less than `K` does _not_ change!
    // So there is no benefit of iterating _all_ subsets, but only those of size _exactly_ `K`.

    // Iterate only the subsets with `K` bits.
    for (int64_t mask = (1ULL << dim) - 1, r, c; mask < (1ULL << n); c = mask & -mask, r = mask + c, mask = r | (((r ^ mask) >> 2) / c)) {
      store[dim][mask] = f[mask];
    }

    // And apply the zeta transform only for this level.
    bounded_subset_sum(store[dim], dim);
  }

  void bounded_subset_sum(std::vector<T> &f, const unsigned dim) {
    // In this case, we apply the transform onto `ret`, where only the subsets of size _exactly_ `K` are not empty.
    // This means we do not need to iterate the subsets of size less than `K`, but only those of size greater than `K`.
    for (unsigned d = 0; d != n; ++d) {
      // Iterate subsets of size > `dim`.
      // This is because the lower ones are simply zeros and those of size exactly `dim` are the baseline.
      // Note that we simply skip those sets where we're sure their popcount is <= dim.
      for (uint64_t S = (1ULL << (dim + 1)) - 1; S != N; ++S) {
        if ((__builtin_popcountll(S) > dim) && (S & (1ULL << d)))
          f[S] += f[S ^ (1ULL << d)];
      }
    }

#if 0
    for (uint64_t S = 0; S != N; ++S) {
      std::cerr << S << ": " << f[S] << std::endl;
    }
#endif
  }

  void bounded_inv_subset_sum(std::vector<T> &g, const unsigned dim) {
    // Since we only need the results for subsets of size exactly `K`, we do not need the update the subsets of size greater than `K`.
    for (int d = 0; d != n; ++d) {
      for (uint64_t S = 0; S != N; ++S) {
        if ((__builtin_popcountll(S) <= dim) && (S & (1ULL << d))) {
          g[S] -= g[S ^ (1ULL << d)];
        }
      }
    }
  }

  std::vector<T> bounded_moebius_of_prod(const unsigned K) {
    std::vector<T> tmp(N, T());

    // Exploit the fact that for `mat[e]`, all `store[e][S]` with |S| < e are 0.
    // This for-loop is restrict to `K / 2` due to symmetry of `store`.
    for (unsigned e = 0; e <= K / 2; ++e) {
      // How we compute this:
      // `store[e][S]` != 0 <-> |S| >= e.
      // `store[K - e][S]` != 0 <-> |S| >= K - e
      // Hence, we should only consider those subsets with |S| >= max(e, K - e).
      unsigned limit = std::max(e, K - e);

      // Note that we directly skip those subsets the popcount of which is less than `limit`.
      for (uint64_t S = (1ull << limit) - 1; S != N; ++S) {
        if ((limit <= __builtin_popcountll(S)) && (__builtin_popcountll(S) <= K)) {
          auto curr = store[e][S] * store[K - e][S];

          // Only double the result if we are symmetric.
          if (e != K - e)
            curr *= 2;

          tmp[S] += curr;
        }
      }
    }

    // Apply the inverse transform.
    bounded_inv_subset_sum(tmp, K);

    // And return.
    return tmp;
  }

private:
  unsigned n;
  uint64_t N;
  std::vector<std::vector<T>> store;
};

template <unsigned w, unsigned kForcedShift>
class OptimizedFastSubsetConvolutionForDP {
public:
  OptimizedFastSubsetConvolutionForDP(unsigned n) : n(n), N(1ULL << n) {
    // Compute the subset sums for `K` = 0 and `K` = 1. Init the array.
    store.assign(n + 1, ContArray<w, kForcedShift>(N));

    // Set the empty subset. This will all values to `\infty`.
    // store[0][0] = T();
  }

  ContArray<w, kForcedShift> bounded_convolution(ContArray<w, kForcedShift>& f, const unsigned K) {
    // Apply the Zeta transform.
    bounded_zeta(f, K - 1);

    // Compute the Moebius transform of the product.
    return bounded_moebius_of_prod(K);
  }

  void precomputeDimension(ContArray<w, kForcedShift>& f, const unsigned K) {
    // Only compute the zeta function for `K - 1` to have it for later.
    bounded_zeta(f, K - 1);
  }

private:
  void bounded_zeta(ContArray<w, kForcedShift> &f, const unsigned dim) {
    // We exploit the fact that the DP-table for subsets of size less than `K` does _not_ change!
    // So there is no benefit of iterating _all_ subsets, but only those of size _exactly_ `K`.

    // Iterate only the subsets with `K` bits.
    for (int64_t mask = (1ULL << dim) - 1, r, c; mask < (1ULL << n); c = mask & -mask, r = mask + c, mask = r | (((r ^ mask) >> 2) / c)) {
      // std::cerr << "mask=" << mask << " f[mask]=" << f[mask] << std::endl;
      store[dim][mask].copy(f[mask]);
    }

    // And apply the zeta transform only for this level.
    bounded_subset_sum(store[dim], dim);
  }

  void bounded_subset_sum(ContArray<w, kForcedShift> &f, const unsigned dim) {
    // In this case, we apply the transform onto `ret`, where only the subsets of size _exactly_ `K` are not empty.
    // This means we do not need to iterate the subsets of size less than `K`, but only those of size greater than `K`.
    for (unsigned d = 0; d != n; ++d) {
      // Iterate subsets of size > `dim`.
      // This is because the lower ones are simply zeros and those of size exactly `dim` are the baseline.
      for (uint64_t S = 0; S != N; ++S) {
        if ((__builtin_popcountll(S) > dim) && (S & (1ULL << d)))
          f[S] += f[S ^ (1ULL << d)];
      }
    }

#if 0
    for (uint64_t S = 0; S != N; ++S) {
      std::cerr << S << ": " << f[S] << std::endl;
    }
#endif
  }

  void bounded_inv_subset_sum(ContArray<w, kForcedShift> &g, const unsigned dim) {
    // Since we only need the results for subsets of size exactly `K`, we do not need the update the subsets of size greater than `K`.
    for (int d = 0; d != n; ++d) {
      for (uint64_t S = 0; S != N; ++S) {
        if ((__builtin_popcountll(S) <= dim) && (S & (1ULL << d)))
          g[S] -= g[S ^ (1ULL << d)];
      }
    }
  }

  ContArray<w, kForcedShift> bounded_moebius_of_prod(const unsigned K) {
    ContArray<w, kForcedShift> tmp(N);

    // Exploit the fact that for `mat[e]`, all `store[e][S]` with |S| < e are 0.
    // This for-loop is restrict to `K / 2` due to symmetry of `store`.
    for (unsigned e = 0; e <= K / 2; ++e) {
      // How we compute this:
      // `store[e][S]` != 0 <-> |S| >= e.
      // `store[K - e][S]` != 0 <-> |S| >= K - e
      // Hence, we should only consider those subsets with |S| >= max(e, K - e).
      unsigned limit = std::max(e, K - e);
      for (uint64_t S = 0; S != N; ++S) {
        if ((limit <= __builtin_popcountll(S)) && (__builtin_popcountll(S) <= K)) {
          // Like a vector.
          // These are two chunks.
          if (e == K - e) {
            tmp[S].template update<1>(store[e][S], store[K - e][S]);
          } else {
            tmp[S].template update<2>(store[e][S], store[K - e][S]);
          }
        }
      }
    }

    // Apply the inverse transform.
    bounded_inv_subset_sum(tmp, K);

    // And return.
    return tmp;
  }

private:
  unsigned n;
  uint64_t N;
  unsigned logMaxCard;
  std::vector<ContArray<w, kForcedShift>> store;
};

// Note: Really important the order of paranthesis here since otherwise we might overflow.
#define STORE_SIZE(s) (N - ((1ULL << s) - 1))

// Note: Really important the order of paranthesis here since otherwise we might overflow.
#define STORE_SHIFT(e, S) (S - ((1ULL << e) - 1)) 

class BooleanFSC {
  using data_type_t = uint32_t;
  using supp_t = std::vector<data_type_t>;
public:
  // An auxiliary array.
  supp_t tmp;

  BooleanFSC(unsigned n) : n(n), N(1ULL << n) {
    // Compute the subset sums for `K` = 0 and `K` = 1. Init the array.
    store.assign(n - 2, supp_t(N, 0));
    tmp.resize(N);

    // Set the empty subset. This will all values to `\infty`.
    // store[0][0] = T();
  }

  void bounded_convolution(const unsigned K) {
    // Apply the Zeta transform.
    bounded_zeta(K - 1);

    // Compute the Moebius transform of the product.
    bounded_moebius_of_prod(K);
  }

  void fill_positions(const std::vector<data_type_t>& f, const unsigned dim) {
    for (int64_t mask = (1ULL << dim) - 1, r, c; mask < (1ULL << n); c = mask & -mask, r = mask + c, mask = r | (((r ^ mask) >> 2) / c)) {
      get(dim, mask) = f[mask];
    }
  }

  void precompute_dimension(const std::vector<data_type_t>& f, const unsigned K) {
    // Only compute the zeta function for `K - 1` to have it for later.
    if (K - 1 <= 1)
      return;

    fill_positions(f, K - 1);

    bounded_zeta(K - 1);
  }

  inline data_type_t fetch(unsigned e, uint64_t S) const {
    // assert(e && e < n);
    if (e == 1)
      return __builtin_popcountll(S);
    return store[e - 2][S];
  }

  inline data_type_t& get(unsigned e, uint64_t S) {
    return store[e - 2][S];
  }

private:
  void bounded_zeta(const unsigned dim) {
    // We exploit the fact that the DP-table for subsets of size less than `K` does _not_ change!
    // So there is no benefit of iterating _all_ subsets, but only those of size _exactly_ `K`.
    if (dim <= 1) {
      return;
    }

    // And apply the zeta transform only for this level.
    bounded_subset_sum(dim);
  }

  void bounded_subset_sum(const unsigned dim) {
    // In this case, we apply the transform onto `ret`, where only the subsets of size _exactly_ `K` are not empty.
    // This means we do not need to iterate the subsets of size less than `K`, but only those of size greater than `K`.
    assert(dim >= 2);

    for (unsigned d = 0; d != n; ++d) {
      // Iterate subsets of size > `dim`.
      // This is because the lower ones are simply zeros and those of size exactly `dim` are the baseline.
      // Moreover, we simply skip those sets where we're sure their popcount is <= dim.

      for (uint64_t S = (1ULL << (dim + 1)) - 1; S != N; ++S) {
        if ((__builtin_popcountll(S) > dim) && (S & (1ULL << d))) {
          get(dim, S) += get(dim, S ^ (1ULL << d));
        }
      }
    }
  }

  void bounded_inv_subset_sum(supp_t &g, const unsigned dim) {
    // Since we only need the results for subsets of size exactly `K`, we do not need the update the subsets of size greater than `K`.
    for (int d = 0; d != n; ++d) {
      for (uint64_t S = 0; S != N; ++S) {
        if ((__builtin_popcountll(S) <= dim) && (S & (1ULL << d)))
          g[S] -= g[S ^ (1ULL << d)];
      }
    }
  }

  void bounded_moebius_of_prod(const int K) {
    // Reset `tmp`.
    std::fill(tmp.begin(), tmp.end(), 0);

    // Exploit the fact that for `mat[e]`, all `store[e][S]` with |S| < e are 0.
    // This for-loop is restrict to `K / 2` due to symmetry of `store`.

    for (unsigned e = 1; e <= K / 2; ++e) {
      // How we compute this:
      // `store[e][S]` != 0 <-> |S| >= e.
      // `store[K - e][S]` != 0 <-> |S| >= K - e
      // Hence, we should only consider those subsets with |S| >= max(e, K - e).
      unsigned limit = std::max(e, K - e);

      // Moreover, we directly skip those subsets the popcount of which is less than `limit`.
      for (uint64_t S = (1ULL << limit) - 1; S != N; ++S) {
        if ((limit <= __builtin_popcountll(S)) && (__builtin_popcountll(S) <= K)) {
          tmp[S] += (1 + (e != (K - e))) * fetch(e, S) * fetch(K - e, S);
        }
      }
    }

    // Apply the inverse transform.
    bounded_inv_subset_sum(tmp, K);
  }

private:
  unsigned n;
  uint64_t N;
  unsigned logMaxCard;
  std::vector<supp_t> store;
};

class BoostedBooleanFSC {
  using data_type_t = uint32_t;
  using supp_t = std::vector<data_type_t>;
public:
  // An auxiliary array.
  supp_t tmp;

  BoostedBooleanFSC(unsigned n) : n(n), N(1ULL << n) {
    // Compute the subset sums for `K` = 0 and `K` = 1. Init the array.
    store.assign(n - 2, supp_t(N, 0));
    tmp.resize(N);

    // Set the empty subset. This will all values to `\infty`.
    // store[0][0] = T();
  }

  void bounded_convolution(const unsigned K) {
    // Apply the Zeta transform.
    bounded_zeta(K - 1);

    // Compute the Moebius transform of the product.
    bounded_moebius_of_prod(K);
  }

  void fill_positions(const std::vector<data_type_t>& f, const unsigned dim) {
    for (int64_t mask = (1ULL << dim) - 1, r, c; mask < (1ULL << n); c = mask & -mask, r = mask + c, mask = r | (((r ^ mask) >> 2) / c)) {
      get(dim, mask) = f[mask];
    }
  }

  void precompute_dimension(const std::vector<data_type_t>& f, const unsigned K) {
    // Only compute the zeta function for `K - 1` to have it for later.
    if (K - 1 <= 1)
      return;

    fill_positions(f, K - 1);

    bounded_zeta(K - 1);
  }

  inline data_type_t fetch(unsigned e, uint64_t S) const {
    if (e == 1)
      return __builtin_popcountll(S);
    return store[e - 2][S];
  }

  inline data_type_t& get(unsigned e, uint64_t S) {
    return store[e - 2][S];
  }

private:
  void bounded_zeta(const unsigned dim) {
    // We exploit the fact that the DP-table for subsets of size less than `K` does _not_ change!
    // So there is no benefit of iterating _all_ subsets, but only those of size _exactly_ `K`.
    if (dim <= 1) {
      return;
    }

    // And apply the zeta transform only for this level.
    bounded_subset_sum(dim);
  }

  void bounded_subset_sum(const unsigned dim) {
    // In this case, we apply the transform onto `ret`, where only the subsets of size _exactly_ `K` are not empty.
    // This means we do not need to iterate the subsets of size less than `K`, but only those of size greater than `K`.
    assert(dim >= 2);

    for (unsigned d = 0; d != n; ++d) {
      // Iterate subsets of size > `dim`.
      // This is because the lower ones are simply zeros and those of size exactly `dim` are the baseline.
      // Moreover, we simply skip those sets where we're sure their popcount is <= dim.

      for (uint64_t S = (1ULL << (dim + 1)) - 1; S != N; ++S) {
        if ((__builtin_popcountll(S) > dim) && (S & (1ULL << d))) {
          get(dim, S) += get(dim, S ^ (1ULL << d));
        }
      }
    }
  }

  void bounded_inv_subset_sum(supp_t &g, const unsigned lb, const unsigned dim) {
    // Since we only need the results for subsets of size exactly `K`, we do not need the update the subsets of size greater than `K`.
    auto start_subset = (1ULL << (lb + 1)) - 1;
    for (int d = 0; d != n; ++d) {
      for (uint64_t S = start_subset; S != N; ++S) {
        if ((__builtin_popcountll(S) <= dim) && (S & (1ULL << d)))
          g[S] -= g[S ^ (1ULL << d)];
      }
    }
  }

  void bounded_moebius_of_prod(const int K) {
    std::fill(tmp.begin(), tmp.end(), 0);

    // Exploit the fact that for `mat[e]`, all `store[e][S]` with |S| < e are 0.
    // This for-loop is restrict to `K / 2` due to symmetry of `store`.

    for (unsigned e = 1; e <= K / 2; ++e) {
      // How we compute this:
      // `store[e][S]` != 0 <-> |S| >= e.
      // `store[K - e][S]` != 0 <-> |S| >= K - e
      // Hence, we should only consider those subsets with |S| >= max(e, K - e).
      unsigned limit = std::max(e, K - e);

      // Moreover, we directly skip those subsets the popcount of which is less than `limit`.
      for (uint64_t S = (1ULL << limit) - 1; S != N; ++S) {
        if ((limit <= __builtin_popcountll(S)) && (__builtin_popcountll(S) <= K)) {
          tmp[S] += (1 + (e != (K - e))) * fetch(e, S) * fetch(K - e, S);
        }
      }
    }

    // Apply the inverse transform.
    auto touched_lower_bound_size = (K + 1) / 2;
    bounded_inv_subset_sum(tmp, touched_lower_bound_size, K);
  }

private:
  unsigned n;
  uint64_t N;
  unsigned logMaxCard;
  std::vector<supp_t> store;
};

template <typename T>
class NaiveFastSubsetConvolutionForDP {
public:
  NaiveFastSubsetConvolutionForDP(unsigned n) : n(n), N(1ULL << n) {}

  std::vector<T> naive_subset_convolution(const std::vector<T>& f) {
    // Apply the zeta transform.
    auto ff = build_zeta_(n + 1, f);

    // Apply the Moebius transform.
    return custom_moebius_of_prod_(ff);
  }

private:
  std::vector<T> custom_moebius_of_prod(const std::vector<std::vector<T>> &mat1) {
    std::vector<std::vector<unsigned>> pc2i(n + 1);
    for (int i = 0; i != N; i++) pc2i[__builtin_popcount(i)].push_back(i);

    std::vector<T> tmp, ret(mat1[0].size());
    for (unsigned d = 0; d <= n; d++) {
      tmp.assign(mat1[d].size(), T());
      for (unsigned e = 0; e <= d; e++)
        for (uint64_t i = 0; i != N; ++i)
          tmp[i] += mat1[e][i] * mat1[d - e][i];

      // Apply the Moebius transform.
      subset_sum_inv(tmp);

      // Build the final result `ret` and also reset `tmp`.
      for (auto i : pc2i[d]) ret[i] = tmp[i];
    }
    return ret;
  }

private:
  unsigned n;
  uint64_t N;
};

// Subset sum (fast zeta transform)
// Complexity: O(N 2^N) for array of size 2^N
template <typename T> void subset_sum(std::vector<T> &f) {
    const int sz = f.size(), n = __builtin_ctz(sz);
    assert(__builtin_popcount(sz) == 1);
    for (int d = 0; d < n; d++) {
      for (int S = 0; S < 1 << n; S++)
        if (S & (1 << d)) f[S] += f[S ^ (1 << d)];
    }
}

// Inverse of subset sum (fast moebius transform)
// Complexity: O(N 2^N) for array of size 2^N
template <typename T> void subset_sum_inv(std::vector<T> &g) {
    const int sz = g.size(), n = __builtin_ctz(sz);
    assert(__builtin_popcount(sz) == 1);
    for (int d = 0; d < n; d++) {
        for (int S = 0; S < 1 << n; S++)
            if (S & (1 << d)) g[S] -= g[S ^ (1 << d)];
    }
}

// Superset sum / its inverse (fast zeta/moebius transform)
// Complexity: O(N 2^N) for array of size 2^N
template <typename T> void superset_sum(std::vector<T> &f) {
    const int sz = f.size(), n = __builtin_ctz(sz);
    assert(__builtin_popcount(sz) == 1);
    for (int d = 0; d < n; d++) {
        for (int S = 0; S < 1 << n; S++)
            if (!(S & (1 << d))) f[S] += f[S | (1 << d)];
    }
}
template <typename T> void superset_sum_inv(std::vector<T> &g) {
    const int sz = g.size(), n = __builtin_ctz(sz);
    assert(__builtin_popcount(sz) == 1);
    for (int d = 0; d < n; d++) {
        for (int S = 0; S < 1 << n; S++)
            if (!(S & (1 << d))) g[S] -= g[S | (1 << d)];
    }
}

template <typename T> std::vector<std::vector<T>> build_zeta_(int D, const std::vector<T> &f) {
    int n = f.size();
    std::vector<std::vector<T>> ret(D, std::vector<T>(n));
    for (int i = 0; i < n; i++) ret[__builtin_popcount(i)][i] += f[i];
    for (auto &vec : ret) subset_sum(vec);
    return ret;
}

template <typename T>
std::vector<T> get_moebius_of_prod_(const std::vector<std::vector<T>> &mat1,
                                    const std::vector<std::vector<T>> &mat2) {
    int D = mat1.size(), n = mat1[0].size();
    std::vector<std::vector<int>> pc2i(D);
    for (int i = 0; i < n; i++) pc2i[__builtin_popcount(i)].push_back(i);

    std::vector<T> tmp, ret(mat1[0].size());
    for (int d = 0; d < D; d++) {
        tmp.assign(mat1[d].size(), T());
        for (int e = 0; e <= d; e++) {
            for (int i = 0; i < int(tmp.size()); i++) tmp[i] += mat1[e][i] * mat2[d - e][i];
        }
        subset_sum_inv(tmp);

        // Build the final result `ret` and also reset `tmp`.
        for (auto i : pc2i[d]) ret[i] = tmp[i];
    }
    return ret;
}

// Subset convolution
// h[S] = \sum_T f[T] * g[S - T]
// Complexity: O(N^2 2^N) for arrays of size 2^N
template <typename T> std::vector<T> subset_convolution(std::vector<T> f, std::vector<T> g) {
    const int sz = f.size(), m = __builtin_ctz(sz) + 1;
    assert(__builtin_popcount(sz) == 1 and f.size() == g.size());
    std::cerr << "m=" << m << " vs sz=" << sz << std::endl;
    auto ff = build_zeta_(m, f), fg = build_zeta_(m, g);
    return get_moebius_of_prod_(ff, fg);
}

// https://hos-lyric.hatenablog.com/entry/2021/01/14/201231
template <class T, class Function> void subset_func(std::vector<T> &f, const Function &func) {
    const int sz = f.size(), m = __builtin_ctz(sz) + 1;
    assert(__builtin_popcount(sz) == 1);

    auto ff = build_zeta_(m, f);

    std::vector<T> p(m);
    for (int i = 0; i < sz; i++) {
        for (int d = 0; d < m; d++) p[d] = ff[d][i];
        func(p);
        for (int d = 0; d < m; d++) ff[d][i] = p[d];
    }

    for (auto &vec : ff) subset_sum_inv(vec);
    for (int i = 0; i < sz; i++) f[i] = ff[__builtin_popcount(i)][i];
}

// log(f(x)) for f(x), f(0) == 1
// Requires inv()
template <class T> void poly_log(std::vector<T> &f) {
    assert(f.at(0) == T(1));
    static std::vector<T> invs{0};
    const int m = f.size();
    std::vector<T> finv(m);
    for (int d = 0; d < m; d++) {
        finv[d] = (d == 0);
        if (int(invs.size()) <= d) invs.push_back(T(d).inv());
        for (int e = 0; e < d; e++) finv[d] -= finv[e] * f[d - e];
    }
    std::vector<T> ret(m);
    for (int d = 1; d < m; d++) {
        for (int e = 0; d + e < m; e++) ret[d + e] += f[d] * d * finv[e] * invs[d + e];
    }
    f = ret;
}

// log(f(S)) for set function f(S), f(0) == 1
// Requires inv()
// Complexity: O(n^2 2^n)
// https://atcoder.jp/contests/abc213/tasks/abc213_g
template <class T> void subset_log(std::vector<T> &f) { subset_func(f, poly_log<T>); }

// exp(f(S)) for set function f(S), f(0) == 0
// Complexity: O(n^2 2^n)
// https://codeforces.com/blog/entry/92183
template <class T> void subset_exp(std::vector<T> &f) {
    const int sz = f.size(), m = __builtin_ctz(sz);
    assert(sz == 1 << m);
    assert(f.at(0) == 0);
    std::vector<T> ret{T(1)};
    ret.reserve(sz);
    for (int d = 0; d < m; d++) {
        auto c = subset_convolution({f.begin() + (1 << d), f.begin() + (1 << (d + 1))}, ret);
        ret.insert(ret.end(), c.begin(), c.end());
    }
    f = ret;
}

// sqrt(f(x)), f(x) == 1
// Requires inv of 2
// Compelxity: O(n^2)
template <class T> void poly_sqrt(std::vector<T> &f) {
    assert(f.at(0) == T(1));
    const int m = f.size();
    static const auto inv2 = T(2).inv();
    for (int d = 1; d < m; d++) {
        if (~(d & 1)) f[d] -= f[d / 2] * f[d / 2];
        f[d] *= inv2;
        for (int e = 1; e < d - e; e++) f[d] -= f[e] * f[d - e];
    }
}

// sqrt(f(S)) for set function f(S), f(0) == 1
// Requires inv()
// https://atcoder.jp/contests/xmascon20/tasks/xmascon20_h
template <class T> void subset_sqrt(std::vector<T> &f) { subset_func(f, poly_sqrt<T>); }

// exp(f(S)) for set function f(S), f(0) == 0
template <class T> void poly_exp(std::vector<T> &P) {
    const int m = P.size();
    assert(m and P[0] == 0);
    std::vector<T> Q(m), logQ(m), Qinv(m);
    Q[0] = Qinv[0] = T(1);
    static std::vector<T> invs{0};

    auto set_invlog = [&](int d) {
        Qinv[d] = 0;
        for (int e = 0; e < d; e++) Qinv[d] -= Qinv[e] * Q[d - e];
        while (d >= int(invs.size())) {
            int sz = invs.size();
            invs.push_back(T(sz).inv());
        }
        logQ[d] = 0;
        for (int e = 1; e <= d; e++) logQ[d] += Q[e] * e * Qinv[d - e];
        logQ[d] *= invs[d];
    };
    for (int d = 1; d < m; d++) {
        Q[d] += P[d] - logQ[d];
        set_invlog(d);
        assert(logQ[d] == P[d]);
        if (d + 1 < m) set_invlog(d + 1);
    }
    P = Q;
}

// f(S)^k for set function f(S)
// Requires inv()
template <class T> void subset_pow(std::vector<T> &f, long long k) {
    auto poly_pow = [&](std::vector<T> &f) {
        const int m = f.size();
        if (k == 0) f[0] = 1, std::fill(f.begin() + 1, f.end(), T(0));
        if (k <= 1) return;
        int nzero = 0;
        while (nzero < int(f.size()) and f[nzero] == T(0)) nzero++;
        int rem = std::max<long long>((long long)f.size() - nzero * k, 0LL);
        if (rem == 0) {
            std::fill(f.begin(), f.end(), 0);
            return;
        }
        f.erase(f.begin(), f.begin() + nzero);
        f.resize(rem);
        const T f0 = f.at(0), f0inv = f0.inv(), f0pow = f0.pow(k);
        for (auto &x : f) x *= f0inv;
        poly_log(f);
        for (auto &x : f) x *= k;
        poly_exp(f);
        for (auto &x : f) x *= f0pow;
        f.resize(rem, 0);
        f.insert(f.begin(), m - int(f.size()), T(0));
    };
    subset_func(f, poly_pow);
}