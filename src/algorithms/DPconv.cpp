#include <vector>
#include <iostream>

#include "Util.hpp"
#include "TypeUtil.hpp"
#include "SubsetConvolution.hpp"

template <typename U> std::vector<U> naive_subset_convolution(std::vector<U> f, std::vector<U> g) {
  auto N = f.size();
  assert((N & (N - 1)) == 0);
  std::vector<U> h(N, 0);
  for (uint64_t S = 0, limit = N; S != limit; ++S) {
    uint64_t current = 0;
    do {
      h[S] += f[current] * g[S ^ current];
      current = (current - S) & S;
    } while (current);
  }
  return h;
}

template <class ConvolutionType, typename U>
static std::pair<U, std::shared_ptr<JoinNode>> runMinMaxDPconv_instant(const std::vector<U>& scalar_sizes, bool shouldExtractJoinTree = true) {
  unsigned N = scalar_sizes.size();
  unsigned n = static_cast<unsigned>(std::log2(N));

  auto permute = [&]() -> std::vector<uint32_t> {
    std::vector<uint32_t> perm(scalar_sizes.size());
    for (unsigned index = 0, limit = scalar_sizes.size(); index != limit; ++index)
      perm[index] = index;
    std::sort(perm.begin(), perm.end(), [&](auto a, auto b) { return scalar_sizes[a] > scalar_sizes[b]; });
    return perm;
  };

  // Compress the sizes.
  auto perm = permute();

  // Sort the sizes and find the position in the sorted order of size[2^n - 1].
  uint64_t full_card = scalar_sizes[N - 1];
  int pos_of_full_card = N, start_pos = -1;

  for (int index = 0, limit = N; index != limit; ++index) {
    if (scalar_sizes[perm[index]] == full_card) {
      pos_of_full_card = std::min(pos_of_full_card, index);
    }
    if (scalar_sizes[perm[index]] == std::numeric_limits<uint64_t>::max()) {
      start_pos = std::max(start_pos, index);
    }
  }

  std::vector<uint32_t> dp(N);
  for (unsigned index = 0; index != n; ++index)
    dp[1ull << index] = 1;

  auto is_solution = [&](int curr_pos, bool forceJoinTreeBuild = false) -> std::pair<bool, std::shared_ptr<JoinNode>>  {
    auto W = scalar_sizes[perm[curr_pos]];
    
    // Define the convolution.
    ConvolutionType fsc(n);

    // The DP.
    // Note: we hard-code the subset convolutions for `k <= 5` to improve the running time.
    // This is explained in the paper under paragraph: `Constant-Factor Optimizations`.
    for (unsigned k = 2; k <= n; ++k) {
      // Second layer.
      if (k == 2) {
        // Precompute the 2nd layer to have it for later layers.
        fsc.precompute_dimension(dp, 2);

        // And iterate all 2-bit masks.
        for (unsigned index1 = 0; index1 != n - 1; ++index1) {
          for (unsigned index2 = index1 + 1; index2 != n; ++index2) {
            auto mask = (1ULL << index1) + (1ULL << index2);
            dp[mask] = (scalar_sizes[mask] <= W);
          }
        }
        continue;
      } else if (k == 3) {
        // Precompute the 3nd layer to have it for later layers.
        fsc.precompute_dimension(dp, 3);

        // And iterate all 3-bit masks.
        for (unsigned index1 = 0; index1 != n - 2; ++index1) {
          for (unsigned index2 = index1 + 1; index2 != n - 1; ++index2) {
            for (unsigned index3 = index2 + 1; index3 != n; ++index3) {
              auto mask = (1ULL << index1) + (1ULL << index2) + (1ULL << index3);
              auto mask1 = (1ULL << index1) + (1ULL << index2);
              auto mask2 = (1ULL << index2) + (1ULL << index3);
              auto mask3 = (1ULL << index1) + (1ULL << index3);
              dp[mask] = (scalar_sizes[mask] <= W) && ((dp[mask1] && dp[mask ^ mask1]) || (dp[mask2] && dp[mask ^ mask2]) || (dp[mask3] && dp[mask ^ mask3]));
            }
          }
        }
        continue;
      } else if (k == 4) {
        // Precompute the 4nd layer to have it for later layers.
        fsc.precompute_dimension(dp, 4);

        // And iterate all 4-bit masks.
        for (unsigned index1 = 0; index1 != n - 3; ++index1) {
          for (unsigned index2 = index1 + 1; index2 != n - 2; ++index2) {
            for (unsigned index3 = index2 + 1; index3 != n - 1; ++index3) {
              for (unsigned index4 = index3 + 1; index4 != n; ++index4) {
                auto mask = (1ULL << index1) + (1ULL << index2) + (1ULL << index3) + (1ULL << index4);
                auto mask1 = (1ULL << index1) + (1ULL << index2);
                auto mask2 = (1ULL << index1) + (1ULL << index3);
                auto mask3 = (1ULL << index1) + (1ULL << index4);
                dp[mask] = (scalar_sizes[mask] <= W) && (
                  (dp[mask1] && dp[mask ^ mask1]) ||
                  (dp[mask2] && dp[mask ^ mask2]) ||
                  (dp[mask3] && dp[mask ^ mask3]) ||
                  (dp[1ULL << index1] && dp[mask ^ (1ULL << index1)]) ||
                  (dp[1ULL << index2] && dp[mask ^ (1ULL << index2)]) ||
                  (dp[1ULL << index3] && dp[mask ^ (1ULL << index3)]) ||
                  (dp[1ULL << index4] && dp[mask ^ (1ULL << index4)])
                );
              }
            }
          }
        }
        continue;
      } else if (k == 5) {
        // Precompute the 5nd layer to have it for later layers.
        fsc.precompute_dimension(dp, 5);

        // And iterate all 5-bit masks.
        for (unsigned index1 = 0; index1 != n - 4; ++index1) {
          for (unsigned index2 = index1 + 1; index2 != n - 3; ++index2) {
            for (unsigned index3 = index2 + 1; index3 != n - 2; ++index3) {
              for (unsigned index4 = index3 + 1; index4 != n - 1; ++index4) {
                for (unsigned index5 = index4 + 1; index5 != n; ++index5) {
                  auto mask = (1ULL << index1) + (1ULL << index2) + (1ULL << index3) + (1ULL << index4) + (1ULL << index5);

                  auto mask1 = (1ULL << index1) + (1ULL << index2);
                  auto mask2 = (1ULL << index1) + (1ULL << index3);
                  auto mask3 = (1ULL << index1) + (1ULL << index4);
                  auto mask4 = (1ULL << index1) + (1ULL << index5);
                  auto mask5 = (1ULL << index2) + (1ULL << index3);
                  auto mask6 = (1ULL << index2) + (1ULL << index4);
                  auto mask7 = (1ULL << index2) + (1ULL << index5);
                  auto mask8 = (1ULL << index3) + (1ULL << index4);
                  auto mask9 = (1ULL << index3) + (1ULL << index5);
                  auto mask10 = (1ULL << index4) + (1ULL << index5);

                  dp[mask] = (scalar_sizes[mask] <= W) && (
                    (dp[mask1] && dp[mask ^ mask1]) ||
                    (dp[mask2] && dp[mask ^ mask2]) ||
                    (dp[mask3] && dp[mask ^ mask3]) ||
                    (dp[mask4] && dp[mask ^ mask4]) ||
                    (dp[mask5] && dp[mask ^ mask5]) ||
                    (dp[mask6] && dp[mask ^ mask6]) ||
                    (dp[mask7] && dp[mask ^ mask7]) ||
                    (dp[mask8] && dp[mask ^ mask8]) ||
                    (dp[mask9] && dp[mask ^ mask9]) ||
                    (dp[mask10] && dp[mask ^ mask10]) ||
                    (dp[1ULL << index1] && dp[mask ^ (1ULL << index1)]) ||
                    (dp[1ULL << index2] && dp[mask ^ (1ULL << index2)]) ||
                    (dp[1ULL << index3] && dp[mask ^ (1ULL << index3)]) ||
                    (dp[1ULL << index4] && dp[mask ^ (1ULL << index4)]) ||
                    (dp[1ULL << index5] && dp[mask ^ (1ULL << index5)])
                  );
                }
              }
            }
          }
        }
        continue;
      }

      if (k == 6) {
        fsc.fill_positions(dp, 5);
      }

      // Do the actual convolution.
      fsc.bounded_convolution(k);

      // Iterate the subsets of size `k`.
      for (int64_t mask = (1ull << k) - 1, r, c; mask < (1ull << n); c = mask & -mask, r = mask + c, mask = r | (((r ^ mask) >> 2) / c)) {
        auto val = (fsc.tmp[mask] > 0) && (scalar_sizes[mask] <= W);
        dp[mask] = val;

        // Note: We already fill the `store` array for the next iteration.
        if (k < n) {
          fsc.get(k, mask) = val;
        }
      }
    }

    // Note: `dp` does not store the optimal value, but rather *whether* `W` is a feasible solution.
    if (dp[N - 1] > 0) {
      if (forceJoinTreeBuild) {
        auto solTree = extractJoinTreeSpecialDP(dp, scalar_sizes);
        return {dp[N - 1] > 0, solTree};
      }
      return {dp[N - 1] > 0, nullptr};
    }
    return {dp[N - 1] > 0, nullptr};
  };

  auto [isSol, joinTree] = is_solution(pos_of_full_card, true && shouldExtractJoinTree);
  if (isSol) {
    return {scalar_sizes[perm[pos_of_full_card]], joinTree};
  }

  int pos = start_pos;
  for (int step = N; step; step >>= 1) {
    // Note: We've already tested `pos_of_full_card` above.
    if (pos + step < pos_of_full_card) {
      auto [isSol, _] = is_solution(pos + step);
      if (isSol) {
        pos += step;
      }
    }
  }

  auto [finalSol, finalJoinTree] = is_solution(pos, true && shouldExtractJoinTree);
  assert(finalSol);
  return {scalar_sizes[perm[pos]], finalJoinTree};
}
