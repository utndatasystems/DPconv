#include <iostream>
#include <fstream>
#include <cassert>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <memory>
#include <cmath>
#include <array>

#include "Util.hpp"

// Optimize wrt C_{out} with DPsub, capping the max. intermediate size.
template <typename U>
std::pair<U, std::shared_ptr<JoinNode>> runMinMaxDPsub_hybrid(const std::vector<U>& scalarSizes, const uint64_t capped_size) {
  auto N = scalarSizes.size();
  assert((N & (N - 1)) == 0);
  std::vector<U> dp(N, std::numeric_limits<U>::max());
  for (uint64_t S = 0, limit = N; S != limit; ++S) {
    if ((S & (S - 1)) == 0) {
      dp[S] = 0;
      continue;
    }

    // Prune unconnected subgraphs.
    if (scalarSizes[S] == std::numeric_limits<U>::max() || scalarSizes[S] > capped_size)
      continue;

    uint64_t current = 0;
    do {
      if ((current != 0) && (current != S) &&
        (dp[current] != std::numeric_limits<U>::max()) &&
        (dp[S ^ current] != std::numeric_limits<U>::max()))
          dp[S] = std::min(dp[S], dp[current] + dp[S ^ current]);
      current = (current - S) & S;
    } while (current);

    // NOTE: We changed this.
    if (dp[S] != std::numeric_limits<U>::max())
      dp[S] += scalarSizes[S];
  }

  // If we also need to time for building the join tree (default case for the benchmarks).
  // Note: In this case, we could also store an `OPT`-array. However, this also takes space.
  auto solTree = extractJoinTreeMinPlus(dp, scalarSizes);
  return {dp[N - 1], solTree};
}

// Optimize wrt C_{max} with DPsub.
template <typename U>
std::pair<U, std::shared_ptr<JoinNode>> runMinMaxDPsub(const std::vector<U>& scalarSizes, bool shouldExtractJoinTree = true) {
  auto N = scalarSizes.size();
  assert((N & (N - 1)) == 0);
  std::vector<U> dp(N, std::numeric_limits<U>::max());
  for (uint64_t S = 0, limit = N; S != limit; ++S) {
    if ((S & (S - 1)) == 0) {
      dp[S] = 0;
      continue;
    }

    // Prune unconnected subgraphs.
    if (scalarSizes[S] == std::numeric_limits<U>::max())
      continue;

    uint64_t current = 0;
    do {
      if ((current != 0) && (current != S) &&
        (dp[current] != std::numeric_limits<U>::max()) &&
        (dp[S ^ current] != std::numeric_limits<U>::max()))
        dp[S] = std::min(dp[S], std::max(dp[current], dp[S ^ current]));
      current = (current - S) & S;
    } while (current);
    dp[S] = std::max(dp[S], scalarSizes[S]);
  }

  // If we also need to time for building the join tree (default case for the benchmarks).
  // Note: In this case, we could also store an `OPT`-array. However, this also takes space.
  if (!shouldExtractJoinTree) {
    return {dp[N - 1], nullptr};
  }

  auto solTree = extractJoinTreeMinMax(dp, scalarSizes);
  return {dp[N - 1], solTree};
}

// Optimize wrt C_{out} with DPsub.
template <typename U>
std::pair<U, std::shared_ptr<JoinNode>> runMinPlusDPsub(const std::vector<U>& scalarSizes) {
  auto N = scalarSizes.size();
  assert((N & (N - 1)) == 0);
  std::vector<U> dp(N, std::numeric_limits<U>::max());
  for (uint64_t S = 0, limit = N; S != limit; ++S) {
    if ((S & (S - 1)) == 0) {
      dp[S] = 0;
      continue;
    }

    // Prune unconnected subgraphs.
    if (scalarSizes[S] == std::numeric_limits<U>::max())
      continue;

    uint64_t current = 0;
    do {
      if ((current != 0) && (current != S) &&
        (dp[current] != std::numeric_limits<U>::max()) &&
        (dp[S ^ current] != std::numeric_limits<U>::max()))
          dp[S] = std::min(dp[S], dp[current] + dp[S ^ current]);
      current = (current - S) & S;
    } while (current);

    // NOTE: We changed this.
    if (dp[S] != std::numeric_limits<U>::max())
      dp[S] += scalarSizes[S];
  }

  // If we also need to time for building the join tree (default case for the benchmarks).
  // Note: In this case, we could also store an `OPT`-array. However, this also takes space.
  auto solTree = extractJoinTreeMinPlus(dp, scalarSizes);
  return {dp[N - 1], solTree};
}

// Optimize wrt C_{hash} with DPsub.
template <typename U>
std::pair<U, std::shared_ptr<JoinNode>> runHashDPsub(const std::vector<U>& scalarSizes) {
  auto N = scalarSizes.size();
  assert((N & (N - 1)) == 0);
  std::vector<U> dp(N, std::numeric_limits<U>::max());

  // for (unsigned index = 0; index != 10; ++index) {
  //   std::cerr << "index=" << index << " size=" << scalarSizes[index] << std::endl;
  // }

  for (uint64_t S = 0, limit = N; S != limit; ++S) {
    if ((S & (S - 1)) == 0) {
      dp[S] = 0;
      continue;
    }

    // Prune unconnected subgraphs.
    if (scalarSizes[S] == std::numeric_limits<U>::max())
      continue;

    uint64_t current = 0;
    do {
      if ((current != 0) && (current != S) &&
        (dp[current] != std::numeric_limits<U>::max()) &&
        (dp[S ^ current] != std::numeric_limits<U>::max())) {
          dp[S] = std::min(dp[S],
            std::max(
              std::max(dp[current], dp[S ^ current]),
              std::min(scalarSizes[current], scalarSizes[S ^ current])
            )
          );
        }
      current = (current - S) & S;
    } while (current);
    assert(dp[S] != std::numeric_limits<U>::max());
  }

  // If we also need to time for building the join tree (default case for the benchmarks).
  // Note: In this case, we could also store an `OPT`-array. However, this also takes space.
  auto solTree = extractJoinTreeHash(dp, scalarSizes);
  return {dp[N - 1], solTree};
}

// Approximative version: Optimize wrt C_{max} with DPsub, but only take the power of two (to have the 2-approximation).
// The running time is the same, but we can compare the results with our own approximation algorithms (which are faster).
template <typename U>
U runApproxMinMaxDPsub(const std::vector<U>& scalarSizes) {
  auto N = scalarSizes.size();
  assert((N & (N - 1)) == 0);
  std::vector<U> dp(N, std::numeric_limits<U>::max());

  for (uint64_t S = 0, limit = N; S != limit; ++S) {
    if ((S & (S - 1)) == 0) {
      dp[S] = 0;//sizes[S];
      continue;
    }

    // Prune unconnected subgraphs.
    if (scalarSizes[S] == std::numeric_limits<U>::max())
      continue;

    uint64_t current = 0;
    do {
      if ((current != 0) && (current != S)) {
        auto currVal = std::max(dp[current], dp[S ^ current]);
        if (currVal < dp[S]) {
          dp[S] = currVal;
        }
      }
      current = (current - S) & S;
    } while (current);
    U scaledDownVal = (!scalarSizes[S]) ? scalarSizes[S] : (1ULL << static_cast<unsigned>(std::log2(scalarSizes[S])));
    dp[S] = std::max(dp[S], scaledDownVal);
  }
  return dp[N - 1];
}