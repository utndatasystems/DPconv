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
#include "QueryGraph.hpp"

enum class OptimizationType : unsigned {
  MinMax,
  MinPlus
};

template <class U>
class DPccpWrapper {
public:
  DPccpWrapper(OptimizationType type, QueryGraph& qg, const std::vector<U>& scalarSizes, U capped_size = std::numeric_limits<U>::max())
  : type_(type), capped_size_(capped_size), qg_(qg), scalarSizes_(scalarSizes), dp(scalarSizes.size(), std::numeric_limits<U>::max()) {}

  std::pair<U, std::shared_ptr<JoinNode>> runDPccp(bool shouldBuildJoinTree = true) {
    // Note: We use a DP-table, since the variant with generated plans is too slow, i.e., the result for a subset `S` is directly stored in `dp[S]`.
    // Init the base plans.
    init_base_relations();

    for (unsigned index = qg_.size() - 1, limit = qg_.size(); index < limit; --index) {
      // Set the start node.
      BitSet64 vi({index});

      // Prohibit everything we will handle in later loop iterations.
      BitSet64 Bi = index ? BitSet64::fill(index - 1) : BitSet64();

      // Emit the first connected component.
      dpCcpEmit(vi);

      // Grow recursively.
      dpCcpEnumerateRec(vi, Bi);
    }

    if (!shouldBuildJoinTree) {
      return {dp[(1u << qg_.size()) - 1], nullptr};
    }

    if (type_ == OptimizationType::MinMax) {
      auto solTree = extractJoinTreeMinMax(dp, scalarSizes_);
      return {dp[(1 << qg_.size()) - 1], solTree};
    } else if (type_ == OptimizationType::MinPlus) {
      auto solTree = extractJoinTreeMinPlus(dp, scalarSizes_);
      return {dp[(1 << qg_.size()) - 1], solTree};  
    } else {
      assert(0);
    }
  }

  void reset(OptimizationType type, U capped_size) {
    this->type_ = type;
    this->capped_size_ = capped_size;
    std::fill(dp.begin(), dp.end(), std::numeric_limits<U>::max());
  }

private:
  OptimizationType type_;
  QueryGraph& qg_;
  const std::vector<U>& scalarSizes_;
  BitSet64 leftPlan;
  U capped_size_;
  std::vector<uint64_t> dp;

  void dpCcpEmit2(BitSet64 rightProblem)
  // Emit a connected complement component.
  {
    // Check the DP table.
    if (dp[rightProblem.to_uint64_t()] == std::numeric_limits<U>::max())
      return;

    auto rightPlan = rightProblem;

    // Create the plan.
    create_plan(leftPlan, rightPlan);
  }

  void dpCcpEnumerateRec2(BitSet64 s, BitSet64 x)
  // Enumerate connected component recursively.
  {
    auto neighbors = qg_.compute_neighbors(s, x);
  
    // Emit all subsets as connected.
    for (auto subSet : neighbors.subsets())
      dpCcpEmit2(s + subSet);

    // Grow recursively.
    auto xr = x + neighbors;
    for (auto subSet : neighbors.subsets())
      dpCcpEnumerateRec2(s + subSet, xr);
  }

  void dpCcpEmit(BitSet64 s)
  // Emit a connected component.
  {
    // Check the DP table.
    if (dp[s.to_uint64_t()] == std::numeric_limits<U>::max())
      return;

    // Prune for capped Cout.
    if ((s.size() != 1) && (scalarSizes_[s.to_uint64_t()] > capped_size_)) {
      return;
    }

    leftPlan = s;

    // Grow in neighbors.
    auto Bmin = BitSet64::fill(*s.begin()) + s;
    auto neighbors = qg_.compute_neighbors(s, Bmin);
    for (auto r : neighbors.reverseorder()) {
      BitSet64 vi({r});
      BitSet64 xi = Bmin;
      for (auto r2 : neighbors) {
        if (r2 >= r)
          break;
        xi.insert(r2);
      }

      dpCcpEmit2(vi);
      dpCcpEnumerateRec2(vi, xi);
    }
  }

  void dpCcpEnumerateRec(BitSet64 s, BitSet64 x) {
    auto neighbors = qg_.compute_neighbors(s, x);

    for (auto subset : neighbors.subsets())
      dpCcpEmit(s + subset);

    // Grow recursively.
    auto xr = x + neighbors;
    for (auto subset : neighbors.subsets())
      dpCcpEnumerateRec(s + subset, xr);
  }

  void create_plan(BitSet64 l, BitSet64 r) {
    // Fetch the problems.
    auto rightProblem = l;
    auto leftProblem = r;

    assert((leftProblem & rightProblem).empty());
    auto totalProblem = leftProblem + rightProblem;
    
    // Compute the join cost.
    auto join_size = scalarSizes_[totalProblem.to_uint64_t()];

    // Prune for capped Cout.
    if (join_size > capped_size_) {
      return;
    }

    auto l_cost = dp[l.to_uint64_t()];
    auto r_cost = dp[r.to_uint64_t()];

    uint64_t currCost = 0;
    if (type_ == OptimizationType::MinPlus) {
      currCost = join_size + l_cost + r_cost;
    } else if (type_ == OptimizationType::MinMax) {
      currCost = std::max(
        join_size,
        std::max(
          l_cost,
          r_cost
        )
      );
    } else {
      assert(0);
    }

    dp[totalProblem.to_uint64_t()] = std::min(
      dp[totalProblem.to_uint64_t()],
      currCost
    );
  }

  void init_base_relations() {
    for (unsigned index = 0, limit = qg_.size(); index != limit; ++index) {
      dp[1u << index] = 0;
    }
  }
};