#ifndef H_QueryGraph
#define H_QueryGraph

#include <fstream>
#include <queue>
#include <optional>
#include <mutex>
#include <thread>

#include "BitSet.hpp"
#include "Common.hpp"

class QueryGraph {  
public:
  QueryGraph() = default;

  QueryGraph(unsigned n, std::vector<std::pair<unsigned, unsigned>>& joins)
  : n_(n), joins_(joins) {}

  // Compute the neighbors for a given set, forbidding those from `x`.
  BitSet64 compute_neighbors(BitSet64 s, BitSet64 x) const {
    BitSet64 result;
    for (unsigned index = 0, limit = joins_.size(); index != limit; ++index) {
      auto [u, v] = joins_[index];
      auto l = BitSet64({u}), r = BitSet64({v});
        
      if (l.isSubsetOf(s) && (!r.doesIntersectWith(s + x)) && (!r.doesIntersectWith(result)))
        result.insert(*r.begin());
      if (r.isSubsetOf(s) && (!l.doesIntersectWith(s + x)) && (!l.doesIntersectWith(result)))
        result.insert(*l.begin());
    }
    return result;
  }

  std::size_t size() const {
    return n_;
  }

  void dump(const std::vector<uint64_t>& sizes, uint64_t W, std::size_t seed_val) {
    // For debug purposes.
#if 0
    system("mkdir -p ../used-cliques");
    std::ofstream out("../used-cliques/clique-" + std::to_string(n_) + "-" + std::to_string(W) + ".log");
    assert((1u << n_) == sizes.size());
    for (unsigned index = 0, limit = sizes.size(); index != limit; ++index)
      out << sizes[index] << std::endl;
    out.close();
#endif
  }

private:
  unsigned n_;
  std::vector<std::pair<unsigned, unsigned>> joins_;
};

#endif