// Contains the builders for the join trees.

#ifndef H_Util
#define H_Util

#include "Common.hpp"

#include <filesystem>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <cstring>
#include <chrono>
#include <vector>
#include <limits>
#include <ctime>

namespace fs = std::filesystem;
using namespace std::chrono;

// Utils for building a join tree.

struct MetaInfo {
public:
  // Cout
  uint64_t cout_opt;
  uint64_t cout_when_cmax;

  // Cmax
  uint64_t cmax_opt;
  uint64_t cmax_when_cout;

  // Cout of Ccap.
  uint64_t bounded_cout_opt;

  static std::string schema_debug() {
    return "cout_opt,cout_when_cmax,cmax_opt,cmax_when_cout,bounded_cout_opt";
  }

  std::string data_debug() {
    return std::to_string(cout_opt) + ","
         + std::to_string(cout_when_cmax) + ","
         + std::to_string(cmax_opt) + ","
         + std::to_string(cmax_when_cout) + ","
         + std::to_string(bounded_cout_opt);
  }
};

// TODO: use unique_ptrs.
class JoinNode {
public:
  JoinNode(uint64_t set, uint64_t size, std::shared_ptr<JoinNode> left, std::shared_ptr<JoinNode> right)
  : set(set), size(size), left(left), right(right) {}
  
  bool operator == (const JoinNode&& other) const {
    return set == other.set;
  }

  // // Deconstructors to debug dangling pointers.
  // void operator()(JoinNode *pNode) { 
  //   if (!pNode)
  //     throw std::runtime_error("Dangling pointer. You're leaking stuff somewhere");
  //   delete pNode;
  // }

  uint64_t set;
  std::shared_ptr<JoinNode> left, right;
  uint64_t size;

  void debug(std::string filename, std::vector<std::string>& tns, const std::vector<uint64_t>& sizes, std::string extra) {
    // Take the actual filename from `filename` and put `extra` as the file extension.
    fs::path filePath(filename);
    std::string fileNameOnly = filePath.stem().string(); // Get only the filename
    fileNameOnly += "_" + extra;
    auto fileNameFull = "../job_join_trees/" + fileNameOnly;
    std::cout << "Debug filename: " << fileNameFull << std::endl;

    std::cerr << extra << std::endl;

    std::ofstream out(fileNameFull);
    auto [vec_str, _] = this->rec_debug(tns, sizes);
    for (auto elem : vec_str)
      out << elem << std::endl;
    out.close();
  }
private:
  bool is_single() const {
    assert(set);
    return (set & (set - 1)) == 0;
  }

  unsigned extract_bit() const {
    return static_cast<unsigned>(std::log2(set));
  }

  std::pair<std::vector<std::string>, std::string> rec_debug(std::vector<std::string>& tns, const std::vector<uint64_t>& sizes) {
    if (this->is_single()) {
      std::vector<std::string> tmp;
      auto ret = "(" + tns[this->extract_bit()] + ")";
      return {tmp, ret};
    }
    auto [l_vec, l] = left->rec_debug(tns, sizes);
    auto [r_vec, r] = right->rec_debug(tns, sizes);
    l_vec.insert(l_vec.end(), r_vec.begin(), r_vec.end());
#if 0
    if (left->size >= right->size) {
      std::cerr << "join: size=" << size << " "  << l << " [" << std::to_string(left->size) << "]" << " " << r << " [" << std::to_string(right->size) << "]" << std::endl;
      return "(" + l + " [" + std::to_string(left->size) + "]" + " " + r + " [" + std::to_string(right->size) + "])";
    } else {
      std::cerr << "join: size=" << size << " " << r << " [" << std::to_string(right->size) << "]" << " " << l << " [" << std::to_string(left->size) << "]" << std::endl;
      return "(" + r + " [" + std::to_string(right->size) + "]" + " " + l + " [" + std::to_string(left->size) + "])";
    }
#else
    std::string total_ret;
    if (left->size >= right->size) {
      std::cerr << "(" << l << " | " << r << ")" << std::endl;
      total_ret = "(" + l + " | " + r + ")";
    } else {
      std::cerr << "(" << r << " | " << l << ")" << std::endl;
      total_ret = "(" + r + " | " + l + ")";
    }
    l_vec.push_back(total_ret);
    return {l_vec, total_ret};
#endif
  }
};

template <typename U>
static std::shared_ptr<JoinNode> extractJoinTreeSpecialDP(std::vector<U>& dp, const std::vector<uint64_t>& sizes) {
  // Find the optimal split of set `S`.
  auto findOptSplit = [&](uint64_t S) {
    // TODO: Improvement: avoid symmetric cases.
    // TODO: Implement this: https://fishi.devtail.io/weblog/2015/05/18/common-bitwise-techniques-subset-iterations/
    // TODO: find the number of bits => K = \upper{popcount(S)} -> but for even popcount, it will iterate some repetitive.
    // TODO: but you anyway return. But until then maybe it finds other pairs.
    uint64_t T = 0;
    do {
      if ((T) && (T != S)) {
        if (dp[T] && dp[S ^ T]) {
          return T;
        }
      }
      T = (T - S) & S;
    } while (T);
    assert(0);
    return S;
  };

  // Recursive build the solution tree.
  std::function<std::shared_ptr<JoinNode>(uint64_t)> recBuild = [&](uint64_t S) {
    if ((S & (S - 1)) == 0)
      return std::make_shared<JoinNode>(S, sizes[S], nullptr, nullptr);

    auto T = findOptSplit(S);
    return std::make_shared<JoinNode>(S, sizes[S], recBuild(T), recBuild(S ^ T));
  };

  // Build the join tree.
  return recBuild(sizes.size() - 1);
}

template <typename U>
static std::shared_ptr<JoinNode> extractJoinTreeMinMax(std::vector<U>& dp, const std::vector<uint64_t>& sizes) {
  // Find the optimal split of set `S`.
  auto findOptSplit = [&](uint64_t S) {
    // TODO: Improvement: avoid symmetric cases.
    // TODO: Implement this: https://fishi.devtail.io/weblog/2015/05/18/common-bitwise-techniques-subset-iterations/
    // TODO: find the number of bits => K = \upper{popcount(S)} -> but for even popcount, it will iterate some repetitive.
    // TODO: but you anyway return. But until then maybe it finds other pairs.
    uint64_t T = 0;
    do {
      if ((T) && (T != S)) {
        if (dp[S] == std::max(sizes[S], static_cast<uint64_t>(std::max(dp[T], dp[S ^ T])))) {
          return T;
        }
      }
      T = (T - S) & S;
    } while (T);
    assert(0);
    return S;
  };

  // Recursive build the solution tree.
  std::function<std::shared_ptr<JoinNode>(uint64_t)> recBuild = [&](uint64_t S) {
    if ((S & (S - 1)) == 0)
      return std::make_shared<JoinNode>(S, sizes[S], nullptr, nullptr);

    auto T = findOptSplit(S);
    return std::make_shared<JoinNode>(S, sizes[S], recBuild(T), recBuild(S ^ T));
  };

  // Build the join tree.
  return recBuild(sizes.size() - 1);
}

template <typename U>
static std::shared_ptr<JoinNode> extractJoinTreeHash(std::vector<U>& dp, const std::vector<uint64_t>& sizes) {
  // Find the optimal split of set `S`.
  auto findOptSplit = [&](uint64_t S) {
    // TODO: Improvement: avoid symmetric cases.
    // TODO: Implement this: https://fishi.devtail.io/weblog/2015/05/18/common-bitwise-techniques-subset-iterations/
    // TODO: find the number of bits => K = \upper{popcount(S)} -> but for even popcount, it will iterate some repetitive.
    // TODO: but you anyway return. But until then maybe it finds other pairs.
    uint64_t T = 0;
    do {
      if ((T) && (T != S)) {
        if ((dp[T] != std::numeric_limits<U>::max()) &&
            (dp[S ^ T] != std::numeric_limits<U>::max()) &&
            (dp[S] == std::max(dp[T],
              std::max(dp[S ^ T], std::min(sizes[T], sizes[S ^ T]))))) {
          return T;
        }
      }
      T = (T - S) & S;
    } while (T);
    assert(0);
    return S;
  };

  // Recursive build the solution tree.
  std::function<std::shared_ptr<JoinNode>(uint64_t)> recBuild = [&](uint64_t S) {
    if ((S & (S - 1)) == 0)
      return std::make_shared<JoinNode>(S, sizes[S], nullptr, nullptr);

    auto T = findOptSplit(S);
    return std::make_shared<JoinNode>(S, sizes[S], recBuild(T), recBuild(S ^ T));
  };

  // for (unsigned index = 0, limit = sizes.size(); index != limit; ++index) {
  //   std::cerr << "index=" << index << " val=" << sizes[index] << " dp: " << dp[index] << std::endl;
  // }

  // Build the join tree.
  return recBuild(sizes.size() - 1);
}

template <typename U>
static std::shared_ptr<JoinNode> extractJoinTreeMinPlus(std::vector<U>& dp, const std::vector<uint64_t>& sizes) {
  // Find the optimal split of set `S`.
  auto findOptSplit = [&](uint64_t S) {
    // TODO: Improvement: avoid symmetric cases.
    // TODO: Implement this: https://fishi.devtail.io/weblog/2015/05/18/common-bitwise-techniques-subset-iterations/
    // TODO: find the number of bits => K = \upper{popcount(S)} -> but for even popcount, it will iterate some repetitive.
    // TODO: but you anyway return. But until then maybe it finds other pairs.
    uint64_t T = 0;
    do {
      if ((T) && (T != S)) {
        if ((dp[T] != std::numeric_limits<U>::max()) &&
            (dp[S ^ T] != std::numeric_limits<U>::max()) &&
            (dp[S] == sizes[S] + dp[T] + dp[S ^ T])) {
          return T;
        }
      }
      T = (T - S) & S;
    } while (T);
    assert(0);
    return S;
  };

  // Recursive build the solution tree.
  std::function<std::shared_ptr<JoinNode>(uint64_t)> recBuild = [&](uint64_t S) {
    if ((S & (S - 1)) == 0)
      return std::make_shared<JoinNode>(S, sizes[S], nullptr, nullptr);

    auto T = findOptSplit(S);
    return std::make_shared<JoinNode>(S, sizes[S], recBuild(T), recBuild(S ^ T));
  };

  // for (unsigned index = 0, limit = sizes.size(); index != limit; ++index) {
  //   std::cerr << "index=" << index << " val=" << sizes[index] << " dp: " << dp[index] << std::endl;
  // }

  // Build the join tree.
  return recBuild(sizes.size() - 1);
}

static uint64_t computeJoinTreeMinMaxCost(std::shared_ptr<JoinNode> joinNode, const std::vector<uint64_t>& sizes) {
  std::function<uint64_t(std::shared_ptr<JoinNode>, unsigned)> recCompute = [&](std::shared_ptr<JoinNode> curr, unsigned depth) -> uint64_t {
    auto S = curr->set;
    if ((S & (S - 1)) == 0) {
      return 0;
    }

    auto l = recCompute(curr->left, depth + 1), r = recCompute(curr->right, depth + 1);
    return std::max(sizes[S], std::max(l, r));
  };

  return recCompute(joinNode, 0);
}

static uint64_t computeJoinTreeHashCost(std::shared_ptr<JoinNode> joinNode, const std::vector<uint64_t>& sizes) {
  std::function<uint64_t(std::shared_ptr<JoinNode>, unsigned)> recCompute = [&](std::shared_ptr<JoinNode> curr, unsigned depth) -> uint64_t {
    auto S = curr->set;
    if ((S & (S - 1)) == 0) {
      return 0;
    }

    auto l = recCompute(curr->left, depth + 1), r = recCompute(curr->right, depth + 1);
    return std::max(l,
      std::max(r, std::min(sizes[curr->left->set], sizes[curr->right->set])));
  };

  return recCompute(joinNode, 0);
}

static uint64_t computeJoinTreeMinPlusCost(std::shared_ptr<JoinNode> joinNode, const std::vector<uint64_t>& sizes) {
  std::function<uint64_t(std::shared_ptr<JoinNode>, unsigned)> recCompute = [&](std::shared_ptr<JoinNode> curr, unsigned depth) -> uint64_t {
    auto S = curr->set;
    if ((S & (S - 1)) == 0) {
      return 0;
    }

    auto l = recCompute(curr->left, depth + 1), r = recCompute(curr->right, depth + 1);
    return sizes[S] + l + r;
  };

  return recCompute(joinNode, 0);
}

struct Sequence {
  int i, j;
};

static std::string mask_to_str(uint64_t x, unsigned enforce = 0) {
  auto pos = (!enforce) ? static_cast<unsigned>(std::log2(x)) : (enforce - 1);
  std::string ret;
  for (unsigned i = pos; i <= pos; --i) {
    ret += std::to_string((x >> i) & 1);
  }
  return ret;
}

static std::string indent(unsigned d) {
  std::string ret = "[";
  while (d--)
    ret += "*";
  ret += "]";
  return ret;
}

static bool isEqual(const char* str1, const char* str2) {
  return !strcmp(str1, str2);
}

static void debugSequence(unsigned n, Sequence* seq) {
  for (unsigned index = 0, limit = n - 1; index != limit; ++index) {
    std::cerr << "index=" << index << " -> (" << seq[index].i << ", " << seq[index].j << ")" << std::endl;
  }
}

static std::vector<std::string> parseDirectory(std::string dir) {
  std::vector<std::string> ret;
  for (const auto &entry : fs::directory_iterator(dir))
    ret.push_back(entry.path());
  return ret;
}

static std::pair<unsigned, unsigned> getInfo(std::string filepath) {
  auto filename = fs::path(filepath).filename().string().substr(4);

  auto split = [&](std::string str, char delim) {
    std::vector<std::string> ret = {};
    std::string curr = "";
    for (char c : str) {
      if (c == delim) {
        ret.push_back(curr);
        curr.clear();
      } else {
        curr += c;
      }
    }
    ret.push_back(curr);
    return ret;
  };

  auto tmp = split(filename, '-');
  auto size = std::stoi(tmp[0]);
  auto index = std::stoi(tmp[1]);
  return {size, index};
}

class Timer {
public:
  Timer(std::string approach, unsigned size = 0)
  : approach_(approach),
    size_(size),
    duration_(0),
    isStopped(false),
    timeout(std::numeric_limits<double>::max()) {
      start();
    }

  void setTimeout(double timeout) {
    // Set timeout in seconds.
    this->timeout = timeout;
  }

  void start() {
    start_ = ::high_resolution_clock::now();
    duration_ = 0;
  }

  void stop() {
    isStopped = true;
    stop_ = ::high_resolution_clock::now();
    duration_ = duration_cast<microseconds>(stop_ - start_).count();
    ++counter_;
  }

  void debug() {
    if (!isStopped) stop();
    std::cerr << "Approach: " << approach_ << " took " << duration_ / 1e3 << " ms" << std::endl;
  }

  bool isTimeout() const {
    auto tmp = ::high_resolution_clock::now();
    return (duration_cast<microseconds>(tmp - start_).count() > timeout * 1e6);
  }

  void merge(const Timer& o) {
    reports.push_back({o.size_, 1.0 * o.duration_ / o.counter_ / 1000});
  }

  void summary() {}

  void flush() {
    const std::time_t now = std::time(nullptr);
    const std::tm ct = *std::localtime( std::addressof(now) ) ;
    std::string time = std::to_string(ct.tm_hour) + "-" + std::to_string(ct.tm_min) + "-" + std::to_string(ct.tm_sec);

    auto filename = "../results/" + approach_ + "_" + time + ".out";
    std::ofstream out(filename);
    assert(out.is_open());
    for (const auto& [s, t] : reports) {
      out << s << ": " << t << std::endl;
    }
  }

private:
  double timeout;
  high_resolution_clock::time_point start_, stop_;
  std::string approach_;
  unsigned size_;
  double duration_ = 0.0;
  unsigned counter_ = 0;
  bool isStopped = false;
  std::vector<std::tuple<unsigned, double>> reports;
};

#endif