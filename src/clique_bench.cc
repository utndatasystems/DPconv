#include <iostream>
#include <fstream>
#include <cassert>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <memory>
#include <thread>
#include <filesystem>
#include <atomic>
#include <cstdlib>
#include <ranges>
#include <random>

#include "BenchmarkRunner.hpp"
#include "Generator.hpp"
#include "QueryGraph.hpp"

namespace fs = std::filesystem;

std::string run_clique(unsigned n, unsigned W, unsigned seed_val) {
  auto sizes = gen_random_sizes<uint64_t>(n, W, 1, W, seed_val);

  std::vector<std::pair<unsigned, unsigned>> joins;
  for (unsigned i = 0; i != n; ++i) {
    for (unsigned j = i + 1; j != n; ++j) {
      joins.push_back({i, j});
    }
  }

  QueryGraph qg(n, joins);

  qg.dump(sizes, W, seed_val);

  auto [meta_info, ret] = optimize_query("clique-" + std::to_string(n), n, qg, sizes, {}, false);
  return ret;
}

void benchmark_cliques(unsigned limit, unsigned W, unsigned seed) {
  std::string nice_output;
  if (W < 1000) {
    nice_output = std::to_string(W);
  } else if (W < 1000000) {
    nice_output = std::to_string(W / 1000) + "K";
  } else if (W < 1000000000) {
    nice_output = std::to_string(W / 1000000) + "M";
  } else {
    nice_output = std::to_string(W / 1000000000) + "B";
  }

  std::vector<std::string> collector;
  int seeds[5] = {1, 10, 100, 1000, 10000};
  for (unsigned i = 0; i != 5; ++i) {
    for (unsigned n = 3; n <= limit; ++n) {
      collector.push_back(run_clique(n, W, seeds[i]));
    }
    flush("cliques-" + nice_output, collector);
  }
  flush("cliques-" + nice_output, collector, true);
}

int main(int argc, char** argv) {
  if ((argc != 3) && (argc != 4)) {
    std::cerr << "Usage: " << argv[0] << " <n:int> <W:int> [<seed:int>]" << std::endl;
    exit(-1);
  }

  auto n = std::atoi(argv[1]);
  auto W = std::atoi(argv[2]);
  assert(W <= 1e9);

  unsigned seed = 0;
  if (argc == 4) seed = std::stoi(argv[2]);
  benchmark_cliques(n, W, seed);
  return 0;
}