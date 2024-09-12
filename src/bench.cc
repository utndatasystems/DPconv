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

namespace fs = std::filesystem;

std::tuple<MetaInfo, std::string> run_query_from_file(std::string filename, bool analyzeRatios = false) {
  assert(endswith(filename, ".csv"));
  std::cerr << "[start] " << filename << std::endl;
  std::ifstream in(filename);
  assert(in.is_open());
  unsigned n, m, s;
  in >> n >> m >> s;
  std::cerr << "n=" << n << " m=" << m << " s=" << s << std::endl;
  std::vector<std::string> tns(n);
  std::vector<std::pair<unsigned, unsigned>> edges(m);
  for (unsigned index = 0; index != n; ++index) {
    in >> tns[index];
  }
  for (unsigned index = 0; index != m; ++index) {
    in >> edges[index].first >> edges[index].second;
  }
  std::vector<uint64_t> sizes(1u << n, std::numeric_limits<uint64_t>::max());

  uint64_t maxCard = 0;
  while (s--) {
    uint64_t bitset;
    double card;
    in >> bitset >> card;
    sizes[bitset] = static_cast<uint64_t>(card);
  }

  QueryGraph qg(n, edges);

  return optimize_query(filename, n, qg, sizes, tns, analyzeRatios);
}

std::vector<std::tuple<std::string, MetaInfo>> benchmark_directory_or_file(std::string dir_or_file, unsigned limit, bool analyzeRatios = false) {
  // Collect the files to be benchmarked.
  std::vector<std::string> fns;
  if (fs::is_directory(dir_or_file)) {
    for (const auto& entry : fs::directory_iterator(dir_or_file)) {
      if (entry.is_regular_file()) {
        fns.push_back(entry.path());
      }
    }
  } else {
    fns.push_back(dir_or_file);
  }

  std::atomic<unsigned> taskIndex = 0;
  auto numTasks = std::min(limit, static_cast<unsigned>(fns.size()));

  std::vector<std::string> collector(numTasks);
  std::vector<std::tuple<std::string, MetaInfo>> ratios;
  for (unsigned index = 0; index != numTasks; ++index) {
    auto [meta_info, actual_ret] = run_query_from_file(fns[index], analyzeRatios);
    ratios.push_back({fns[index], meta_info});
    collector[index] = actual_ret;
  }

  flush("dpconv", collector);
  return ratios;
}

int main(int argc, char** argv) {
  if ((argc != 2) && (argc != 3)) {
    std::cerr << "Usage: " << argv[0] << " <input:{directory, file}> [<limit:int>]" << std::endl;
    exit(-1);
  }

  auto dir_or_file = std::string(argv[1]);
  std::cerr << "Benchmark " << dir_or_file << std::endl;
  unsigned limit = std::numeric_limits<unsigned>::max();
  if (argc == 3) limit = std::stoi(argv[2]);

  auto ratios = benchmark_directory_or_file(dir_or_file, limit, true);

  std::ofstream out("../cout_cmax_ratios.csv");
  assert(out.is_open());
  out << "query" << ","
      << "n" << ","
      << MetaInfo::schema_debug()
      << std::endl;
  for (auto [fn, meta_info] : ratios) {
    std::ifstream in(fn);
    assert(in.is_open());
    unsigned n; in >> n;
    in.close();

    out << fn << "," << n << "," << meta_info.data_debug() << "\n";
  }
  out.close();

  return 0;
}