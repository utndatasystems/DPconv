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

#include "DPccp.cpp"
#include "DPsub.cpp"
#include "DPconv.cpp"

#include "Util.hpp"

#include "SubsetConvolution.hpp"

#define FULL_FLEDGED 1

// Adapt here for the corresponding benchmark.
#define CLIQUE_BENCHMARK 1
#define CAPPED_COUT_BENCHMARK 0
#define CLIQUE_CAPPED_COUT_BENCHMARK 0
#define CEB_BENCHMARK 0

#if CEB_BENCHMARK
  #define MINMAX_DPCCP 0
  #define MINMAX_DPSUB 1

  #define APPROX_MINMAX_DPSUB 0
  #define APPROX_MINMAX_DPCONV_NAIVE 0
  #define APPROX_MINMAX_DPCONV_BOOSTED 0
  #define MINMAX_DPCONV_NAIVE 0
  #define MINMAX_DPCONV_LAYERED 0
  #define MINMAX_DPCONV_BOOSTED 0
  #define MINMAX_DPCONV_INSTANT_NAIVE 0
  #define MINMAX_DPCONV_INSTANT_BOOSTED 0

  #define CAPPED_DPSUB_INSTANT_BOOSTED 0

  #define CAPPED_DPCCP_SIMPLE 0
  #define CAPPED_DPSUB_SIMPLE 1
  
  #define MINPLUS_DPCCP 0
  #define MINPLUS_DPSUB 1
#endif

#if CLIQUE_BENCHMARK
  #define MINMAX_DPCCP 0
  #define MINMAX_DPSUB 1

  #define APPROX_MINMAX_DPSUB 0
  #define APPROX_MINMAX_DPCONV_NAIVE 0
  #define APPROX_MINMAX_DPCONV_BOOSTED 0
  #define MINMAX_DPCONV_NAIVE 0
  #define MINMAX_DPCONV_LAYERED 0
  #define MINMAX_DPCONV_BOOSTED 0
  #define MINMAX_DPCONV_INSTANT_NAIVE 1
  #define MINMAX_DPCONV_INSTANT_BOOSTED 1

  #define CAPPED_DPSUB_INSTANT_BOOSTED 1

  #define CAPPED_DPCCP_SIMPLE 0
  #define CAPPED_DPSUB_SIMPLE 1
  
  #define MINPLUS_DPCCP 0
  #define MINPLUS_DPSUB 1
#endif

#if CAPPED_COUT_BENCHMARK
  #define MINMAX_DPCCP 1
  #define MINMAX_DPSUB 1
  
  #define APPROX_MINMAX_DPSUB 0
  #define APPROX_MINMAX_DPCONV_NAIVE 0
  #define APPROX_MINMAX_DPCONV_BOOSTED 0
  #define MINMAX_DPCONV_NAIVE 0
  #define MINMAX_DPCONV_LAYERED 0
  #define MINMAX_DPCONV_BOOSTED 0
  #define MINMAX_DPCONV_INSTANT_NAIVE 0
  #define MINMAX_DPCONV_INSTANT_BOOSTED 0

  #define CAPPED_DPSUB_INSTANT_BOOSTED 0

  #define CAPPED_DPCCP_SIMPLE 1
  #define CAPPED_DPSUB_SIMPLE 1
  
  #define MINPLUS_DPCCP 1
  #define MINPLUS_DPSUB 1
#endif

#if CLIQUE_CAPPED_COUT_BENCHMARK
  #define MINMAX_DPCCP 0
  #define MINMAX_DPSUB 1 // just to avoid some errors
  
  #define APPROX_MINMAX_DPSUB 0
  #define APPROX_MINMAX_DPCONV_NAIVE 0
  #define APPROX_MINMAX_DPCONV_BOOSTED 0
  #define MINMAX_DPCONV_NAIVE 0
  #define MINMAX_DPCONV_LAYERED 0
  #define MINMAX_DPCONV_BOOSTED 0
  #define MINMAX_DPCONV_INSTANT_NAIVE 1
  #define MINMAX_DPCONV_INSTANT_BOOSTED 1

  #define CAPPED_DPSUB_INSTANT_BOOSTED 1

  #define CAPPED_DPCCP_SIMPLE 0
  #define CAPPED_DPSUB_SIMPLE 1
  
  #define MINPLUS_DPCCP 0
  #define MINPLUS_DPSUB 1
#endif

// Source: https://stackoverflow.com/a/18681293/10348282
inline bool endswith(const std::string& s, const std::string& suffix) { return suffix.size() > s.size() ? false : (s.rfind(suffix) == s.size() - suffix.size()); }

std::tuple<MetaInfo, std::string> optimize_query(
  std::string fn, unsigned n, QueryGraph qg, const std::vector<uint64_t>& sizes, std::vector<std::string> tns, bool analyzeRatios = false) {
  auto maxCard = *std::ranges::max_element(sizes);
  auto logMaxCard = static_cast<unsigned>(std::log2(maxCard));
  std::cerr << "Max. cardinality: " << maxCard << " log=" << logMaxCard << std::endl;

  // ------------------------------------------------- //

  // `runMinMaxDPccp`
  auto time0 = std::chrono::high_resolution_clock::now();

#if MINMAX_DPCCP
  std::cerr << "[START] " << fn << " {MINMAX_DPCCP}" << std::endl;
  auto wrapper = DPccpWrapper(OptimizationType::MinMax, qg, sizes);
  auto [ret0, join_tree0] = wrapper.runDPccp();
#else
  auto ret0 = 0;
#endif
  std::cerr << "[STOP] " << fn << " ret0=" << ret0 << std::endl;
  auto time1 = std::chrono::high_resolution_clock::now();

  // !!! Set time !!!
  auto minmax_dpccp_time = std::chrono::duration_cast<std::chrono::microseconds>(time1 - time0).count();
  
  // Check for correctness.
#if MINMAX_DPCCP
  assert(computeJoinTreeMinMaxCost(join_tree0, sizes) == ret0);
#endif
  std::cerr << "MINMAX_DPSUB (time): " << std::chrono::duration_cast<std::chrono::microseconds>(time1 - time0).count() << std::endl;
  // ------------------------------------------------- //

  // Reset `time1`.
  time1 = std::chrono::high_resolution_clock::now();

  // ------------------------------------------------- //

  // `runMinMaxDPsub`
#if MINMAX_DPSUB
  std::cerr << "[START] " << fn << " {MINMAX_DPSUB}" << std::endl;
  auto [ret1, join_tree1] = runMinMaxDPsub(sizes);
#else
  auto ret1 = 0;
#endif
  std::cerr << "[STOP] " << fn << " ret1=" << ret1 << std::endl;
  auto time2 = std::chrono::high_resolution_clock::now();

  // !!! Set time !!!
  auto minmax_dpsub_time = std::chrono::duration_cast<std::chrono::microseconds>(time2 - time1).count();
  
  // Check for correctness.
#if MINMAX_DPSUB
  assert(computeJoinTreeMinMaxCost(join_tree1, sizes) == ret1);
#endif
  std::cerr << "MINMAX_DPSUB (time): " << std::chrono::duration_cast<std::chrono::microseconds>(time2 - time1).count() << std::endl;
  // ------------------------------------------------- //

  // Reset `time2`.
  time2 = std::chrono::high_resolution_clock::now();

  // ------------------------------------------------- //




#if 0
  // `runApproxMinMaxDPsub`
  std::cerr << "[START] " << fn << " {APPROX_MINMAX_DPSUB}" << std::endl;
#if APPROX_MINMAX_DPSUB
  auto ret2 = runApproxMinMaxDPsub(sizes);
#else
  auto ret2 = 0;
#endif
  std::cerr << "[STOP] " << fn << " ret2=" << ret2 << std::endl;
  auto time3 = std::chrono::high_resolution_clock::now();
  
  // `runApproxMinMaxDPconv_naive`
  std::cerr << "[START] " << fn << " {APPROX_MINMAX_DPCONV_NAIVE}" << std::endl;
#if APPROX_MINMAX_DPCONV_NAIVE
  #if FULL_FLEDGED
    auto ret3 = DPconv<64>::runApproxMinMaxDPconv_naive(logMaxCard, sizes);
  #else
    auto ret3 = DPconvImpl<64>::runApproxMinMaxDPconv_naive(sizes);
  #endif  
#else
  auto ret3 = 0;
#endif
  std::cerr << "[STOP] " << fn << " ret3=" << ret3 << std::endl;
  auto time4 = std::chrono::high_resolution_clock::now();

  // `runApproxMinMaxDPconv_boosted`
  std::cerr << "[START] " << fn << " {APPROX_MINMAX_DPCONV_BOOSTED}" << std::endl;
#if APPROX_MINMAX_DPCONV_BOOSTED
  #if FULL_FLEDGED
    auto ret4 = DPconv<64>::runApproxMinMaxDPconv_boosted(logMaxCard, sizes);
  #else
    auto ret4 = DPconvImpl<64>::runApproxMinMaxDPconv_boosted(sizes);
  #endif
#else
  auto ret4 = 0;
#endif
  std::cerr << "[STOP] " << fn << " ret4=" << ret4 << std::endl;
  auto time5 = std::chrono::high_resolution_clock::now();

  // `runMinMaxDPconv_naive`
  std::cerr << "[START] " << fn << " {MINMAX_DPCONV_NAIVE}" << std::endl;
#if MINMAX_DPCONV_NAIVE
  #if FULL_FLEDGED
    auto ret5 = DPconv<64>::runMinMaxDPconv_naive(logMaxCard, sizes);
  #else
    auto ret5 = DPconvImpl<64>::runMinMaxDPconv_naive(sizes);
  #endif
#else
  auto ret5 = 0;
#endif
  std::cerr << "[STOP] " << fn << " ret5=" << ret5 << std::endl;
  auto time6 = std::chrono::high_resolution_clock::now();

  // `runMinMaxDPconv_layered`
  std::cerr << "[START] " << fn << " {MINMAX_DPCONV_LAYERED}" << std::endl;
#if MINMAX_DPCONV_LAYERED
  #if FULL_FLEDGED
    auto pos_ret6 = DPconv<64>::runMinMaxDPconv_layered(n, compressedSizes);
  #else
    auto pos_ret6 = DPconvImpl<64>::runMinMaxDPconv_layered(compressedSizes);
  #endif

    // Find the actual size.
    assert((0 <= pos_ret6) && (pos_ret6 < sizes.size()));
    uint64_t ret6 = 0;
    for (unsigned index = 0; index != compressedSizes.size(); ++index) {
      if (compressedSizes[index] == pos_ret6)
        ret6 = sizes[index];
    }
#else
  auto ret6 = 0;
#endif

  std::cerr << "[STOP] " << fn << " ret6=" << ret6 << std::endl;
  auto time7 = std::chrono::high_resolution_clock::now();

  // `runMinMaxDPconv_boosted`
  // Analyze which semi-ring will have the support lower.
  std::cerr << "[START] " << fn << " {MINMAX_DPCONV_BOOSTED}" << std::endl;
#if MINMAX_DPCONV_BOOSTED
  uint64_t ret7 = 0;
  if (n < logMaxCard) {
#if FULL_FLEDGED
    auto pos_ret7 = DPconv<64>::runMinMaxDPconv_boosted(n, compressedSizes);
#else
    auto pos_ret7 = DPconvImpl<64>::runMinMaxDPconv_boosted(compressedSizes);
#endif

    // Find the actual size.
    assert((0 <= pos_ret7) && (pos_ret7 < sizes.size()));
    for (unsigned index = 0; index != compressedSizes.size(); ++index) {
      if (compressedSizes[index] == pos_ret7)
        ret7 = sizes[index];
    }
  } else {
#if FULL_FLEDGED
    ret7 = DPconv<64>::runMinMaxDPconv_boosted(logMaxCard, sizes);
#else
    ret7 = DPconvImpl<64>::runMinMaxDPconv_boosted(sizes);
#endif
  }
#else
  auto ret7 = 0;
#endif
  std::cerr << "[STOP] " << fn << " ret7=" << ret7 << std::endl;
#endif
  auto time8 = std::chrono::high_resolution_clock::now();

  // ------------------------------------------------- //

  // Reset `time8`.
  time8 = std::chrono::high_resolution_clock::now();

  // ------------------------------------------------- //







  // `runMinMaxDPconv_instant` with naive boolean FSC
  std::cerr << "[START] " << fn << " {MINMAX_DPCONV_INSTANT_NAIVE}" << std::endl;
#if MINMAX_DPCONV_INSTANT_NAIVE
  auto [ret8, join_tree8] = runMinMaxDPconv_instant<BooleanFSC>(sizes);
#else
  auto ret8 = 0;
#endif
  std::cerr << "[STOP] " << fn << " ret8=" << ret8 << std::endl;
  auto time9 = std::chrono::high_resolution_clock::now();
  
  // !!! Set time !!!
  auto minmax_dpconv_instant_naive_time = std::chrono::duration_cast<std::chrono::microseconds>(time9 - time8).count();
  
  // Check for correctness.
#if MINMAX_DPCONV_INSTANT_NAIVE
  assert(computeJoinTreeMinMaxCost(join_tree8, sizes) == ret8);
#endif
  std::cerr << "MINMAX_DPCONV_INSTANT_NAIVE (time): " << std::chrono::duration_cast<std::chrono::microseconds>(time9 - time8).count() << std::endl;
  // ------------------------------------------------- //

  // Reset `time9`.
  time9 = std::chrono::high_resolution_clock::now();

  // ------------------------------------------------- //






  // `runMinMaxDPconv_instant` with naive boolean FSC
  std::cerr << "[START] " << fn << " {MINMAX_DPCONV_INSTANT_BOOSTED}" << std::endl;
#if MINMAX_DPCONV_INSTANT_BOOSTED
  auto [ret9, join_tree9] = runMinMaxDPconv_instant<BoostedBooleanFSC>(sizes);
#else
  auto ret9 = 0;
#endif
  std::cerr << "[STOP] " << fn << " ret9=" << ret9 << std::endl;
  auto time10 = std::chrono::high_resolution_clock::now();

  // !!! Set time !!!
  auto minmax_dpconv_instant_boosted_time = std::chrono::duration_cast<std::chrono::microseconds>(time10 - time9).count();

  // Check for correctness.
#if MINMAX_DPCONV_INSTANT_BOOSTED
  assert(computeJoinTreeMinMaxCost(join_tree9, sizes) == ret9);
#endif
  std::cerr << "MINMAX_DPCONV_INSTANT_BOOSTED (time): " << std::chrono::duration_cast<std::chrono::microseconds>(time10 - time9).count() << std::endl;
  // ------------------------------------------------- //

  // Reset `time10`.
  time10 = std::chrono::high_resolution_clock::now();

  // ------------------------------------------------- //






  // Capped C_{out} with DPsub + DPconv.
  std::cerr << "[START] " << fn << " {CAPPED_DPSUB_INSTANT_BOOSTED}" << std::endl;
#if CAPPED_DPSUB_INSTANT_BOOSTED
  // Note: We don't build the join tree! 
  auto [tmp_ret10, tmp_join_tree10] = runMinMaxDPconv_instant<BoostedBooleanFSC>(sizes, false);
  std::cerr << "tmp_ret10=" << tmp_ret10 << std::endl;
  auto [ret10, join_tree10] = runMinMaxDPsub_hybrid(sizes, tmp_ret10);
#else
  auto ret10 = 0;
#endif
  std::cerr << "[STOP] " << fn << " ret10=" << ret10 << std::endl;
  auto time11 = std::chrono::high_resolution_clock::now();

  // !!! Set time !!!
  auto hybrid_capped_dpsub_time = std::chrono::duration_cast<std::chrono::microseconds>(time11 - time10).count();
  
  std::cerr << "CAPPED_DPSUB_INSTANT_BOOSTED (time): " << std::chrono::duration_cast<std::chrono::microseconds>(time11 - time10).count() << std::endl;
  // ------------------------------------------------- //

  // Reset `time11`.
  time11 = std::chrono::high_resolution_clock::now();

  // ------------------------------------------------- //








  // Capped C_{out} with DPccp + DPccp.
  std::cerr << "[START] " << fn << " {CAPPED_DPCCP_SIMPLE}" << std::endl;
#if CAPPED_DPCCP_SIMPLE
  // Note: We don't build the join tree!
  auto capped_wrapper = DPccpWrapper(OptimizationType::MinMax, qg, sizes);
  auto [capped_size11, _] = capped_wrapper.runDPccp(false);
  capped_wrapper.reset(OptimizationType::MinPlus, capped_size11);
  auto [ret11, join_tree11] = capped_wrapper.runDPccp(true);
#else
  auto ret11 = 0;
#endif
  std::cerr << "[STOP] " << fn << " ret11=" << ret11 << std::endl;
  auto time12 = std::chrono::high_resolution_clock::now();

  // !!! Set time !!!
  auto simple_capped_dpccp_time = std::chrono::duration_cast<std::chrono::microseconds>(time12 - time11).count();
  
#if CAPPED_DPCCP_SIMPLE
  assert(computeJoinTreeMinMaxCost(join_tree11, sizes) == capped_size11);
#endif
  std::cerr << "CAPPED_DPCCP_SIMPLE (time): " << std::chrono::duration_cast<std::chrono::microseconds>(time12 - time11).count() << std::endl;
  // ------------------------------------------------- //

  // Reset `time12`.
  time12 = std::chrono::high_resolution_clock::now();

  // ------------------------------------------------- //









  // Capped C_{out} with DPsub + DPconv.
  std::cerr << "[START] " << fn << " {CAPPED_DPSUB_SIMPLE}" << std::endl;
#if CAPPED_DPSUB_SIMPLE
  // Note: We don't build the join tree!
  auto [capped_size12, __] = runMinMaxDPsub(sizes, false);
  auto [ret12, join_tree12] = runMinMaxDPsub_hybrid(sizes, capped_size12);
#else
  auto ret12 = 0;
#endif
  std::cerr << "[STOP] " << fn << " ret12=" << ret12 << std::endl;
  auto time13 = std::chrono::high_resolution_clock::now();

  // !!! Set time !!!
  auto simple_capped_dpsub_time = std::chrono::duration_cast<std::chrono::microseconds>(time13 - time12).count();

#if CAPPED_DPSUB_SIMPLE
  assert(computeJoinTreeMinMaxCost(join_tree12, sizes) == capped_size12);
#endif
  std::cerr << "CAPPED_DPSUB_SIMPLE (time): " << std::chrono::duration_cast<std::chrono::microseconds>(time13 - time12).count() << std::endl;
  // ------------------------------------------------- //

  // Reset `time12`.
  time13 = std::chrono::high_resolution_clock::now();

  // ------------------------------------------------- //


  // Optimal C_{out} with DPccp.
  std::cerr << "[START] " << fn << " {MINPLUS_DPCCP}" << std::endl;
#if MINPLUS_DPCCP
  auto minplus_wrapper = DPccpWrapper(OptimizationType::MinPlus, qg, sizes);
  auto [ret13, join_tree13] = minplus_wrapper.runDPccp();
#else
  auto ret13 = 0;
#endif
  std::cerr << "[STOP] " << fn << " ret13=" << ret13 << std::endl;
  auto time14 = std::chrono::high_resolution_clock::now();

  // !!! Set time !!!
  auto minplus_dpccp_time = std::chrono::duration_cast<std::chrono::microseconds>(time14 - time13).count();
  
  // This helps us to show that Cout doesn't always output the Cmax-optimal plan.
#if MINPLUS_DPCCP
  auto max_inter_size14 = computeJoinTreeMinMaxCost(join_tree13, sizes);
#else
  auto max_inter_size14 = 0;
#endif
  // ------------------------------------------------- //

  time14 = std::chrono::high_resolution_clock::now();

  // ------------------------------------------------- //

  // Optimal C_{out} with DPsub.
  std::cerr << "[START] " << fn << " {MINPLUS_DPSUB}" << std::endl;
#if MINPLUS_DPSUB
  auto [ret14, join_tree14] = runMinPlusDPsub(sizes);
#else
  auto ret14 = 0;
#endif
  std::cerr << "[STOP] " << fn << " ret14=" << ret14 << std::endl;
  auto time15 = std::chrono::high_resolution_clock::now();

  // !!! Set time !!!
  auto minplus_dpsub_time = std::chrono::duration_cast<std::chrono::microseconds>(time15 - time14).count();

  // This helps us to show that Cout doesn't always output the Cmax-optimal plan.
#if MINPLUS_DPSUB
  auto max_inter_size15 = computeJoinTreeMinMaxCost(join_tree14, sizes);
#else
  auto max_inter_size15 = 0;
#endif
  // ------------------------------------------------- //

  // time15 = std::chrono::high_resolution_clock::now();

  // ------------------------------------------------- //

  // assert(max_inter_size14 == max_inter_size15);
  auto max_inter_size = std::max(static_cast<uint64_t>(max_inter_size14), static_cast<uint64_t>(max_inter_size15));

  // Check whether capped Cout has been optmized correctly.
  // assert(tmp_ret10 == ret1);
  // auto bounded_sizes = sizes;
  // for (unsigned index = 0, limit = sizes.size(); index != limit; ++index) {
  //   if ((index & (index - 1)) && (index != limit - 1)) {
  //     if (sizes[index] > tmp_ret10) {
  //       bounded_sizes[index] = std::numeric_limits<uint64_t>::max();
  //     }
  //   }
  // }
  // auto [bounded_opt_ret, bounded_opt_join_tree] = runMinPlusDPsub(bounded_sizes);
  // assert(bounded_opt_ret == ret10);


  double ratio_cout = 0.0, ratio_cmax = 0.0;
  auto cmax_when_cout = 0.0;
  auto cout_when_cmax = 0.0;
  double cmax_opt = 0.0;
  double cout_opt = 0.0;
  double bounded_cout_opt = 0.0;
  if (analyzeRatios) {
    join_tree14->debug(fn, tns, sizes, "cout_opt");
    join_tree1->debug(fn, tns, sizes, "cmax_opt");

    cout_opt = ret14;
    cmax_opt = ret1;
    bounded_cout_opt = ret12;

    join_tree12->debug(fn, tns, sizes, "ccap_opt");

    // Test if Cmax is preserved.
    assert(computeJoinTreeMinMaxCost(join_tree12, sizes) == cmax_opt);

    // Compute the C_{out}-cost of the join tree from DPsub with C_{max}.
    cout_when_cmax = computeJoinTreeMinPlusCost(join_tree1, sizes);
    cmax_when_cout = computeJoinTreeMinMaxCost(join_tree14, sizes);
  }

  // Exact.
  // assert(ret1 == ret5);
  // assert(ret1 == ret6);
  // assert(ret1 == ret7);
  // assert(ret0 == ret1);
#if MINMAX_DPCONV_INSTANT_NAIVE
  assert(ret1 == ret8);
#endif

#if MINMAX_DPCONV_INSTANT_BOOSTED
  assert(ret1 == ret9);
#endif

#if CAPPED_DPCCP_SIMPLE
  assert(ret11 == ret12);
#endif

#if MINPLUS_DPCCP
  assert(ret13 == ret14);
#endif

  auto ret = fn + ","
          + std::to_string(n) + ","
          + std::to_string(max_inter_size) + ","
          + std::to_string(ret0) + "," + std::to_string(minmax_dpccp_time) + ","
          + std::to_string(ret1) + "," + std::to_string(minmax_dpsub_time) + ","
          // + std::to_string(ret2) + "," + std::to_string(approx_minmax_dpsub_time) + ","
          // + std::to_string(ret3) + "," + std::to_string(approx_minmax_dpconv_naive_time) + ","
          // + std::to_string(ret4) + "," + std::to_string(approx_minmax_dpconv_boosted_time) + ","
          // + std::to_string(ret5) + "," + std::to_string(minmax_dpconv_naive_time) + ","
          // + std::to_string(ret6) + "," + std::to_string(minmax_dpconv_layered_time) + ","
          // + std::to_string(ret7) + "," + std::to_string(minmax_dpconv_boosted_time) + ","
          + std::to_string(ret8) + "," + std::to_string(minmax_dpconv_instant_naive_time) + ","
          + std::to_string(ret9) + "," + std::to_string(minmax_dpconv_instant_boosted_time) + ","
          + std::to_string(ret10) + "," + std::to_string(hybrid_capped_dpsub_time) + ","

          + std::to_string(ret11) + "," + std::to_string(simple_capped_dpccp_time) + ","
          + std::to_string(ret12) + "," + std::to_string(simple_capped_dpsub_time) + ","
          
          + std::to_string(ret13) + "," + std::to_string(minplus_dpccp_time) + ","
          + std::to_string(ret14) + "," + std::to_string(minplus_dpsub_time);


  std::cerr << ret << std::endl;

  std::cerr << "[stop] " << fn << std::endl;

  MetaInfo meta_info;
  meta_info.cout_opt = cout_opt;
  meta_info.cout_when_cmax = cout_when_cmax;
  meta_info.cmax_opt = cmax_opt;
  meta_info.cmax_when_cout = cmax_when_cout;
  meta_info.bounded_cout_opt = bounded_cout_opt;
  return {meta_info, ret};
}

const std::string currentDateTime() {
  time_t     now = time(0);
  struct tm  tstruct;
  char       buf[80];
  tstruct = *localtime(&now);
  // Visit http://en.cppreference.com/w/cpp/chrono/c/strftime for more information about date/time format
  strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);
  return buf;
}

void flush(std::string type, std::vector<std::string>& collector, bool isFinal=false) {
  std::string extra;
  unsigned clique_bench = 0;
  unsigned cap_job_bench = 0;
  unsigned cap_clique_bench = 0;
  unsigned ceb_bench = 0;
#if CLIQUE_BENCHMARK
  clique_bench = 1;
#endif

#if CAPPED_COUT_BENCHMARK
  cap_job_bench = 1;
#endif

#if CLIQUE_CAPPED_COUT_BENCHMARK
  cap_clique_bench = 1;
#endif

#if CEB_BENCHMARK
  ceb_bench = 1;
#endif
  assert(clique_bench + cap_job_bench + cap_clique_bench + ceb_bench == 1);

  if (clique_bench)
    extra = "normal-clique";
  else if (cap_job_bench)
    extra = "cap-cout";
  else
    extra = "cap-clique";

  if (isFinal)
    extra += "-final";
  std::ofstream out("../benchs/" + extra + "-" + currentDateTime() + ".csv");
  assert(out.is_open());
  out << "query" << ","
      << "size" << ","
      << "cout_max_inter_size" << ","
      << "minmax_dpccp_cost,minmax_dpccp_time [mus]" << ","
      << "minmax_dpsub_cost,minmax_dpsub_time [mus]" << ","
      // << "approx_minmax_dpsub_cost,approx_minmax_dpsub_time [mus]" << ","
      // << "approx_minmax_dpconv_naive_cost,approx_minmax_dpconv_naive_time [mus]" << ","
      // << "approx_minmax_dpconv_boosted_cost,approx_minmax_dpconv_boosted_time [mus]" << ","
      // << "minmax_dpconv_naive_cost,minmax_dpconv_naive_time [mus]" << ","
      // << "minmax_dpconv_layered_cost,minmax_dpconv_layered_time [mus]" << ","
      // << "minmax_dpconv_boosted_cost,minmax_dpconv_boosted_time [mus]" << ","
      << "minmax_dpconv_instant_naive_cost,minmax_dpconv_instant_naive_time [mus]" << ","
      << "minmax_dpconv_instant_boosted_cost,minmax_dpconv_instant_boosted_time [mus]" << ","

      << "hybrid_capped_dpsub_cost,hybrid_capped_dpsub_time [mus]" << ","

      << "simple_capped_dpccp_cost,simple_capped_dpccp_time [mus]" << ","
      << "simple_capped_dpsub_cost,simple_capped_dpsub_time [mus]" << ","

      << "minplus_dpccp_cost,minplus_dpccp_time [mus]" << ","
      << "minplus_dpsub_cost,minplus_dpsub_time [mus]" << std::endl;
  for (auto elem : collector) {
    out << elem << "\n";
  }
}