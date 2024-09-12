#pragma once

#include <cmath>
#include <limits>
#include <memory>
#include <cassert>

static bool globalFlag = false; // Global flag initialized to false

// Convert __int128_t to a string and output it to the ostream
static std::ostream& operator<<(std::ostream& os, __int128_t value) {
  char buffer[128];
  snprintf(buffer, sizeof(buffer), "%lld", static_cast<long long>(value));
  return os << buffer;
}

// The template `U` is used for either `uint64_t` (for true cardinalities) or `unsigned` (for compressed cardinalities).
// In the latter case, we actually don't need the case for `x == std::numeric_limits<U>::max`, since that would always be the last compressed cardinality.
template <unsigned w, unsigned kForcedShift>
inline static unsigned computeLog(uint64_t x) {
  // Round down to the next power of two.
  // In case `x` equals 0, we introduce an artifical slot to simulate `-\infty`, which is freqs[0].
  // Thus, the entire array is shifted to the right by one.
  unsigned log;
  if (kForcedShift == 1) {
    // Is this subgraph unconnected?
    if (x == std::numeric_limits<uint64_t>::max()) {
      // Then take the max. possible log.
      log = 1 + w;
    } else {
      log = (!x) ? 0 : (1 + 63 - __builtin_clzll(x));
    }
  } else {
    // Is this subgraph unconnected?
    if (x == std::numeric_limits<uint64_t>::max()) {
      // Then take the maximum possible log.
      log = w;
    } else {
      // Otherwise, treat normally.
      assert(x);
      log = 63 - __builtin_clzll(x);
    }
  }
  return log;
}

template <unsigned w, unsigned kForcedShift>
inline static unsigned computeLog(uint32_t x) {
  // Round down to the next power of two.
  // In case `x` equals 0, we introduce an artifical slot to simulate `-\infty`, which is freqs[0].
  // Thus, the entire array is shifted to the right by one.
  if (kForcedShift == 1)
    return (!x) ? 0 : (1 + 31 - __builtin_clz(x));
  
  return 31 - __builtin_clz(x);
}