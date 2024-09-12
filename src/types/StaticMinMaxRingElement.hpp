// This implements a semi-ring element with a custom width for Sec. 3.4.
// Used by `DPconv_semiring.cpp`.
// See `DPconv.cpp` for the simpler algorithm.

#include "TypeUtil.hpp"

using int128_t = __int128_t;

template <unsigned w, unsigned kForcedShift = 0>
class StaticMinMaxRingElement {
  // To support sizes equal to 0, we add an artificial slot at the beginning which simulates that.
  static constexpr unsigned kSize = 1 + w + kForcedShift;

#define array_t std::array<int128_t, kSize>
  // This is actually a set which stores the logs of the contraction costs.
  // Since the number of unique elements is low, we maintain it as a frequency array.
  array_t freqs{};

  static array_t add(const array_t& x, const array_t& y) {
    array_t ret{};
    for (unsigned index = 0; index != kSize; ++index)
      ret[index] = x[index] + y[index];
    return ret;
  }

  static void inplace_add(array_t& x, const array_t& y) {
    for (unsigned index = 0; index != kSize; ++index)
      x[index] += y[index];
  }

  static array_t minus(const array_t& x, const array_t& y) {
    array_t ret{};
    for (unsigned index = 0; index != kSize; ++index)
      ret[index] = x[index] - y[index];
    return ret;
  }

  static void inplace_minus(array_t& x, const array_t& y) {
    for (unsigned index = 0; index != kSize; ++index)
      x[index] -= y[index];
  }

#define POS(a) (((a) > 0) ? (a) : 0)
  static array_t mul(const array_t& xs, const array_t& ys) {
    // The result.
    array_t ret{};

    // Init the first position.
    ret[0] = POS(xs[0]) * POS(ys[0]);

    int128_t curr_sum = POS(ys[0]), mars_val = POS(xs[0]);
    for (unsigned index = 1; index != kSize; ++index) {
      // Cumulate the contribution from those below me.
      ret[index] += curr_sum * POS(xs[index]); 

      // Announce the ones above me that they will also get my contribution.
      mars_val += POS(xs[index]);

      // Compute from right to left.
      ret[index] += POS(ys[index]) * mars_val;

      // Update the partial sum.
      curr_sum += POS(ys[index]);
    }
    return ret;
  }

  static void inplace_mul(array_t& xs, const array_t& ys) {
    // Init the first position.
    int128_t curr_sum = POS(ys[0]), mars_val = POS(xs[0]);

    // Update the first position. Note: this should always come before the last line.
    xs[0] = POS(xs[0]) * POS(ys[0]);

    for (unsigned index = 1; index != kSize; ++index) {
      // Cumulate the contribution from those below me.
      auto curr_ret = curr_sum * POS(xs[index]);

      // Announce the ones above me that they will also get my contribution.
      mars_val += POS(xs[index]);

      // Compute from right to left.
      curr_ret += POS(ys[index]) * mars_val;

      // Update the partial sum.
      curr_sum += POS(ys[index]);

      // And store the result.    
      xs[index] = curr_ret;
    }
  }
#undef POS

  static unsigned getFirstPositivePosition(const array_t& x) {
    for (unsigned index = 0; index != kSize; ++index) {
      if (x[index] > 0)
        return index;
    }
    return kSize;
  }

  static void debug_array(const array_t& xs) {
    for (unsigned pos = 0; pos != kSize; ++pos)
      std::cerr << xs[pos] << " ";
    std::cerr << std::endl;
  }

public:
  static constexpr unsigned forcedShift = kForcedShift;
  StaticMinMaxRingElement() {
    // This really represents the +oo value we're referring to in the paper.
  }

  StaticMinMaxRingElement(array_t&& freqs) { this->freqs = freqs; }

  StaticMinMaxRingElement(const StaticMinMaxRingElement& other) { this->freqs = other.freqs; } 

  StaticMinMaxRingElement& operator = (const StaticMinMaxRingElement& other) { this->freqs = other.freqs; return *this; }

  template <class U>
  void set(U x) {
    auto log = computeLog<w, kForcedShift>(x);
    assert(log < kSize);
    freqs[log] = 1;
  }

  // This simulates `min`. This is actually a set operation.
  StaticMinMaxRingElement operator + (const StaticMinMaxRingElement& other) const { return StaticMinMaxRingElement(add(freqs, other.freqs)); }
  
  // This is the reverse operation. This is actually a set operation.
  StaticMinMaxRingElement operator - (const StaticMinMaxRingElement& other) const { return StaticMinMaxRingElement(minus(freqs, other.freqs)); }

  StaticMinMaxRingElement operator * (const StaticMinMaxRingElement& other) const { return StaticMinMaxRingElement(mul(freqs, other.freqs)); }

  void operator *= (unsigned x) { for (unsigned pos = 0; pos != kSize; ++pos) freqs[pos] *= x; }

  void operator += (const StaticMinMaxRingElement& other) { inplace_add(freqs, other.freqs); }
  void operator -= (const StaticMinMaxRingElement& other) { inplace_minus(freqs, other.freqs); }
  void operator *= (const StaticMinMaxRingElement& other) { inplace_mul(freqs, other.freqs); }

  // Only used to compare the actual value, not the abstract representation.
  bool operator == (StaticMinMaxRingElement&& other) const {
    return freqs == other.freqs;
  }

  uint64_t getValue() const {
    auto pos = getFirstPositivePosition(freqs);
    if (kForcedShift)
      return (pos) ? (1ULL << (pos - 1)) : 0;

    // Treat the extreme case.
    if (pos == 64)
      return std::numeric_limits<uint64_t>::max();
    return (1ULL << pos);
  }

  std::pair<unsigned, bool> getLogValue() const {
    // std::cerr << "\n[getLogValue] " << *this << std::endl;
    auto pos = getFirstPositivePosition(freqs);
    if (kForcedShift)
      return {pos - 1, pos != 0};
    return {pos, true};
  }

  friend std::ostream& operator<< (std::ostream& os, const StaticMinMaxRingElement& other) {
    for (unsigned index = 0, limit = other.freqs.size(); index != limit; ++index)
      os << other.freqs[index] << " ";
    return os;
  }
};
#undef array_t