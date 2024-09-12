// This implements a semi-ring element with a custom width for Sec. 3.4.
// Used by `DPconv_semiring.cpp`.
// See `DPconv.cpp` for the simpler algorithm.

#include <vector>

template <unsigned kForcedShift = 0>
class DynamicMinMaxRingElement {
#define array_t std::vector<int64_t>
  // This is actually a set which stores the logs of the contraction costs.
  // Since the number of unique elements is low, we maintain it as a frequency array.
  array_t freqs;

  static array_t add(const array_t& x, const array_t& y) {
    unsigned array_size = x.size();
    array_t ret(array_size, 0);
    for (unsigned index = 0; index != array_size; ++index)
      ret[index] = x[index] + y[index];
    return ret;
  }

  static void inplace_add(array_t& x, const array_t& y) {
    for (unsigned index = 0, limit = x.size(); index != limit; ++index)
      x[index] += y[index];
  }

  static array_t minus(const array_t& x, const array_t& y) {
    unsigned array_size = x.size();
    array_t ret(array_size, 0);
    for (unsigned index = 0; index != array_size; ++index)
      ret[index] = x[index] - y[index];
    return ret;
  }

  static void inplace_minus(array_t& x, const array_t& y) {
    for (unsigned index = 0, limit = x.size(); index != limit; ++index)
      x[index] -= y[index];
  }

#define POS(a) (((a) > 0) ? (a) : 0)
  static array_t mul(const array_t& xs, const array_t& ys) {
    // The result.
    unsigned array_size = xs.size();
    array_t ret(array_size, 0);

    // Init the first position.
    ret[0] = POS(xs[0]) * POS(ys[0]);

    int64_t curr_sum = POS(ys[0]), mars_val = POS(xs[0]);
    for (unsigned index = 1; index != array_size; ++index) {
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
    int64_t curr_sum = POS(ys[0]), mars_val = POS(xs[0]);

    // Update the first position. Note: this should always come before the last line.
    xs[0] = POS(xs[0]) * POS(ys[0]);

    for (unsigned index = 1, limit = xs.size(); index != limit; ++index) {
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

  static unsigned getFirstPositivePosition(const array_t& xs) {
    // TODO: Non-empty or positive? Can we go into negative?
    // TODO: check with (+, *) actually and see whether we go into negative.
    // TODO: But only with the logs! (small numbers)
    for (unsigned index = 0, limit = xs.size(); index != limit; ++index) {
      if (xs[index] > 0)
        return index;
    }
    return xs.size();
  }

  static void debug_array(const array_t& xs) {
    for (unsigned pos = 0, limit = xs.size(); pos != limit; ++pos)
      std::cerr << xs[pos] << " ";
    std::cerr << std::endl;
  }

public:
  static constexpr unsigned forcedShift = kForcedShift;

  DynamicMinMaxRingElement(unsigned const_width) {
    // Initialize the frequency vector.
    // To support sizes equal to 0, we add an artificial slot at the beginning which simulates that.
    freqs.assign(1 + const_width + forcedShift, 0);
    // This really represents the +oo value we're referring to in the paper.
  }

  DynamicMinMaxRingElement(uint64_t x, unsigned const_width = -1) {
    assert(0);
  }

  DynamicMinMaxRingElement(const DynamicMinMaxRingElement& other) { this->freqs = other.freqs; } 

  DynamicMinMaxRingElement& operator = (const DynamicMinMaxRingElement& other) { this->freqs = other.freqs; return *this; }

  void set(uint64_t x) {
    assert(!freqs.empty());
    // Round down to the next power of two.
    // In case `x` equals 0, we introduce an artifical slot to simulate `-\infty`, which is freqs[0].
    // Thus, the entire array is shifted to the right by one.
    unsigned log;
    if (kForcedShift == 1) {
      // Is this subgraph unconnected?
      if (x == std::numeric_limits<uint64_t>::max()) {
        // Then take the max. possible log.
        log = 1 + 64;
      } else {
        log = (!x) ? 0 : (1 + static_cast<unsigned>(std::log2(x)));
      }
    } else {
      // Is this subgraph unconnected?
      if (x == std::numeric_limits<uint64_t>::max()) {
        // Then take the maximum possible log.
        log = 64;
      } else {
        // Otherwise, treat normally.
        assert(x);
        log = static_cast<unsigned>(std::log2(x));
      }
    }
    // std::cerr << "log=" << log << std::endl;
    assert(log < freqs.size());
    freqs[log] = 1;
  }

  // This simulates `min`. This is actually a set operation.
  DynamicMinMaxRingElement operator + (const DynamicMinMaxRingElement& other) const { return DynamicMinMaxRingElement(add(freqs, other.freqs)); }
  
  // This is the reverse operation. This is actually a set operation.
  DynamicMinMaxRingElement operator - (const DynamicMinMaxRingElement& other) const { return DynamicMinMaxRingElement(minus(freqs, other.freqs)); }

  DynamicMinMaxRingElement operator * (const DynamicMinMaxRingElement& other) const { return DynamicMinMaxRingElement(mul(freqs, other.freqs)); }

  void operator *= (unsigned x) { for (unsigned pos = 0, limit = freqs.size(); pos != limit; ++pos) freqs[pos] *= x; }

  void operator += (const DynamicMinMaxRingElement& other) { inplace_add(freqs, other.freqs); }
  void operator -= (const DynamicMinMaxRingElement& other) { inplace_minus(freqs, other.freqs); }
  void operator *= (const DynamicMinMaxRingElement& other) { inplace_mul(freqs, other.freqs); }

  // Only used to compare the actual value, not the abstract representation.
  bool operator == (DynamicMinMaxRingElement&& other) const {
    return freqs == other.freqs;
    // return this->getValue() == other.getValue();
  }

  uint64_t getValue() const {
    // std::cerr << "\n[getValue] " << *this << std::endl;
    auto pos = getFirstPositivePosition(freqs);
    if (kForcedShift)
      return (pos) ? (1ULL << (pos - 1)) : 0;

    // Treat the extreme case.
    // TODO: maybe replace this by sth else? Since 64 is not anymore the maximum possible.
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

  friend std::ostream& operator<< (std::ostream& os, const DynamicMinMaxRingElement& other) {
    for (unsigned index = 0, limit = other.freqs.size(); index != limit; ++index)
      os << other.freqs[index] << " ";
    return os;
  }
};
#undef array_t