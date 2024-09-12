// This implements the semi-ring for Sec. 3.4, using a flattened array for _all_ subsets.
// Used by `DPconv_old.cpp`.

#include "TypeUtil.hpp"

#include <cstring>

using type_t = __int128_t;

template <unsigned w, unsigned kForcedShift = 0>
class ContArray {
  static constexpr unsigned kSize = 1 + w + kForcedShift;

public:
  class Chunk {
  private:
    type_t* freqs;

    static void inplace_add(type_t* x, type_t* y) { for (unsigned index = 0; index != kSize; ++index) x[index] += y[index]; }
    static void inplace_minus(type_t* x, type_t* y) { for (unsigned index = 0; index != kSize; ++index) x[index] -= y[index]; }

    static unsigned getFirstPositivePosition(type_t* x) {
      for (unsigned index = 0; index != kSize; ++index) {
        if (x[index] > 0)
          return index;
      }
      return kSize;
    }

  public:
    Chunk(type_t* ptr) : freqs(ptr) {}

    type_t operator [](unsigned index) const { return freqs[index]; }

    void copy(Chunk&& other) {
      for (unsigned index = 0; index != kSize; ++index)
        freqs[index] = other.freqs[index];
    }

    template <class U>
    void set(U x) {
      auto log = computeLog<w, kForcedShift>(x);
      assert(log < kSize);
      freqs[log] = 1;
    }

    void operator += (const Chunk& other) { inplace_add(freqs, other.freqs); }
    void operator -= (const Chunk& other) { inplace_minus(freqs, other.freqs); }

#define POS(a) (((a) > 0) ? (a) : 0)
    // `Update` implements *this += f * xs * ys
    template <unsigned f>
    void update(const Chunk& xs, const Chunk& ys) {
      // Update the first position.
      freqs[0] += f * POS(xs[0]) * POS(ys[0]);

      type_t curr_sum = POS(ys[0]), mars_val = POS(xs[0]);
      for (unsigned index = 1; index != kSize; ++index) {
        // Cumulate the contribution from those below me.
        auto curr_ret = curr_sum * POS(xs[index]); 

        // Announce the ones above me that they will also get my contribution.
        mars_val += POS(xs[index]);

        // Compute from right to left.
        curr_ret += POS(ys[index]) * mars_val;

        // Update the partial sum.
        curr_sum += POS(ys[index]);
      
        // Update the value.
        freqs[index] += f * curr_ret;
      }
    }

    // `Set` implements *this = xs * ys.
    void set(const Chunk& xs, const Chunk& ys) {
      // Set the first position.
      freqs[0] = POS(xs[0]) * POS(ys[0]);

      type_t curr_sum = POS(ys[0]), mars_val = POS(xs[0]);
      for (unsigned index = 1; index != kSize; ++index) {
        // Cumulate the contribution from those below me.
        freqs[index] = curr_sum * POS(xs[index]); 

        // Announce the ones above me that they will also get my contribution.
        mars_val += POS(xs[index]);

        // Compute from right to left.
        freqs[index] += POS(ys[index]) * mars_val;

        // Update the partial sum.
        curr_sum += POS(ys[index]);
      }
    }

    // `Set` implements *this = xs * val.
    template <class U>
    void set(const Chunk& xs, U val) {
      // Set the first position.
      auto log = computeLog<w, kForcedShift>(val);

      // Reset the frequencies.
      std::memset(freqs, 0, kSize);

      // Acumulate the values from below.
      for (unsigned index = 0; index <= log; ++index)
        freqs[log] += POS(xs[index]);

      // Simply take the values from `xs`.
      for (unsigned index = log + 1; index != kSize; ++index)
        freqs[index] = POS(xs[index]);
    }
#undef POS

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
      auto pos = getFirstPositivePosition(freqs);
      if (kForcedShift)
        return {pos - 1, pos != 0};
      return {pos, true};
    }

    friend std::ostream& operator<< (std::ostream& os, const Chunk& other) {
      for (unsigned index = 0, limit = kSize; index != limit; ++index)
        os << other.freqs[index] << " ";
      return os;
    }
  };

  ContArray(unsigned size) : size_(size) {
    data_.assign(kSize * size, 0);
  }

  Chunk operator [](unsigned index) {
    assert(index < size_);
    return Chunk(&data_[kSize * index]);
  }

  std::size_t size() const {
    return size_;
  }

private:
  std::size_t size_;
  std::vector<type_t> data_;
};