// A generator for cardinalities, which respects the cross-product condition.

#ifndef H_Generator
#define H_Generator

#include "Common.hpp"

template <class U>
static std::vector<U> gen_random_sizes(unsigned n, unsigned W, unsigned min_base_card, unsigned max_base_card, unsigned seed_val) {
  // Note: This is the correct place where we should instantiate the random generator.
  // Putting it inside the lambda creates the same numbers for a given arity.
  std::seed_seq seed{seed_val};
  std::mt19937 gen(seed);

  std::vector<U> sizes(1ull << n);

  auto gen_card = [&](uint64_t S) {
    auto num_relations = __builtin_popcount(S);

    // Account for the case distinction made in `mutable`, that base tables should have a bounded cardinality.
    if (num_relations == 1) {
      std::uniform_int_distribution<> distribution(min_base_card, max_base_card);
      return distribution(gen);
    } else {
      uint64_t current = 0;
      long double max_card = 0.0;
      do {
        if ((current != 0) && (current != S)) {
          max_card = std::max(
            max_card,
            static_cast<long double>(sizes[current]) * static_cast<long double>(sizes[S ^ current])
          );
        }
        current = (current - S) & S;
      } while (current);
      unsigned w = 2 * W / num_relations;
      
      // Lower the `w` if that bound is lower.
      if (max_card < static_cast<long double>(w)) {
        w = static_cast<unsigned>(max_card);
      }
      std::uniform_int_distribution<> distribution(1, w);
      return distribution(gen);
    }
  };
  
  sizes[0] = 0;
  for (uint64_t index = 1; index != (1ull << n); ++index) {
    sizes[index] = gen_card(index);
  }
  return sizes;
}

#endif 