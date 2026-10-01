//===- ZeroDomain.h - The abstract domain ---------------------------------===//
//
// A four-point lattice recording whether an integer value is known to be zero.
//
//        Top          nothing is known
//       /   \
//    Zero  NonZero
//       \   /
//       Bottom       unreachable, or not yet analyzed
//
// This is the file to replace first when building a different analysis.  MLIR's
// dataflow framework asks only three things of a lattice value:
//
//   * a default constructor, which must produce the bottom element, because the
//     solver starts every value optimistically and lowers it as facts arrive;
//   * a static join(), which must be commutative, associative, idempotent, and
//     monotone -- assertions in Lattice<> check monotonicity in debug builds;
//   * operator== and print().
//
//===----------------------------------------------------------------------===//

#ifndef ZERO_DOMAIN_H
#define ZERO_DOMAIN_H

#include <algorithm>
#include "FloatFormats.h"
#include "llvm/Support/raw_ostream.h"

namespace zero {

enum class Kind { Bottom, Zero, NonZero, Top };

inline const char *name(Kind kind) {
  switch (kind) {
  case Kind::Bottom:
    return "bottom";
  case Kind::Zero:
    return "zero";
  case Kind::NonZero:
    return "nonzero";
  case Kind::Top:
    return "top";
  }
  return "top";
}

struct IntReprState {
  Kind kind = Kind::Bottom;
  int lowestOne = -1;
  int highestOne = -1;

  IntReprState() = default;
  /* implicit */ IntReprState(Kind kind) : kind(kind) {}
  IntReprState(Kind kind, int lowestOne, int highestOne)
      : kind(kind), lowestOne(lowestOne), highestOne(highestOne) {}

  static IntReprState bottom() { return Kind::Bottom; }
  static IntReprState top() { return Kind::Top; }

  bool isBottom() const { return kind == Kind::Bottom; }

  int lpo() {
    if (lowestOne == -1)
      return 0;
    return lowestOne;
  }

  int hpo(int width) {
    if (highestOne == -1)
      return width - 1;
    return highestOne;
  }

  static IntReprState join(const IntReprState &lhs, const IntReprState &rhs) {
    if (lhs.kind == Kind::Bottom)
      return rhs;
    if (rhs.kind == Kind::Bottom)
      return lhs;
    auto joinedKind = lhs.kind == rhs.kind ? lhs.kind : Kind::Top;
    if (lhs.kind == Kind::Top || rhs.kind == Kind::Top)
      joinedKind = Kind::Top;

    if (lhs.kind == Kind::Zero)
      return {joinedKind, rhs.lowestOne, rhs.highestOne};
    if (rhs.kind == Kind::Zero)
      return {joinedKind, lhs.lowestOne, lhs.highestOne};

    if (lhs.lowestOne == -1 || rhs.lowestOne == -1)
      return {joinedKind, -1, -1};
    return {joinedKind, std::min(lhs.lowestOne, rhs.lowestOne),
            std::max(lhs.highestOne, rhs.highestOne)};
  }

  bool operator==(const IntReprState &other) const {
    return kind == other.kind && lowestOne == other.lowestOne &&
           highestOne == other.highestOne;
  }
  bool operator!=(const IntReprState &other) const { return !(*this == other); }

  void print(llvm::raw_ostream &os) const {
    os << name(kind);
    bool first = true;
    for (const auto &format : formats) {
      if (!format.can_represent(*this))
        continue;
      os << (first ? "; available \"int\" representations" : ", ") << format.name;
      first = false;
    }
  }
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const IntReprState &state) {
  state.print(os);
  return os;
}

} // namespace zero

#endif
