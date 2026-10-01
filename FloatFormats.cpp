#include "FloatFormats.h"
#include "ZeroDomain.h"

namespace zero {

FloatFormats formats[7] = {{"i8", 8, 0}, {"i16", 16, 0}, {"i32", 32, 0}, {"i64", 64, 0}, {"f16", 11, 5}, {"f32", 24, 8}, {"f64", 53, 11}};

bool FloatFormats::can_represent(const IntReprState &state) const {
  if (state.kind == Kind::Bottom)
    return false;
  if (state.kind == Kind::Zero)
    return true;
  if (state.lowestOne == -1 || state.highestOne < state.lowestOne)
    return false;
  if (significandBits == 0 || exponentBits > 31)
    return false;

  if (exponentBits == 0)
    return static_cast<unsigned>(state.highestOne) < significandBits;

  auto span = static_cast<unsigned>(state.highestOne - state.lowestOne + 1);
  auto maxExponent = (1u << (exponentBits - 1)) - 1;
  return span <= significandBits &&
         static_cast<unsigned>(state.highestOne) <= maxExponent;
}

}
