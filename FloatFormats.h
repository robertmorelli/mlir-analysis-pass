#ifndef FLOAT_FORMATS_H
#define FLOAT_FORMATS_H

namespace zero {

struct IntReprState;

struct FloatFormat {
  const char *name;
  unsigned significandBits;
  unsigned exponentBits;

  bool can_represent(const IntReprState &state) const;
};

extern FloatFormat formats[8];

}

#endif
