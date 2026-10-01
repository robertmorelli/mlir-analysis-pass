#ifndef FLOAT_FORMATS_H
#define FLOAT_FORMATS_H

namespace zero {

struct IntReprState;

struct FloatFormats {
  const char *name;
  unsigned significandBits;
  unsigned exponentBits;

  bool can_represent(const IntReprState &state) const;
};

extern FloatFormats formats[7];

}

#endif
