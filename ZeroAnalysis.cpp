#include "ZeroAnalysis.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;

namespace zero {

int resultBitWidth(Operation *op) {
  auto type = op->getResult(0).getType();
  return type.getIntOrFloatBitWidth();
}

IntReprState transitionConstant(llvm::APInt bits) {
  if (bits.isZero())
    return IntReprState(Kind::Zero);
  auto low = bits.countTrailingZeros();
  auto high = bits.getActiveBits() - 1;
  return IntReprState(Kind::NonZero, low, high);
}

IntReprState transitionAnd(IntReprState lhs, IntReprState rhs, int width) {
  if (lhs.kind == Kind::Zero || rhs.kind == Kind::Zero)
    return IntReprState(Kind::Zero);

  auto low = std::max(lhs.lpo(), rhs.lpo());
  auto high = std::min(lhs.hpo(width), rhs.hpo(width));
  if (low > high)
    return IntReprState(Kind::Zero);
  return IntReprState(Kind::Top, low, high);
}

IntReprState transitionOr(IntReprState lhs, IntReprState rhs, int width) {
  if (lhs.kind == Kind::Zero)
    return rhs;
  if (rhs.kind == Kind::Zero)
    return lhs;

  auto kind = Kind::Top;
  if (lhs.kind == Kind::NonZero || rhs.kind == Kind::NonZero)
    kind = Kind::NonZero;
  auto low = std::min(lhs.lpo(), rhs.lpo());
  auto high = std::max(lhs.hpo(width), rhs.hpo(width));
  return IntReprState(kind, low, high);
}

IntReprState transitionAdd(IntReprState lhs, IntReprState rhs, int width) {
  if (lhs.kind == Kind::Zero)
    return rhs;
  if (rhs.kind == Kind::Zero)
    return lhs;

  auto low = std::min(lhs.lpo(), rhs.lpo());
  auto high = std::max(lhs.hpo(width), rhs.hpo(width)) + 1;
  high = std::min(width - 1, high);
  return IntReprState(Kind::Top, low, high);
}

IntReprState transitionSub(IntReprState lhs, IntReprState rhs, int width) {
  if (rhs.kind == Kind::Zero)
    return lhs;

  if (lhs.kind == Kind::Zero) {
    auto kind = Kind::Top;
    if (rhs.kind == Kind::NonZero)
      kind = Kind::NonZero;
    return IntReprState(kind, rhs.lpo(), width - 1);
  }

  auto low = std::min(lhs.lpo(), rhs.lpo());
  return IntReprState(Kind::Top, low, width - 1);
}

IntReprState transitionMul(IntReprState lhs, IntReprState rhs, int width) {
  if (lhs.kind == Kind::Zero || rhs.kind == Kind::Zero)
    return IntReprState(Kind::Zero);

  auto low = lhs.lpo() + rhs.lpo();
  if (low >= width)
    return IntReprState(Kind::Zero);

  auto high = lhs.hpo(width) + rhs.hpo(width) + 1;
  auto kind = Kind::Top;
  auto cannotWrap = high < width;
  auto lhsNonZero = lhs.kind == Kind::NonZero;
  auto rhsNonZero = rhs.kind == Kind::NonZero;
  if (cannotWrap && lhsNonZero && rhsNonZero)
    kind = Kind::NonZero;
  high = std::min(width - 1, high);
  return IntReprState(kind, low, high);
}

IntReprState transitionUDiv(IntReprState lhs, IntReprState rhs, int width) {
  if (rhs.kind == Kind::Zero)
    return IntReprState::top();
  if (lhs.kind == Kind::Zero)
    return IntReprState(Kind::Zero);

  auto divisorIsPowerOfTwo = rhs.kind == Kind::NonZero;
  divisorIsPowerOfTwo = divisorIsPowerOfTwo && rhs.lpo() == rhs.hpo(width);
  if (divisorIsPowerOfTwo && rhs.lpo() == 0)
    return lhs;

  if (!divisorIsPowerOfTwo)
    return IntReprState(Kind::Top, 0, lhs.hpo(width));
  if (lhs.hpo(width) < rhs.lpo())
    return IntReprState(Kind::Zero);

  auto kind = Kind::Top;
  if (lhs.kind == Kind::NonZero && lhs.lpo() >= rhs.lpo())
    kind = Kind::NonZero;
  auto low = std::max(0, lhs.lpo() - rhs.lpo());
  auto high = lhs.hpo(width) - rhs.lpo();
  return IntReprState(kind, low, high);
}

IntReprState transitionURem(IntReprState lhs, IntReprState rhs, int width) {
  if (rhs.kind == Kind::Zero)
    return IntReprState::top();

  auto divisorNonZero = rhs.kind == Kind::NonZero;
  auto onlyBitZero = rhs.lpo() == 0 && rhs.hpo(width) == 0;
  if (lhs.kind == Kind::Zero || (divisorNonZero && onlyBitZero))
    return IntReprState(Kind::Zero);

  auto low = std::min(lhs.lpo(), rhs.lpo());
  auto high = std::min(width - 1, rhs.hpo(width));
  return IntReprState(Kind::Top, low, high);
}

IntReprState transitionXor(IntReprState lhs, IntReprState rhs, int width) {
  if (lhs.kind == Kind::Zero)
    return rhs;
  if (rhs.kind == Kind::Zero)
    return lhs;

  auto low = std::min(lhs.lpo(), rhs.lpo());
  auto high = std::max(lhs.hpo(width), rhs.hpo(width));
  return IntReprState(Kind::Top, low, high);
}

IntReprState transitionTrunc(IntReprState state, int width) {
  if (state.kind == Kind::Zero)
    return state;
  auto hasKnownHigh = state.highestOne != -1;
  auto highFits = state.highestOne < width;
  if (hasKnownHigh && highFits)
    return state;
  if (state.lowestOne >= width)
    return IntReprState(Kind::Zero);

  auto low = std::max(0, state.lowestOne);
  return IntReprState(Kind::Top, low, width - 1);
}

int constantShift(Operation *op, unsigned width) {
  IntegerAttr shiftAttr;
  auto matched = matchPattern(op->getOperand(1), m_Constant(&shiftAttr));
  if (!matched)
    return -1;
  if (shiftAttr.getValue().getActiveBits() > 32)
    return -1;

  auto shift = shiftAttr.getValue().getZExtValue();
  if (shift >= width)
    return -1;
  return shift;
}

IntReprState transitionShl(IntReprState state, int shift, int width) {
  if (state.kind == Kind::Zero)
    return state;
  if (shift == -1 || state.lowestOne == -1)
    return IntReprState::top();

  auto low = state.lowestOne + shift;
  auto high = state.highestOne + shift;
  if (high >= width)
    return IntReprState::top();
  return IntReprState(state.kind, low, high);
}

IntReprState transitionLShr(IntReprState state, int shift, int width) {
  if (state.kind == Kind::Zero)
    return state;
  if (shift == -1)
    return IntReprState::top();
  if (state.lowestOne == -1)
    return IntReprState(Kind::Top, 0, width - shift - 1);
  if (state.highestOne < shift)
    return IntReprState(Kind::Zero);

  auto kind = Kind::Top;
  if (state.kind == Kind::NonZero && state.lowestOne >= shift)
    kind = Kind::NonZero;
  auto low = std::max(0, state.lowestOne - shift);
  auto high = state.highestOne - shift;
  return IntReprState(kind, low, high);
}

void ZeroAnalysis::setToEntryState(ZeroLattice *lattice) {
  propagateIfChanged(lattice, lattice->join(IntReprState::top()));
}

LogicalResult ZeroAnalysis::record(ZeroLattice *lattice, IntReprState state) {
  propagateIfChanged(lattice, lattice->join(state));
  return success();
}

LogicalResult ZeroAnalysis::visitOperation(Operation *op, ArrayRef<const ZeroLattice *> operands, ArrayRef<ZeroLattice *> results) {
  if (op->getNumResults() != 1 || !op->getResult(0).getType().isIntOrIndex()) {
    setAllToEntryStates(results);
    return success();
  }
  auto *result = results[0];
 
  IntegerAttr value;
  if (matchPattern(op, m_Constant(&value)))
    return record(result, transitionConstant(value.getValue()));

  if (isa<LLVM::TruncOp>(op)) {
    auto state = operands[0]->getValue();
    if (state.isBottom())
      return success();
    return record(result, transitionTrunc(state, resultBitWidth(op)));
  }

  if (isa<LLVM::ShlOp, LLVM::LShrOp>(op)) {
    auto state = operands[0]->getValue();
    if (state.isBottom())
      return success();
    auto width = resultBitWidth(op);
    auto shift = constantShift(op, width);
    if (isa<LLVM::ShlOp>(op))
      return record(result, transitionShl(state, shift, width));
    return record(result, transitionLShr(state, shift, width));
  }

  if (!isa<LLVM::AndOp, LLVM::OrOp, LLVM::AddOp, LLVM::SubOp, LLVM::MulOp, LLVM::UDivOp, LLVM::URemOp, LLVM::XOrOp>(op)) {
    setAllToEntryStates(results);
    return success();
  }

  auto lhs = operands[0]->getValue();
  auto rhs = operands[1]->getValue();
  if (lhs.isBottom() || rhs.isBottom())
    return success();
  // no switches?
  // ⠀⠀⠀⢘⣾⣾⣿⣾⣽⣯⣼⣿⣿⣴⣽⣿⣽⣭⣿⣿⣿⣿⣿⣧
  // ⠀⠀⠀⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
  // ⠀⠀⠠⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
  // ⠀⠀⣰⣯⣾⣿⣿⡼⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿
  // ⠀⠀⠛⠛⠋⠁⣠⡼⡙⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠁
  // ⠀⠀⠀⠤⣶⣾⣿⣿⣿⣦⡈⠉⠉⠉⠙⠻⣿⣿⣿⣿⣿⠿⠁⠀
  // ⠀⠀⠀⠀⠈⠟⠻⢛⣿⣿⣿⣷⣶⣦⣄⠀⠸⣿⣿⣿⠗⠀⠀⠀
  // ⠀⠀⠀⠀⠀⣼⠀⠄⣿⡿⠋⣉⠈⠙⢿⣿⣦⣿⠏⡠⠂⠀⠀⠀
  // ⠀⠀⠀⠀⢰⡌⠀⢠⠏⠇⢸⡇⠐⠀⡄⣿⣿⣃⠈⠀⠀⠀⠀⠀
  // ⠀⠀⠀⠀⠈⣻⣿⢫⢻⡆⡀⠁⠀⢈⣾⣿⠏⠀⠀⠀⠀⠀⠀⠀
  // ⠀⠀⠀⠀⢀⣿⣻⣷⣾⣿⣿⣷⢾⣽⢭⣍⠀⠀⠀⠀⠀⠀⠀⠀
  // ⠀⠀⠀⠀⣼⣿⣿⣿⣿⡿⠈⣹⣾⣿⡞⠐⠁⠀⠀⠀⠁⠀⠀⠀
  // ⠀⠀⠀⠨⣟⣿⢟⣯⣶⣿⣆⣘⣿⡟⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀
  // ⠀⠀⠀⠀⠀⡆⠀⠐⠶⠮⡹⣸⡟⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
  auto width = resultBitWidth(op);
  if (isa<LLVM::AndOp>(op))
    return record(result, transitionAnd(lhs, rhs, width));
  if (isa<LLVM::OrOp>(op))
    return record(result, transitionOr(lhs, rhs, width));
  if (isa<LLVM::AddOp>(op))
    return record(result, transitionAdd(lhs, rhs, width));
  if (isa<LLVM::SubOp>(op))
    return record(result, transitionSub(lhs, rhs, width));
  if (isa<LLVM::MulOp>(op))
    return record(result, transitionMul(lhs, rhs, width));
  if (isa<LLVM::UDivOp>(op))
    return record(result, transitionUDiv(lhs, rhs, width));
  if (isa<LLVM::URemOp>(op))
    return record(result, transitionURem(lhs, rhs, width));
  return record(result, transitionXor(lhs, rhs, width));
}

}
