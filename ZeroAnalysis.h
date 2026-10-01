
#ifndef ZERO_ANALYSIS_H
#define ZERO_ANALYSIS_H

#include "ZeroDomain.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"

namespace zero {

using ZeroLattice = mlir::dataflow::Lattice<IntReprState>;

class ZeroAnalysis : public mlir::dataflow::SparseForwardDataFlowAnalysis<ZeroLattice> {
public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  mlir::LogicalResult visitOperation(mlir::Operation *op, llvm::ArrayRef<const ZeroLattice *> operands, llvm::ArrayRef<ZeroLattice *> results) override;

  void setToEntryState(ZeroLattice *lattice) override;

  mlir::LogicalResult record(ZeroLattice *lattice, IntReprState state);
};

} // namespace zero

#endif
