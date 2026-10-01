# MLIR out-of-tree dataflow analysis template

A starting point for writing an MLIR dataflow analysis as a loadable `mlir-opt`
plugin, with no LLVM source tree required and nothing to patch upstream.

The included analysis, `zero-analysis`, decides which integer values in the LLVM
dialect are known to be zero. It has exactly two transfer rules and is meant to
be replaced: the point is the scaffolding around it.

## Building

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

That is the whole procedure on Linux, macOS, and WSL2. There is no platform
flag to set and no path to edit. `CMakeLists.txt` finds MLIR by asking
whichever `llvm-config` is on your `PATH` where its CMake package lives, so if
`mlir-opt` runs, the build should configure.

To build against a specific MLIR instead:

```sh
cmake -S . -B build -DMLIR_DIR=/path/to/prefix/lib/cmake/mlir
```

You need an LLVM built with MLIR enabled and plugins enabled
(`-DLLVM_ENABLE_PROJECTS=mlir -DLLVM_ENABLE_PLUGINS=ON`; both are ordinary on
Linux and macOS). Distribution packages work: on Debian and Ubuntu that is
`libmlir-dev` alongside `llvm-dev`. On macOS, Homebrew's `llvm` is the easy
route if it ships `mlir-opt` for your version; otherwise build LLVM yourself.
The configure step diagnoses the cases it can detect — no MLIR
found, plugins disabled in the host LLVM, or an `mlir-opt` on `PATH` whose
version does not match what you are building against.

## Running

```sh
./run.sh input.mlir
```

`run.sh` locates the plugin whatever it is called on your platform and puts the
annotated listing on stdout. Or invoke `mlir-opt` yourself:

```sh
mlir-opt --load-pass-plugin=build/ZeroAnalysis.so \
         --pass-pipeline='builtin.module(zero-analysis)' \
         input.mlir -o /dev/null
```

using `build/ZeroAnalysis.dylib` on macOS. The pass leaves the IR unchanged and
writes it to stdout as usual; the annotated view goes to stderr, so the two
streams can be redirected independently. Annotations are comments, so the
annotated listing is still valid MLIR. Values at top or bottom are left
unannotated, so that what prints is exactly what was proved.

Get input in the LLVM dialect from C with:

```sh
clang -S -emit-llvm -o - input.c | mlir-translate --import-llvm
```

## What is where

Two files hold the analysis; the rest is reusable scaffolding.

| File | |
|---|---|
| `ZeroDomain.h` | The abstract domain: the lattice elements and their join. |
| `ZeroAnalysis.cpp` | The transfer function: two rules, plus a default. |
| `ZeroAnalysis.h` | Ties the domain to MLIR's sparse forward analysis. |
| `Annotate.{h,cpp}` | Prints IR with a comment on each value. Domain-agnostic. |
| `Plugin.cpp` | The pass, the solver setup, and the `mlir-opt` entry point. |
| `cmake/RunTest.cmake` | The test runner. |

To build a different analysis, replace `ZeroDomain.h` and the transfer
functions in `ZeroAnalysis.cpp`. To rename the whole thing, rename the files,
the `zero` namespace, and the three places `ZeroAnalysis` and `zero-analysis`
appear in `CMakeLists.txt` and `Plugin.cpp`.

## Tests

`test/zero.mlir` exercises every transfer rule. `test/zero.expected` lists
facts that must appear in the output, and — with a leading `!` — facts that
must not. The negative checks are the ones that matter: an unsound transfer
function still produces plausible-looking output, and only a test that pins
down what the analysis must *not* claim will catch it.

Note that MLIR's printer renumbers SSA values, so the checks are written
against operation text rather than the names in `zero.mlir`. After adding or
reordering operations, regenerate with `./run.sh test/zero.mlir`.

## Notes on portability

Most of the platform-specific knowledge lives in `CMakeLists.txt`, next to the
code it affects. The parts worth knowing about:

**The plugin's file name differs.** It is `ZeroAnalysis.dylib` on macOS and
`ZeroAnalysis.so` on Linux and WSL2. Nothing in this project spells that out:
CMake is asked via `$<TARGET_FILE:ZeroAnalysis>`, and `run.sh` probes for both.

**Linking a plugin on macOS needs special flags.** The plugin deliberately
leaves its MLIR symbols undefined, to be resolved from the `mlir-opt` process
that loads it. On macOS that requires `-undefined dynamic_lookup`, which
`include(HandleLLVMOptions)` supplies. The same include also matches LLVM's
RTTI and exception settings, which differ between distribution packages and
local builds and cause link errors or silent ODR violations when they are
wrong. That is also why `project()` enables C: `HandleLLVMOptions` probes flags
with the C compiler and fails if none is configured.

**A plugin only loads into the LLVM it was built against.** The version is
recorded at compile time and checked at load time, so a mismatch is a clear
error rather than a crash. The configure step warns about it earlier still, by
comparing against the `mlir-opt` it finds.

**The test suite needs no shell.** `cmake/RunTest.cmake` is a CMake script
rather than a shell script, so `ctest` depends on nothing the build did not
already require.

**Under WSL2, build on the Linux filesystem.** A tree under `/mnt/c` is
slow enough to be noticeable and does not reliably carry execute bits.
`.gitattributes` forces LF endings, which keeps `run.sh` working when a
repository is cloned by a Windows git and built inside WSL2.

## How the analysis works

`Plugin.cpp` loads three analyses into one solver. `DeadCodeAnalysis` supplies
reachability — without it the solver must assume every branch is taken — and
`SparseConstantPropagation` resolves branch conditions on its behalf. These are
prerequisites for a precise result, not optional extras. `ZeroAnalysis` then
propagates zeroness through operations and block arguments until the solver
reaches a fixed point, which is when the pass queries it.

The transfer function has two rules, one of each kind an analysis needs:

- **Constants** are zero or nonzero as written. This is the only rule that does
  not consult its operands, and without some rule of this kind there would be
  no facts to propagate at all.
- **`x & y` is zero if either operand is zero**, because a zero operand clears
  every bit. Note what this does not say: two nonzero operands prove nothing,
  since `1 & 2` is `0`.

Everything else is unknown. That is always sound, just imprecise — `llvm.or`
and `llvm.add` are left unhandled in the test file precisely so their output
shows what "unknown" looks like. Adding a third rule should be a matter of
adding a third `if`.

Values reaching the analysis from outside — function arguments, and results of
any operation without a rule — start at top. The domain's fourth element,
bottom, means "not yet proved reachable"; the solver starts everything there
and raises it as facts arrive, which is what makes the fixed-point iteration
terminate.

The analysis is intraprocedural. It does not refine facts on branch conditions,
so a value tested against zero is not known nonzero on the taken edge — that,
and a rule for `llvm.or`, are the natural first extensions.

## Find fun facts
this will find a few locations in sqlite where f32 works as an int representation but i32 doesnt
```sh
clang -S -emit-llvm -o - input.c |
  mlir-translate --import-llvm |
  mlir-opt --load-pass-plugin=build/ZeroAnalysis.dylib \
    --pass-pipeline='builtin.module(zero-analysis)' -o /dev/null 2>&1 |
  grep -v 'i32' | grep 'i64, f32' 
```
so i guess they could have saved 4 bytes somewhere... maybe