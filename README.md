# NICO's collection of MLIR examples

This project contains some examples of how to compile and run mlir files with
`linalg` and `affine` dialect operations. Some examples are inspired by tests
present on the MLIR project others were handcoded to implement specific
algorithms.

Before running the examples you must:

- Have llvm with mlir compiled somewhere. 
- Find the folders paths to `mlir-opt` and `libmlir_runner_utils.so`
  - `<llvm-project-build-folder>/bin`
  - `<llvm-project-build-folder>/lib`
- Edit the folder paths in the script file: `<this-project>/mlir/run.sh`

Tu run a mlir program that has a `main` function execute:

./run.sh <path-to-mlir-file>

NOTE: Some intermediate files will be generated on the folder that you run the
`run.sh`. Take a look at them if you are interested on intermediate steps.

NOTE2: Some mlir files have a comment on top stating if the files cannot be
compiled and why there is a problem.

NOTE3: Most of the code was developed using llvm-project commit
`da59c2e4dc1c7b502cc2098247b2ee38ac297e0c`. MLIR is moving fast, so there may be
files that are not working anymore, or others that are now fully supported.