// #!/bin/bash

// # USAGE
// #   Modify LIB_DIR to point to where libmlir_runner_utils.so is available,
// #   then execute:
// #   ./run.sh <path-to-mlir-file>

// INPUT=$1
// # LIB_DIR=/home/nico/Development/llvm-project/build-mlir-standalone/lib
// LIB_DIR=/working_dir/llvm-project/build-mlir-standalone/lib


// set -e
// set -x
// # mlir-opt $INPUT -linalg-generalize-named-ops -o tmp.mlir
// mlir-opt $INPUT -o tmp.mlir
// mlir-opt tmp.mlir -linalg-bufferize -convert-linalg-to-affine-loops -o tmp1.mlir
// # mlir-opt tmp.mlir -linalg-bufferize -convert-linalg-to-llvm -o tmp1.mlir
// mlir-opt tmp1.mlir --func-bufferize --tensor-constant-bufferize --tensor-bufferize --finalizing-bufferize -o tmp2.mlir
// mlir-opt tmp2.mlir -lower-affine  -convert-scf-to-std   -convert-std-to-llvm -o tmp3.mlir
// mlir-cpu-runner tmp3.mlir -e main -entry-point-result=void -shared-libs=$LIB_DIR/libmlir_runner_utils.so,$LIB_DIR/libmlir_c_runner_utils.so



module  {
  func @main() {
    %c1 = constant 1 : index
    %c2 = constant 2 : index
    %c3 = constant 3 : index
    %c5 = constant 5 : index
    %cst = constant 1.000000e+00 : f32
    %cst_0 = constant 0.000000e+00 : f32
    %cst_1 = constant 1.000000e-01 : f32
    %input = call @alloc_4d_filled_f32(%c2, %c5, %c5, %c1, %cst) : (index, index, index, index, f32) -> memref<?x?x?x?xf32>
    %filter = call @alloc_4d_filled_f32(%c3, %c3, %c1, %c1, %cst_1) : (index, index, index, index, f32) -> memref<?x?x?x?xf32>
    %output = call @alloc_4d_filled_f32(%c2, %c5, %c5, %c1, %cst_0) : (index, index, index, index, f32) -> memref<?x?x?x?xf32>

    // Conv named op
    // Per my understanding arguments are: 
    //    filter(kH,kW,C,O), input(N,H,W,C), output(N,H,W,O)
    linalg.conv(%filter, %input, %output) {
        dilations = [1, 1], 
        padding = dense<1> : tensor<2x2xi64>, 
        strides = [1, 1]
      } : memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, memref<?x?x?x?xf32>

    call @print_memref_4d_f32(%input) : (memref<?x?x?x?xf32>) -> ()
    call @print_memref_4d_f32(%filter) : (memref<?x?x?x?xf32>) -> ()
    call @print_memref_4d_f32(%output) : (memref<?x?x?x?xf32>) -> ()
    return
  }

  // Affine implementation of linalg.fill
  func private @alloc_4d_filled_f32(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: f32) -> memref<?x?x?x?xf32> {
    %0 = memref.alloc(%arg0, %arg1, %arg2, %arg3) : memref<?x?x?x?xf32>
    affine.for %arg5 = 0 to %arg0 {
      affine.for %arg6 = 0 to %arg1 {
        affine.for %arg7 = 0 to %arg2 {
          affine.for %arg8 = 0 to %arg3 {
            affine.store %arg4, %0[%arg5, %arg6, %arg7, %arg8] : memref<?x?x?x?xf32>
          }
        }
      }
    }
    return %0 : memref<?x?x?x?xf32>
  }
  func private @print_memref_4d_f32(memref<?x?x?x?xf32>) attributes {llvm.emit_c_interface}
}

