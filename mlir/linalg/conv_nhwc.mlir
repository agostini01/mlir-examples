// RUN:   mlir-opt %s -convert-async-to-llvm                                   \
// RUN:               -convert-linalg-to-loops                                 \
// RUN:               -convert-linalg-to-llvm                                  \
// RUN:               -convert-std-to-llvm                                     \
// RUN: | mlir-cpu-runner                                                      \
// RUN:     -e main -entry-point-result=void -O0                               \
// RUN:     -shared-libs=%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext  \
// RUN:     -shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext    

func @main() {
  %c0 = constant 0 : index 
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %c3 = constant 3 : index
  %c4 = constant 4 : index
  %c5 = constant 5 : index

  %v1 = constant 1.0000000e+00 : f32
  %v0 = constant 0.00e+00 : f32
  %v3 = constant 0.100000e+00 : f32 

  //%input = call @alloc_4d_filled_f32(%c1, %c5, %c5, %c2, %v1) : 
  %input = call @alloc_4d_increasing_f32(%c1, %c5, %c5, %c2, %v1) : 
    (index, index, index, index, f32) -> (memref<?x?x?x?xf32>)

  %filter = call @alloc_4d_filled_f32(%c1, %c3, %c3, %c2, %v3) :
  // %filter = call @alloc_4d_filled_f32(%c1, %c1, %c3, %c3, %v3) :
  (index, index, index, index, f32) -> (memref<?x?x?x?xf32>) 

  %output = call @alloc_4d_filled_f32(%c1, %c3, %c3, %c1, %v0) : 
  // %output = call @alloc_4d_filled_f32(%c1, %c1, %c3, %c3, %v0) : 
    (index, index, index, index, f32) -> (memref<?x?x?x?xf32>)


  // In this call convolution happens without padding
  //  the input has shape: batch_size, (H, W), in_channels
  //  the filter has shape: out_channels, (kH, kW), in_channels
  //  the output has shape: batch_size, (H, W), out_channels
  linalg.conv_2d_nhwc ins (%input, %filter : memref<?x?x?x?xf32>, memref<?x?x?x?xf32>)
                      outs (%output: memref<?x?x?x?xf32>)


  call @print_memref_4d_f32(%input) : (memref<?x?x?x?xf32>) -> ()
  call @print_memref_4d_f32(%filter) : (memref<?x?x?x?xf32>) -> ()
  call @print_memref_4d_f32(%output) : (memref<?x?x?x?xf32>) -> ()

  return
}

func private @alloc_4d_filled_f32(%s1 : index, %s2 : index, %s3 : index, %s4 : index, %f : f32) -> memref<?x?x?x?xf32> {
  %buf = memref.alloc(%s1, %s2, %s3, %s4) : memref<?x?x?x?xf32> 
  linalg.fill(%buf, %f) : memref<?x?x?x?xf32>, f32
  return %buf : memref<?x?x?x?xf32>
}

/// Allocates 4D data structure with the values based on the index of the
///   innermost dimension
func private @alloc_4d_increasing_f32(%s1 : index, %s2 : index, %s3 : index, %s4 : index, %f : f32) -> memref<?x?x?x?xf32> {
  %buf = memref.alloc(%s1, %s2, %s3, %s4) : memref<?x?x?x?xf32> 
  affine.for %i = 0 to %s1 {
    affine.for %j = 0 to %s2 {
      affine.for %k = 0 to %s3 {
        affine.for %w = 0 to %s4 {
          %v_tmp = index_cast %w: index to i32
          %v = sitofp %v_tmp: i32 to f32
          memref.store %v, %buf[%i, %j, %k, %w] : memref<?x?x?x?xf32>
        }
      }
    }
  }
  return %buf : memref<?x?x?x?xf32>
}


func private @print_memref_f32(memref<*xf32>) attributes { llvm.emit_c_interface }
func private @print_memref_4d_f32(memref<?x?x?x?xf32>) attributes { llvm.emit_c_interface }