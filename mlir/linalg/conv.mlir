func @main() {
  %c0 = constant 0 : index 
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %c3 = constant 3 : index
  %c4 = constant 4 : index
  %c5 = constant 5 : index
  %c6 = constant 6 : index
  %c7 = constant 7 : index

  %v1 = constant 1.0000000e+00 : f32
  %v0 = constant 0.00e+00 : f32
  %v3 = constant 0.100000e+00 : f32 

  %input = call @alloc_4d_filled_f32(%c1, %c7, %c7, %c1, %v1) : 
    (index, index, index, index, f32) -> (memref<?x?x?x?xf32>)

  %filter = call @alloc_4d_filled_f32(%c3, %c3, %c1, %c2, %v3) :
  // %filter = call @alloc_4d_filled_f32(%c1, %c1, %c3, %c3, %v3) :
  (index, index, index, index, f32) -> (memref<?x?x?x?xf32>) 

  %output = call @alloc_4d_filled_f32(%c1, %c5, %c5, %c2, %v0) : 
    (index, index, index, index, f32) -> (memref<?x?x?x?xf32>)

  // Conv named op
  // Per my understanding arguments are: 
  //    filter(kH,kW,C,O), input(N,H,W,C), output(N,H,W,O)
  linalg.conv(%filter, %input, %output) {
      dilations = [1,1],
      strides = [1,1],
      padding = dense<0> : tensor<2x2xi64> 
    } : memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, memref<?x?x?x?xf32>

  call @print_memref_4d_f32(%input) : (memref<?x?x?x?xf32>) -> ()
  call @print_memref_4d_f32(%filter) : (memref<?x?x?x?xf32>) -> ()
  call @print_memref_4d_f32(%output) : (memref<?x?x?x?xf32>) -> ()

  return
}

// func private @alloc_4d_filled_f32(%s1 : index, %s2 : index, %s3 : index, %s4 : index, %f : f32) -> memref<?x?x?x?xf32> {
//   %buf = alloc(%s1, %s2, %s3, %s4) : memref<?x?x?x?xf32> 
//   linalg.fill(%buf, %f) : memref<?x?x?x?xf32>, f32
//   return %buf : memref<?x?x?x?xf32>
// }

func private @alloc_4d_filled_f32(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: f32) -> memref<?x?x?x?xf32> {
  %0 = alloc(%arg0, %arg1, %arg2, %arg3) : memref<?x?x?x?xf32>
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


func private @print_memref_f32(memref<*xf32>) attributes { llvm.emit_c_interface }
func private @print_memref_4d_f32(memref<?x?x?x?xf32>) attributes { llvm.emit_c_interface }
func private @print_memref_3d_f32(memref<?x?x?xf32>) attributes { llvm.emit_c_interface }