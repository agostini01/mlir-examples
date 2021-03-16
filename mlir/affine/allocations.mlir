#colmajor_map = affine_map<(i,j)->(j,i)>

func @main() {

  %c0 = constant 0 : index 
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %c3 = constant 3 : index
  %c4 = constant 4 : index
  %c5 = constant 5 : index

  %v1 = constant 1.0000000e+00 : f32
  %v0 = constant 0.50e+00 : f32
  %v3 = constant 0.100000e+00 : f32 

  %colmajor2d = memref.alloc(%c3, %c4) : memref<?x?xf32>
  linalg.fill(%colmajor2d, %v1) : memref<?x?xf32>, f32
  call @print_memref_2d_f32(%colmajor2d) : (memref<?x?xf32>) -> ()
  
  %colmajor3d = memref.alloc(%c3, %c4, %c5) : memref<?x?x?xf32>
  linalg.fill(%colmajor3d, %v1) : memref<?x?x?xf32>, f32
  call @print_memref_3d_f32(%colmajor3d) : (memref<?x?x?xf32>) -> ()
  
  return
}


func private @print_memref_2d_f32(memref<?x?xf32>) attributes { llvm.emit_c_interface }
func private @print_memref_3d_f32(memref<?x?x?xf32>) attributes { llvm.emit_c_interface }