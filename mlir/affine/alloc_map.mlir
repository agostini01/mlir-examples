#colmajor_map = affine_map<(i,j)->(j,i)>

func @main() {

  %c3 = constant 3 : index
  %c4 = constant 4 : index
  %v1 = constant 1.0000000e+00 : f32

  // %rowmajor2d = memref.alloc(%c3, %c4) : memref<?x?xf32, #colmajor_map>
  // linalg.fill(%rowmajor2d, %v1) : memref<?x?xf32, #colmajor_map>, f32
  // call @print_memref_2d_rowmaj_f32(%rowmajor2d) : (memref<?x?xf32, #colmajor_map>) -> ()

  return
}

// func @strided_memref(%ind: index) -> (memref<32x64xf32, affine_map<(i, j)[M] -> (32 + M * i + j)>>) {
//   %0 = memref.alloc()[%ind] : memref<32x64xf32, affine_map<(i, j)[M] -> (32 + M * i + j)>>
//   std.return %0
// }

func @strided_memref(%ind: index) {
  %0 = memref.alloc()[%ind] : memref<32x64xf32, affine_map<(i, j)[M] -> (32 + M * i + j)>>
  std.return
}

// func private @print_memref_2d_f32(memref<?x?xf32>) attributes { llvm.emit_c_interface }
// func private @print_memref_2d_rowmaj_f32(memref<?x?xf32, #colmajor_map>) attributes { llvm.emit_c_interface }