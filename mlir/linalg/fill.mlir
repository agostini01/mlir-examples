// https://github.com/llvm/llvm-project/blob/e9c11c1934f90725673f8a808fe6fcc390224a87/mlir/lib/Dialect/Linalg/Transforms/Generalization.cpp#L119

// This does not work, and breaks with an Aborted (core dumped) error
// mlir-opt -linalg-generalize-named-ops -convert-linalg-to-affine-loops %s

// Converting directly to affine loops works
// mlir-opt -convert-linalg-to-affine-loops %s


func private @alloc_4d_filled_f32(%s1 : index, %s2 : index, %s3 : index, %s4 : index, %f : f32) -> memref<?x?x?x?xf32> {
  %buf = alloc(%s1, %s2, %s3, %s4) : memref<?x?x?x?xf32> 
  linalg.fill(%buf, %f) : memref<?x?x?x?xf32>, f32
  return %buf : memref<?x?x?x?xf32>
}