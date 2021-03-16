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

  %inA = call @alloc_1d_filled_f32(%c5, %v1) : 
    (index, f32) -> (memref<?xf32>)

  %out = call @alloc_1d_filled_f32(%c1, %v1) : 
    (index, f32) -> (memref<?xf32>)

  call @print_memref_1d_f32(%out) : (memref<?xf32>) -> ()
  call @print_memref_1d_f32(%inA) : (memref<?xf32>) -> ()

  call @reduction(%inA, %out) : (memref<?xf32>,memref<?xf32>) -> ()
  call @print_memref_1d_f32(%out) : (memref<?xf32>) -> ()

  return
}

#id_1d = affine_map<(N) -> (N)>
#id_reduction = affine_map<(N) -> (0)>
#reduction_trait = {
  args_in = 2,
  args_out = 1,
  indexing_maps = [#id_1d, #id_reduction],
  iterator_types = ["reduction"]
}

// output=sum_i{a[i]}
func @reduction(%arg0: memref<?xf32>, %arg2: memref<?xf32>) {
  linalg.generic #reduction_trait
    ins(%arg0 : memref<?xf32>)
    outs(%arg2 : memref<?xf32>) {
  ^bb0(%arg4: f32, %arg6: f32):
    %4 = addf %arg4, %arg6 : f32
    linalg.yield %4 : f32
  }
  return
}

func private @alloc_1d_filled_f32(%s1 : index, %f : f32) -> memref<?xf32> {
  %buf = memref.alloc(%s1) : memref<?xf32>
  linalg.fill(%buf, %f) : memref<?xf32>, f32
  return %buf : memref<?xf32>
}

func private @print_memref_1d_f32(memref<?xf32>) attributes { llvm.emit_c_interface }