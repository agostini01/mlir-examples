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

  // %inA = call @alloc_2d_filled_f32(%c5, %c5, %v1) : 
  %inA = call @alloc_2d_identity_f32(%c5, %c5, %v1) : 
    (index, index, f32) -> (memref<?x?xf32>)

  %inB = call @alloc_2d_filled_f32(%c5, %c5, %v1) : 
    (index, index, f32) -> (memref<?x?xf32>)

  %out = call @alloc_2d_filled_f32(%c5, %c5, %v0) : 
    (index, index, f32) -> (memref<?x?xf32>)

  call @pointwise(%inA, %inB, %out) : (memref<?x?xf32>,memref<?x?xf32>,memref<?x?xf32>) -> ()
  call @print_memref_2d_f32(%inA) : (memref<?x?xf32>) -> ()
  call @print_memref_2d_f32(%inB) : (memref<?x?xf32>) -> ()
  call @print_memref_2d_f32(%out) : (memref<?x?xf32>) -> ()

  return
}

#id_2d = affine_map<(i, j) -> (i, j)>
#pointwise_2d_trait = {
  args_in = 2,
  args_out = 1,
  indexing_maps = [#id_2d, #id_2d, #id_2d],
  iterator_types = ["parallel", "parallel"]
}

func @pointwise(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) {
  linalg.generic #pointwise_2d_trait
    ins(%arg0, %arg1 : memref<?x?xf32>, memref<?x?xf32>)
    outs(%arg2 : memref<?x?xf32>) {
  ^bb0(%arg4: f32, %arg5: f32, %arg6: f32):   // no predecessors
    %4 = addf %arg4, %arg5 : f32
    %5 = addf %4, %arg6 : f32
    linalg.yield %5 : f32
  }
  return
}

#set_identity = affine_set<(d0, d1)
  : (d0-d1 == 0)>


/// Allocates a data structure with value %f on its main diagonal
func private @alloc_2d_identity_f32(%s1 : index, %s2 : index, %f : f32) -> memref<?x?xf32> {
  %buf = memref.alloc(%s1, %s2) : memref<?x?xf32> 
  %c0 = constant 0.0 : f32 
  %c1 = constant 1.0 : f32
  affine.for %i = 0 to %s1 {
    affine.for %j = 0 to %s2 {
      affine.if #set_identity (%i,%j){
        memref.store %f, %buf[%i, %j] : memref<?x?xf32>
      }
      else {
        memref.store %c0, %buf[%i, %j] : memref<?x?xf32>
      }
    }
  }
  return %buf : memref<?x?xf32>
}

func private @alloc_2d_filled_f32(%s1 : index, %s2 : index, %f : f32) -> memref<?x?xf32> {
  %buf = memref.alloc(%s1, %s2) : memref<?x?xf32>
  linalg.fill(%buf, %f) : memref<?x?xf32>, f32
  return %buf : memref<?x?xf32>
}

func private @print_memref_2d_f32(memref<?x?xf32>) attributes { llvm.emit_c_interface }