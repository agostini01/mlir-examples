func @main() {

  %c0 = constant 0 : index 
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %c3 = constant 3 : index
  %c4 = constant 4 : index
  %c5 = constant 5 : index

  %v0 = constant 0.0e+00 : f32
  %v1 = constant 1.00e+00 : f32
  %v2 = constant 0.50e+00 : f32
  %v3 = constant 0.10e+00 : f32 

  %inA = call @alloc_2d_filled_f32(%c5, %c2, %v1) : 
  // %inA = call @alloc_2d_increasing_f32(%c5, %c2, %v1) : 
    (index, index, f32) -> (memref<?x?xf32>)

  // %inB = call @alloc_2d_filled_f32(%c2, %c4, %v1) : 
  %inB = call @alloc_2d_increasing_f32(%c2, %c4, %v1) : 
    (index, index, f32) -> (memref<?x?xf32>)

  %out = call @alloc_2d_filled_f32(%c5, %c4, %v0) : 
    (index, index, f32) -> (memref<?x?xf32>)

  call @my_matmul(%inA, %inB, %out) : (memref<?x?xf32>,memref<?x?xf32>,memref<?x?xf32>) -> ()
  
  call @print_memref_2d_f32(%inA) : (memref<?x?xf32>) -> ()
  call @print_memref_2d_f32(%inB) : (memref<?x?xf32>) -> ()
  call @print_memref_2d_f32(%out) : (memref<?x?xf32>) -> ()

  return
}

#matmul_accesses = [
  affine_map<(k,m,n)->(m,k)>,
  affine_map<(k,m,n)->(k,n)>,
  affine_map<(k,m,n)->(m,n)>
]

#matmul_trait = {
  indexing_maps = #matmul_accesses,
  iterator_types = ["reduction", "parallel", "parallel"]
}

func @my_matmul(%A: memref<?x?xf32>, %B: memref<?x?xf32>, 
                %C: memref<?x?xf32>){
  linalg.generic #matmul_trait
    ins(%A, %B : memref<?x?xf32>, memref<?x?xf32>)
    outs(%C : memref<?x?xf32>) {
  ^bb0(%a: f32, %b: f32, %c: f32):
    %tmp = mulf %a,%b:f32
    %acc = addf %tmp,%c:f32
    linalg.yield %acc : f32
  }
  return
}

func private @alloc_2d_filled_f32(%arg0: index, %arg1: index, %arg2: f32) -> memref<?x?xf32> {
  %0 = memref.alloc(%arg0, %arg1) : memref<?x?xf32>
  affine.for %arg3 = 0 to %arg0 {
    affine.for %arg4 = 0 to %arg1 {
      affine.store %arg2, %0[%arg3, %arg4] : memref<?x?xf32>
    }
  }
  return %0 : memref<?x?xf32>
}

func private @alloc_2d_increasing_f32(%s1 : index, %s2 : index, %f : f32) -> memref<?x?xf32> {
  %buf = memref.alloc(%s1, %s2) : memref<?x?xf32> 
      affine.for %k = 0 to %s1 {
        affine.for %w = 0 to %s2 {
          %v_tmp = index_cast %w: index to i32
          %v = sitofp %v_tmp: i32 to f32
          memref.store %v, %buf[%k, %w] : memref<?x?xf32>
        }
      }
  return %buf : memref<?x?xf32>
}

func private @print_memref_2d_f32(memref<?x?xf32>) attributes { llvm.emit_c_interface }