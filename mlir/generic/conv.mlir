// This works ok
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
  %v2 = constant 0.200000e+00 : f32 

  %input = call @alloc_4d_filled_f32(%c1, %c7, %c7, %c1, %v1) : 
    (index, index, index, index, f32) -> (memref<?x?x?x?xf32>)

  %filter = call @alloc_4d_filled_f32(%c3, %c3, %c1, %c2, %v3) :
    (index, index, index, index, f32) -> (memref<?x?x?x?xf32>) 
  call @set_dim_to_val(%filter, %v2) : (memref<?x?x?x?xf32>, f32) -> () 

  %output = call @alloc_4d_filled_f32(%c1, %c5, %c5, %c2, %v0) : 
    (index, index, index, index, f32) -> (memref<?x?x?x?xf32>)

  call @generic_conv(%filter, %input, %output) : (memref<?x?x?x?xf32>,memref<?x?x?x?xf32>,memref<?x?x?x?xf32>) -> ()
  
  call @print_memref_4d_f32(%input) : (memref<?x?x?x?xf32>) -> ()
  call @print_memref_4d_f32(%filter) : (memref<?x?x?x?xf32>) -> ()
  call @print_memref_4d_f32(%output) : (memref<?x?x?x?xf32>) -> ()

  return
}


// Pretty sure the order is:
//                  N   H   W   O   C   kH  kW      kH, kW, C , O
#map0 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d5, d6, d4, d3)>
//                  N   H   W   O   C   kH  kW      N , H  + kH, W  + kW, C
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d5, d2 + d6, d4)>
//                  N   H   W   O   C   kH  kW      N , H , W , O
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>

#conv_accesses = [
  #map0,
  #map1,
  #map2
]

#conv_trait = {
  indexing_maps = #conv_accesses,
  // iterator_types = ["window", "window", "window", "window", "window", 
  //                   "window", "window"]
  iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", 
                    "window", "window"]
}

func @generic_conv(%filter: memref<?x?x?x?xf32>,
                    %input: memref<?x?x?x?xf32>,
                    %output: memref<?x?x?x?xf32>) -> () {
  linalg.generic #conv_trait 
    ins(%filter, %input : memref<?x?x?x?xf32>, memref<?x?x?x?xf32>) 
    outs(%output : memref<?x?x?x?xf32>) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):  // no predecessors
    %3 = mulf %arg0, %arg1 : f32
    %4 = addf %3, %arg2 : f32
    linalg.yield %4 : f32
  }
  return
}

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

func private @set_dim_to_val(%arg0: memref<?x?x?x?xf32>, %val: f32) -> () {
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c2 = constant 2 : index
    %c3 = constant 3 : index
    %0 = dim %arg0, %c0 : memref<?x?x?x?xf32>
    %1 = dim %arg0, %c1 : memref<?x?x?x?xf32>
    %2 = dim %arg0, %c2 : memref<?x?x?x?xf32>
    %3 = dim %arg0, %c3 : memref<?x?x?x?xf32>
  affine.for %arg5 = 0 to %0 {
    affine.for %arg6 = 0 to %1 {
      affine.for %arg7 = 0 to %2 {
        affine.for %arg8 = 0 to 1 {
          affine.store %val, %arg0[%arg5, %arg6, %arg7, %arg8] : memref<?x?x?x?xf32>
        }
      }
    }
  }
  return
}


func private @print_memref_f32(memref<*xf32>) attributes { llvm.emit_c_interface }
func private @print_memref_4d_f32(memref<?x?x?x?xf32>) attributes { llvm.emit_c_interface }
func private @print_memref_3d_f32(memref<?x?x?xf32>) attributes { llvm.emit_c_interface }