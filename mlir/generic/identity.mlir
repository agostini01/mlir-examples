func @main() {

  %c0 = constant 0 : index 
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %c3 = constant 3 : index
  %c4 = constant 4 : index
  %c5 = constant 5 : index

  %out = alloc(%c4, %c4) : memref<?x?xf32>

  call @create_identity(%out) : (memref<?x?xf32>) -> ()
  call @print_memref_2d_f32(%out) : (memref<?x?xf32>) -> ()

  return
}

#std_accesses = [
  affine_map<(H,W)->(H,W)>
]

#std_trait = {
  indexing_maps = #std_accesses,
  iterator_types = ["parallel","parallel"]
}

#identity_accesses = [
  affine_map<(H)->(H,H)>
]

#identity_trait = {
  indexing_maps = #identity_accesses,
  iterator_types = ["parallel"]
}

func @create_identity(%C: memref<?x?xf32>){

  // Set zero
  linalg.generic #std_trait
    outs(%C : memref<?x?xf32>) {
    ^bb0(%a: f32):
        %v1 = constant 0.000e+00 : f32
        linalg.yield %v1 : f32
  }

  // Set identity
  linalg.generic #identity_trait
    outs(%C : memref<?x?xf32>) {
    ^bb0(%a: f32):
        %v1 = constant 1.0000000e+00 : f32
        linalg.yield %v1 : f32
  }
  return
}

func private @print_memref_2d_f32(memref<?x?xf32>) attributes { llvm.emit_c_interface }