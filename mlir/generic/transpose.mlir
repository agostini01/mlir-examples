func @main() {

  %c0 = constant 0 : index 
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %c3 = constant 3 : index
  %c4 = constant 4 : index
  %c5 = constant 5 : index

  %v1 = constant 1.0e+00 : f32

  %in = memref.alloc(%c4, %c4) : memref<?x?xf32>
  %out = memref.alloc(%c4, %c4) : memref<?x?xf32>

  call @create_identity(%in) : (memref<?x?xf32>) -> ()
  memref.store %v1, %in[%c3, %c0] : memref<?x?xf32>
  call @print_memref_2d_f32(%in) : (memref<?x?xf32>) -> ()
  
  call @transpose(%in, %out) : (memref<?x?xf32>, memref<?x?xf32>) -> ()
  call @print_memref_2d_f32(%out) : (memref<?x?xf32>) -> ()
  call @transpose2(%out) : (memref<?x?xf32>) -> ()
  call @print_memref_2d_f32(%out) : (memref<?x?xf32>) -> ()
  call @transpose3(%out) : (memref<?x?xf32>) -> ()
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


// Populates matrix with Identity values
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

#transpose_accesses = [
  affine_map<(H,W)->(H,W)>,
  affine_map<(H,W)->(W,H)>
]

#trans_trait = {
  indexing_maps = #transpose_accesses,
  iterator_types = ["parallel","parallel"]
}

// Transpose Matrix `in` storing into `out`
func @transpose(%in: memref<?x?xf32>, %out: memref<?x?xf32>){

  // Transpose
  linalg.generic #trans_trait
    ins(%in : memref<?x?xf32>)
    outs(%out : memref<?x?xf32>) {
    ^bb0(%a: f32, %b:f32):
        linalg.yield %a : f32
  }
  return
}


// Transpose Matrix `out` inplace
func @transpose2(%out: memref<?x?xf32>){

  // Transpose inplace
  linalg.indexed_generic #trans_trait
    outs(%out, %out : memref<?x?xf32>, memref<?x?xf32>) {
    ^bb0(%i: index, %j: index, %a: f32, %b:f32):
      // Are the indexes on the diagonal/upper triangle?
      %cond = cmpi ugt, %i, %j : index
      %r1, %r2 = scf.if %cond -> (f32,f32) {
           scf.yield %b,%a : f32, f32
      } else {
           scf.yield %a,%b : f32, f32
      }
      linalg.yield %r1, %r2 : f32 ,f32
  }
  return
}

// Transpose Matrix `out` inplace
#set0 = affine_set<(d0,d1) : (d0-d1>=0)>
func @transpose3(%out: memref<?x?xf32>){

  // Transpose inplace
  linalg.indexed_generic #trans_trait
    outs(%out, %out : memref<?x?xf32>, memref<?x?xf32>) {
    ^bb0(%i: index, %j: index, %a: f32, %b:f32):
    // https://mlir.llvm.org/docs/Dialects/Affine/#restrictions-on-dimensions-and-symbols
    // This does not work: linalg indeces cannot be used
    // as inputs to the affine set...
    // %r1, %r2 = affine.if #set0(%i,%j) -> (f32,f32) {
    //      affine.yield %b,%a : f32, f32
    // } else {
    //      affine.yield %a,%b : f32, f32
    // }

    // Instead use this
    %1 = cmpi sge, %i, %j : index
    %r1 = select %1, %b, %a : f32
    %r2 = select %1, %a, %b : f32
    linalg.yield %r1, %r2 : f32 ,f32
  }
  return
}



func private @print_memref_2d_f32(memref<?x?xf32>) attributes { llvm.emit_c_interface }