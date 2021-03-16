#map0 = affine_map<(d0, d1) -> (d0 + d1 - 1)>
#map1 = affine_map<(d0) -> (d0, 0)>
module  {
  func @main() {
    %c5 = constant 5 : index
    %cst = constant 1.000000e+00 : f32
    %cst_fil_val = constant 1.000000e-01 : f32
    %c3 = constant 3 : index
    %cst_zero = constant 0.000000e+00 : f32
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c2 = constant 2 : index
    %in = call @alloc_4d_filled_f32(%c1, %c5, %c5, %c1, %cst) : (index, index, index, index, f32) -> memref<?x?x?x?xf32>
    %fil = call @alloc_4d_filled_f32(%c3, %c3, %c1, %c1, %cst_fil_val) : (index, index, index, index, f32) -> memref<?x?x?x?xf32>
    %out = call @alloc_4d_filled_f32(%c1, %c5, %c5, %c1, %cst_zero) : (index, index, index, index, f32) -> memref<?x?x?x?xf32>
    %kH = memref.dim %fil, %c0 : memref<?x?x?x?xf32>
    %kW = memref.dim %fil, %c1 : memref<?x?x?x?xf32>
    %C = memref.dim %fil, %c2 : memref<?x?x?x?xf32>
    %O = memref.dim %fil, %c3 : memref<?x?x?x?xf32>
    %N = memref.dim %in, %c0 : memref<?x?x?x?xf32>
    %H = memref.dim %out, %c1 : memref<?x?x?x?xf32>
    %W = memref.dim %out, %c2 : memref<?x?x?x?xf32>
    affine.for %n = 0 to %N {
      affine.for %i_H = 0 to %H {
        affine.for %j_W = 0 to %W {
          affine.for %channels_out = 0 to %O {
            affine.for %channels_in = 0 to %C {
              affine.for %i_kH = 0 to %kH {
                affine.for %j_kW = 0 to %kW {
                  %i_input = affine.apply #map0(%i_H, %i_kH)
                  %j_input = affine.apply #map0(%j_W, %j_kW)
                  %i_cliped = affine.max #map1(%i_input)
                  %j_cliped = affine.max #map1(%j_input)
                  %dim_H = memref.dim %in, %c1 : memref<?x?x?x?xf32>
                  %dim_W = memref.dim %in, %c2 : memref<?x?x?x?xf32>
                  %is_i_lt_than_zero = cmpi slt, %i_input, %c0 : index
                  %is_i_ge_than_H = cmpi sge, %i_input, %dim_H : index
                  %is_outside_H_input = or %is_i_lt_than_zero, %is_i_ge_than_H : i1
                  %is_j_lt_than_zero = cmpi slt, %j_input, %c0 : index
                  %18 = or %is_outside_H_input, %is_j_lt_than_zero : i1
                  %is_j_ge_than_W = cmpi sge, %j_input, %dim_W : index
                  %is_outside_of_input_bounds = or %18, %is_j_ge_than_W : i1

                  // Load input value or select zero if out of bounds
                  %23 = memref.load %in[%n, %i_cliped, %j_cliped, %channels_in] : memref<?x?x?x?xf32>
                  %24 = select %is_outside_of_input_bounds, %cst_zero, %23 : f32

                  %25 = affine.load %fil[%i_kH, %j_kW, %channels_in, %channels_out] : memref<?x?x?x?xf32>
                  %27 = affine.load %out[%n, %i_H, %j_W, %channels_out] : memref<?x?x?x?xf32>

                  %26 = mulf %25, %24 : f32
                  %28 = addf %27, %26 : f32
                  affine.store %28, %out[%n, %i_H, %j_W, %channels_out] : memref<?x?x?x?xf32>
                }
              }
            }
          }
        }
      }
    }
    call @print_memref_4d_f32(%in) : (memref<?x?x?x?xf32>) -> ()
    call @print_memref_4d_f32(%fil) : (memref<?x?x?x?xf32>) -> ()
    call @print_memref_4d_f32(%out) : (memref<?x?x?x?xf32>) -> ()
    return
  }
  func private @alloc_4d_filled_f32(%n: index, %i_H: index, %j_W: index, %channels_out: index, %channels_in: f32) -> memref<?x?x?x?xf32> {
    %in = memref.alloc(%n, %i_H, %j_W, %channels_out) : memref<?x?x?x?xf32>
    affine.for %i_kH = 0 to %n {
      affine.for %j_kW = 0 to %i_H {
        affine.for %arg7 = 0 to %j_W {
          affine.for %arg8 = 0 to %channels_out {
            affine.store %channels_in, %in[%i_kH, %j_kW, %arg7, %arg8] : memref<?x?x?x?xf32>
          }
        }
      }
    }
    return %in : memref<?x?x?x?xf32>
  }
  func private @print_memref_f32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func private @print_memref_4d_f32(memref<?x?x?x?xf32>) attributes {llvm.emit_c_interface}
  func private @print_memref_3d_f32(memref<?x?x?xf32>) attributes {llvm.emit_c_interface}
}

