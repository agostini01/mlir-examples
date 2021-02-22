// Not all of these ops work together.
// best to comment them out individually

func @main(){return}

// func @generalize_conv(%input : memref<1x225x225x3xf32>, %filter: memref<3x3x3x32xf32>, %output: memref<1x112x112x32xf32>) {
//   linalg.conv(%filter, %input, %output) 
//     {
//       dilations = [2, 3], 
//       strides = [4, 5] 
//     } 
//     : memref<3x3x3x32xf32>, memref<1x225x225x3xf32>, memref<1x112x112x32xf32>
//   return
// }

// ----

// func @generalize_conv_padded(%input : memref<1x225x225x5xf32>, %filter: memref<5x3x3x32xf32>, %output: memref<1x?x?x32xf32>) {
//   linalg.conv(%filter, %input, %output) 
//     {
//       dilations = [1, 1], 
//       strides = [1, 1], 
//       padding = dense<[[1, 1], [1, 1]]> : tensor<2x2xi64>
//     } 
//     : memref<5x3x3x32xf32>, memref<1x225x225x5xf32>, memref<1x?x?x32xf32>
//   return
// }

// -----

// func @generalize_matmul_buffer(%A : memref<16x8xf32>, %B: memref<8x32xf32>, %C: memref<16x32xf32>) {
//   linalg.matmul ins(%A, %B: memref<16x8xf32>, memref<8x32xf32>)
//                outs(%C: memref<16x32xf32>)
//   return
// }

// -----

func @generalize_matmul_tensor(%A : tensor<16x8xf32>, %B: tensor<8x32xf32>, %C: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %0 = linalg.matmul ins(%A, %B: tensor<16x8xf32>, tensor<8x32xf32>)
                    outs(%C: tensor<16x32xf32>) -> tensor<16x32xf32>
  return %0: tensor<16x32xf32>
}

// // -----

// func @depthwise_conv_2d_input_nhwc_filter_hwc(%input: memref<1x113x113x96xf32>, %filter: memref<3x3x96xf32>, %output: memref<1x56x56x96xf32>) {
//   linalg.depthwise_conv_2d_input_nhwc_filter_hwc {strides = dense<2> : vector<2xi64>}
//     ins(%input, %filter: memref<1x113x113x96xf32>, memref<3x3x96xf32>)
//     outs(%output: memref<1x56x56x96xf32>)
//   return
// }

// // -----

// func @conv_1d_input_nwc_filter_wcf(%input: memref<?x?x?xf32>, %filter: memref<?x?x?xf32>, %output: memref<?x?x?xf32>) {
//   linalg.conv_1d_input_nwc_filter_wcf {dilations = dense<1> : tensor<1xi64>,
//                                        strides = dense<1> : tensor<1xi64>}
//      ins (%input, %filter: memref<?x?x?xf32>, memref<?x?x?xf32>)
//     outs (%output: memref<?x?x?xf32>)
//   return
// }

// // -----

// func @conv_1d_input_ncw_filter_wcf(%input: memref<?x?x?xf32>, %filter: memref<?x?x?xf32>, %output: memref<?x?x?xf32>) {
//   linalg.conv_1d_input_ncw_filter_wcf {dilations = dense<1> : tensor<1xi64>,
//                                        strides = dense<1> : tensor<1xi64>}
//      ins (%input, %filter: memref<?x?x?xf32>, memref<?x?x?xf32>)
//     outs (%output: memref<?x?x?xf32>)
//   return
// }

// // -----

// func @conv_2d_input_nhwc_filter_hwcf(%input: memref<?x?x?x?xf32>, %filter: memref<?x?x?x?xf32>, %output: memref<?x?x?x?xf32>) {
//   linalg.conv_2d_input_nhwc_filter_hwcf {dilations = dense<2> : tensor<2xi64>,
//                                          strides = dense<3> : tensor<2xi64>}
//      ins (%input, %filter: memref<?x?x?x?xf32>, memref<?x?x?x?xf32>)
//     outs (%output: memref<?x?x?x?xf32>)
//   return
// }

// // -----

// func @conv_2d_input_nchw_filter_hwcf(%input: memref<?x?x?x?xf32>, %filter: memref<?x?x?x?xf32>, %output: memref<?x?x?x?xf32>) {
//   linalg.conv_2d_input_nchw_filter_hwcf {dilations = dense<1> : tensor<2xi64>,
//                                          strides = dense<1> : tensor<2xi64>}
//      ins (%input, %filter: memref<?x?x?x?xf32>, memref<?x?x?x?xf32>)
//     outs (%output: memref<?x?x?x?xf32>)
//   return
// }

// // -----

// func @conv_3d_input_ndhwc_filter_dhwcf(%input: memref<?x?x?x?x?xf32>, %filter: memref<?x?x?x?x?xf32>, %output: memref<?x?x?x?x?xf32>) {
//   linalg.conv_3d_input_ndhwc_filter_dhwcf {dilations = dense<1> : tensor<3xi64>,
//                                            strides = dense<1> : tensor<3xi64>}
//      ins (%input, %filter: memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>)
//     outs (%output: memref<?x?x?x?x?xf32>)
//   return
// }

// // -----

// func @conv_3d_input_ncdhw_filter_dhwcf(%input: memref<?x?x?x?x?xf32>, %filter: memref<?x?x?x?x?xf32>, %output: memref<?x?x?x?x?xf32>) {
//   linalg.conv_3d_input_ncdhw_filter_dhwcf {dilations = dense<1> : tensor<3xi64>,
//                                            strides = dense<1> : tensor<3xi64>}
//      ins (%input, %filter: memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>)
//     outs (%output: memref<?x?x?x?x?xf32>)
//   return
// }
