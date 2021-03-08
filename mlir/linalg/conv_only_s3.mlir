func @conv_stride_3(%filter: memref<?x?x?x?xf32>, %input: memref<?x?x?x?xf32>, %output: memref<?x?x?x?xf32>) {

  // %c0 = constant 0 : index
  // %c1 = constant 1 : index
  // %c2 = constant 2 : index
  // %c3 = constant 3 : index

  // %N  = dim %output, %c0 : memref<?x?x?x?xf32>
  // %H  = dim %output, %c1 : memref<?x?x?x?xf32>
  // %W  = dim %output, %c2 : memref<?x?x?x?xf32>
  // %O  = dim %output, %c3 : memref<?x?x?x?xf32>
  // %kH = dim %filter, %c0 : memref<?x?x?x?xf32>
  // %kW = dim %filter, %c1 : memref<?x?x?x?xf32>
  // %C  = dim %filter, %c2 : memref<?x?x?x?xf32>



  //    filter(kH,kW,C,O), input(N,H,W,C), output(N,H,W,O)
  linalg.conv(%filter, %input, %output) {
      dilations = [1,1],
      strides = [3,3],
      padding = dense<3> : tensor<2x2xi64> 
    } : memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, memref<?x?x?x?xf32>


  return
}