// RUN: mlir-opt --split-input-file --tosa-to-linalg-on-tensors %s -verify-diagnostics -o -| FileCheck %s

func @test_conv2d(%arg0: tensor<1x4x4x4xf32>, %arg1: tensor<8x1x1x4xf32>, %arg2: tensor<8xf32>) -> tensor<1x4x4x8xf32> {
    %0 = "tosa.conv2d"(%arg0, %arg1, %arg2) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<1x4x4x4xf32>, tensor<8x1x1x4xf32>, tensor<8xf32>) -> tensor<1x4x4x8xf32>
    return %0 : tensor<1x4x4x8xf32>
}

func @test_avg_pool2d(%arg0: tensor<1x7x7x9xf32>) -> tensor<1x7x7x9xf32> {
    %0 = "tosa.avg_pool2d"(%arg0) {kernel = [2, 2], pad = [0, 1, 0, 1], stride = [1, 1]} : (tensor<1x7x7x9xf32>) -> tensor<1x7x7x9xf32>
    return %0 : tensor<1x7x7x9xf32>
}

func @test_matmul(%arg0: tensor<14x19xf32>, %arg1: tensor<19x28xf32>) -> tensor<14x28xf32> {
  %0 = "tosa.matmul"(%arg0, %arg1) : (tensor<14x19xf32>, tensor<19x28xf32>) -> tensor<14x28xf32>
  return %0 : tensor<14x28xf32>
}

func @test_sigmoid(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tosa.sigmoid"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

func @test_relu(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tosa.reluN"(%arg0) {max_fp = 3.40282347E+38 : f32, max_int = 0 : i64} : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

func @test_add(%arg0: tensor<13x21x1xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<13x21x1xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}