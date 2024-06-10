#include <filesystem>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

/// 转换图像数据: 先转换元素类型, (可选)然后归一化到[0, 1], (可选)然后交换RB通道

void convert(const cv::Mat &input, cv::Mat &output, const bool normalize, const bool exchangeRB);

/*!
 * \brief fill_tensor_data_image 对网络的输入为图片数据的节点进行赋值，实现图片数据输入网络
 * \param input_tensor 输入节点的tensor
 * \param input_image 输入图片的数据
 * \return 缩放因子, 该缩放是为了将input_image塞进input_tensor
 */
float fill_tensor_data_image(ov::Tensor &input_tensor, const cv::Mat &input_image);

// 打印模型信息, 这个函数修改自$${OPENVINO_COMMON}/utils/src/args_helper.cpp的同名函数
void printInputAndOutputsInfo(const ov::Model &network);