#include "armor_detector_nn/armor.hpp"
#include "armor_detector_nn/detector.hpp"
#include <cstdint>
#include <openvino/core/type/element_type.hpp>
#include <string>
#include <vector>
namespace rm_auto_aim {
const std::vector<std::string> coconame = {"B1", "B2", "B3", "B4", "B5",
                                           "BO", "BS", "R1", "R2", "R3",
                                           "R4", "R5", "RO", "RS"};
const std::vector<std::string> armortype = {"SMALL", "LARGE"};
const std::vector<std::string> class_names_ = {"1",
"2",
"3",
"4",
"5",
"outpost",
"guard",
"base"};
Detector::Detector(Config &config) : config_(config) {
  ov::Core core; // 创建OpenVINO核心对象。
  core.set_property(
      ov::cache_dir("../cache")); // 启用模型缓存功能，并指定缓存目录。

  // 读取模型文件，`this->xml_path`是模型文件的路径。
  std::shared_ptr<ov::Model> model = core.read_model(config_.xml_path);

  // 创建预处理和后处理配置对象。
  ov::preprocess::PrePostProcessor ppp =
      ov::preprocess::PrePostProcessor(model);

  // 配置模型输入的预处理步骤。
  // 设置输入数据的元素类型为无符号8位整数，布局为NHWC，颜色格式为RGB。
  ppp.input()
      .tensor()
      .set_element_type(ov::element::u8)
      .set_layout("NHWC")
      .set_color_format(ov::preprocess::ColorFormat::RGB);

  // 配置模型输出的后处理步骤。
  // 设置输出数据的元素类型为32位浮点数。
  ppp.output().tensor().set_element_type(ov::element::f32);

  // 配置进一步的输入预处理操作。
  // 转换元素类型为32位浮点数，颜色格式转换为RGB，并进行缩放。
  ppp.input()
      .preprocess()
      .convert_element_type(ov::element::f32)
      .convert_color(ov::preprocess::ColorFormat::RGB)
      .scale({255, 255, 255});

  // 设置模型的输入布局。
  ppp.input().model().set_layout("NCHW");

  // 应用预处理和后处理配置。
  model = ppp.build();

  // 编译模型，设置设备为自动选择（"AUTO"），优化性能模式为低延迟，推理精度为16位浮点数。
  this->compiled_model = core.compile_model(
      model, "CPU",
      ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
      ov::hint::inference_precision(ov::element::bf16));

  // 创建推理请求对象，用于执行推理。
  this->infer_request = compiled_model.create_infer_request();



  cv::Mat warmUp = cv::Mat::zeros(416, 416, CV_32F);;
  float *input_data = reinterpret_cast<float *>(warmUp.data);
  ov::Tensor input_tensor =
      ov::Tensor(compiled_model.input().get_element_type(),
                  compiled_model.input().get_shape(), input_data);
  infer_request.set_input_tensor(input_tensor);
  infer_request.infer();
}

std::vector<Armor> Detector::detect(const cv::Mat &input) {
  preprocessImage(input);
  infer_request.infer();
  const ov::Tensor &output_tensor = infer_request.get_output_tensor();
  ov::Shape output_shape = output_tensor.get_shape();
  float *detections = output_tensor.data<float>();
  this->frame_w = input.cols;
  this->frame_h = input.rows;
  std::vector<Armor> result = postprocess(detections, output_shape);
  // std::vector<Armor> result ;
  return result;
}
void Detector::preprocessImage(const cv::Mat &frame) {
  try {
    // 计算图像缩放比例，以确保缩放后的图像能够保持原始宽高比并适应目标尺寸。
    // 使用 static_cast<float>() 来确保 inpWidth/frame.cols 和
    // inpHeight/frame.rows 的计算结果为浮点数，避免整数除法导致的精度丢失。
    float scale_ratio =
        std::min(static_cast<float>(config_.inpWidth) / frame.cols,
                 static_cast<float>(config_.inpHeight) / frame.rows);

    // 根据计算出的缩放比例，确定缩放后图像的新尺寸。
    // 这里将缩放比例应用于原图像的宽度和高度，并将结果转换为整数，因为像素的数量不能是小数。
    cv::Size scaled_size(int(frame.cols * scale_ratio),
                         int(frame.rows * scale_ratio));

    // 使用 cv::resize 函数对原始图像进行缩放操作，得到缩放后的图像
    // scaled_frame。参数说明： 0,
    // 0：缩放时的x轴和y轴的比例，这里不使用这两个参数，因为缩放的大小已经通过
    // scaled_size 指定。 cv::INTER_LINEAR：缩放时使用的插值方法，INTER_LINEAR
    // 表示双线性插值，适用于缩放操作，可以在保证速度的同时获得较好的视觉效果。
    cv::Mat scaled_frame;
    cv::resize(frame, scaled_frame, scaled_size, 0, 0, cv::INTER_LINEAR);

    // 计算边框的大小
    int top_border = (config_.inpHeight - scaled_size.height) / 2;
    int bottom_border = config_.inpHeight - scaled_size.height - top_border;
    int left_border = (config_.inpWidth - scaled_size.width) / 2;
    int right_border = config_.inpWidth - scaled_size.width - left_border;

    // 添加黑色边框以生成最终的输入图像
    cv::Mat letterbox_frame;
    cv::copyMakeBorder(scaled_frame, letterbox_frame, top_border, bottom_border,
                       left_border, right_border, cv::BORDER_CONSTANT,
                       cv::Scalar(0, 0, 0));

    // 更新缩放比例和偏移量，供后续处理模型推理结果时,将结果映射到原画面上时使用
    this->rx = static_cast<float>(frame.cols) / scaled_size.width;
    this->ry = static_cast<float>(frame.rows) / scaled_size.height;
    this->dx = left_border; // 水平偏移量
    this->dy = top_border;  // 垂直偏移量

    // 准备模型输入数据
    float *input_data = reinterpret_cast<float *>(letterbox_frame.data);
    ov::Tensor input_tensor =
        ov::Tensor(compiled_model.input().get_element_type(),
                   compiled_model.input().get_shape(), input_data);
    infer_request.set_input_tensor(input_tensor);

  } catch (const std::exception &e) {
    std::cerr << "异常: " << e.what() << std::endl;
  } catch (...) {
    std::cerr << "未知异常" << std::endl;
  }
}
std::vector<Armor> Detector::postprocess(float *detections,
                                         ov::Shape &output_shape) {
  // 模型输出的维度，其中行数为属性数量，列数为检测结果数量
  // int num_attributes = output_shape[1]; // 属性数量，这个模型是26个,
  // 为什么用不上注释了呢?因为下面所有属性的提取我们都是写好的,用不上这个值来循环
  int num_detections =
      output_shape[2]; // 检测结果数量，这个模型是3549个检测结果

  // 定义用于存储检测结果的容器
  std::vector<cv::Rect> boxes;
  std::vector<int> class_ids;
  std::vector<float> confidences;
  std::vector<std::vector<cv::Point2f>> keyPointS;
  std::vector<std::vector<Light>> allLights;

  // TODO 这个是常规的筛选方式, 对每个检测结果,寻找其最大的score,
  // 然后用max_Score去和confThreshold进行比较
  //  相比上面要少个}
  bool ignore = false;
  for (int i = 0; i < num_detections; ++i) {
    // 遍历每一个检测结果
    // 从检测结果中提取最大的类别分数和对应的类别ID
    float max_score = 0.0;
    int class_id = -1;
    int j_top = detect_color == RED ? 14 : 7;
    for (int j = detect_color == RED ? 7 : 0; j < j_top; ++j) {
      // 这里14指遍历14个类别的置信度,如果只要BLUE和RED的置信度,可以将其改为7,
      // 直接拦腰截断了一半, 这里可以根据敌方的眼神开选择j的赋值和区间,
      // 来提高推理速度和容错率,
      // 比赛上不同方的机器人和前哨站,基地,颜色都是不一样的.
      float score = detections[i + (4 + j) * num_detections];
      if (score > max_score) {
        ignore = false;
        for (const auto &ignore_class : ignore_classes_) {
          if (j == ignore_class) {
            ignore = true;
            break;
          }
        }
        if (!ignore) {
          max_score = score;
          class_id = j;
        }
      }
    }

    // 如果最大分数大于置信度阈值，则记录该检测结果
    if (max_score > config_.confThreshold) {
      float cx = detections[i];                      // cx
      float cy = detections[i + num_detections];     // cy
      float ow = detections[i + 2 * num_detections]; // 宽
      float oh = detections[i + 3 * num_detections]; // 高
      // 计算边界框的左上角点和宽高
      cv::Rect box(static_cast<int>(cx - 0.5 * ow),
                   static_cast<int>(cy - 0.5 * oh), static_cast<int>(ow),
                   static_cast<int>(oh));
      boxes.push_back(box);
      class_ids.push_back(class_id);
      confidences.push_back(max_score);

      // 关键点的顺序为:左上, 左下, 右下, 右上
      std::vector<cv::Point2f> kpts;
      std::vector<Light> lights;
      for (int k = 0; k < 4; ++k) {
        float kpt_x =
            ((detections[i + (18 + k * 2) * num_detections] - this->dx) *
             this->rx);
        float kpt_y =
            ((detections[i + (19 + k * 2) * num_detections] - this->dy) *
             this->ry);
        kpts.push_back(cv::Point2f(kpt_x, kpt_y));
      }
      Light llight(kpts[0], kpts[1]), rlight(kpts[3], kpts[2]);
      lights.push_back(llight);
      lights.push_back(rlight);
      allLights.push_back(lights);
      keyPointS.push_back(kpts);
    }
  }

  // 非极大值抑制（NMS）
  std::vector<int> nms_result;
  cv::dnn::NMSBoxes(boxes, confidences, config_.confThreshold,
                    config_.nmsThreshold, nms_result);

  // 存储最终的检测结果
  std::vector<Armor> output;
  for (int idx : nms_result) {
    Armor result(allLights[idx][0], allLights[idx][1]);
    result.number = class_names_[class_ids[idx]%7];
    result.confidence = confidences[idx];
    // result.box = boxes[idx];
    result.classfication_result =
        coconame[class_ids[idx]] + " " + std::to_string(result.confidence);
    result.type = isArmor(result.left_light, result.right_light);
    output.emplace_back(result);
  }
  armors_ = output;
  return output;
}
ArmorType Detector::isArmor(const Light &light_1, const Light &light_2) {
  // Distance between the center of 2 lights (unit : light length)
  float avg_light_length = (light_1.length + light_2.length) / 2;
  float center_distance =
      cv::norm(light_1.center - light_2.center) / avg_light_length;
  // Judge armor type
  ArmorType type;
  type = center_distance > min_large_center_distance ? ArmorType::LARGE
                                                     : ArmorType::SMALL;
  return type;
}
void Detector::drawResults(cv::Mat &img) {
  // Draw armors
  for (const auto &armor : armors_) {
    cv::line(img, armor.left_light.top, armor.right_light.bottom,
             cv::Scalar(0, 255, 0), 2);
    cv::line(img, armor.left_light.bottom, armor.right_light.top,
             cv::Scalar(0, 255, 0), 2);
  }

  // Show numbers and confidence
  for (const auto &armor : armors_) {
    cv::putText(img,
                armor.classfication_result + " " +
                    armortype[static_cast<uint8_t>(armor.type)],
                armor.left_light.top, cv::FONT_HERSHEY_SIMPLEX, 0.8,
                cv::Scalar(0, 255, 255), 2);
  }
}
} // namespace rm_auto_aim