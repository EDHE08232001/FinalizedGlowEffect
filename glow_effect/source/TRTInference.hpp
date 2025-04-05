#ifndef TRT_INFERENCE_HPP
#define TRT_INFERENCE_HPP

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <chrono>
#include <numeric>
#include <iterator>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>
#include <opencv2/quality.hpp>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <future>       // for std::async, std::future
#include <thread>       // for std::thread
#include <mutex>        // for std::mutex

// Using declarations for brevity.
using namespace std;
using namespace nvinfer1;
using namespace nvonnxparser;

#include "TRTGeneration.hpp"
#include "ImageProcessingUtil.hpp"

/**
 * @brief A class providing TensorRT inference routines for segmentation and super-resolution.
 */
class TRTInference {
public:
	/**
	 * @brief Measures the performance of TRT inference on super-resolution models.
	 *
	 * Loads a TensorRT plan file, performs inference on a single input tensor, and (optionally)
	 * compares the output with an original image.
	 *
	 * @param trt_plan             Path to the serialized TensorRT engine plan file.
	 * @param original_image_path  Path to the original image file for comparison.
	 * @param img_tensor           A 4D tensor (NCHW) containing the preprocessed input image data.
	 * @param num_trials           Number of inference runs for performance measurement.
	 * @param compare_img_bool     If true, compares the final output with the original image.
	 */
	static void measure_trt_performance(const std::string& trt_plan, const std::string& original_image_path,
		torch::Tensor img_tensor, int num_trials, bool compare_img_bool);

	/**
	 * @brief Measures the performance of TRT inference on segmentation models.
	 *
	 * Performs inference on a single input tensor and measures latency over a specified number of trials.
	 *
	 * @param trt_plan   Path to the serialized TensorRT engine plan file.
	 * @param img_tensor A 4D tensor (NCHW) containing the preprocessed input image data.
	 * @param num_trials Number of inference runs for performance measurement.
	 */
	static void measure_segmentation_trt_performance(const std::string& trt_plan, torch::Tensor img_tensor, int num_trials);

	/**
	 * @brief Performs segmentation inference on a batch of images and returns grayscale outputs.
	 *
	 * Loads a TensorRT plan file, processes a batched input tensor, measures latency, and converts
	 * the final output into a vector of single-channel OpenCV Mats.
	 *
	 * @param trt_plan   Path to the serialized TensorRT engine plan file.
	 * @param img_tensor A 4D tensor (NCHW) containing batched preprocessed images.
	 * @param num_trials Number of inference runs for performance measurement.
	 * @return A vector of OpenCV Mats, each representing a grayscale segmentation map.
	 */
	static std::vector<cv::Mat> measure_segmentation_trt_performance_mul(const std::string& trt_plan, torch::Tensor img_tensor, int num_trials);

	/**
	 * @brief Performs segmentation inference on a batch of images concurrently using multiple streams.
	 *
	 * This function splits the input batch into sub-batches and processes each sub-batch on its own
	 * non-blocking CUDA stream and execution context. Pinned memory is used for fast host-device transfers.
	 * The segmentation results from all sub-batches are merged and returned as a vector of OpenCV Mats.
	 *
	 * @param trt_plan         Path to the serialized TensorRT engine plan file.
	 * @param img_tensor_batch A 4D tensor (NCHW) containing batched preprocessed images.
	 * @param num_trials       Number of inference runs for warm-up.
	 * @return A vector of OpenCV Mats, each representing a grayscale segmentation map.
	 */
	static std::vector<cv::Mat> measure_segmentation_trt_performance_mul_concurrent(const std::string& trt_plan, torch::Tensor img_tensor_batch, int num_trials);

	/**
	 * @brief Performs segmentation inference on a batch of images with CUDA Graph acceleration where possible.
	 *
	 * This function implements a hybrid approach to CUDA Graph acceleration:
	 * - Uses separate streams for graph-compatible operations and memory transfers
	 * - Attempts to capture only TensorRT inference in the CUDA Graph (not memory transfers)
	 * - Falls back to regular execution if TensorRT's internal operations are incompatible with graph capture
	 * - Reports performance metrics for either execution mode
	 *
	 * The function maintains the same multi-threading model as the concurrent version,
	 * processing sub-batches in parallel while using pinned memory for efficient transfers.
	 *
	 * @param trt_plan         Path to the serialized TensorRT engine plan file.
	 * @param img_tensor_batch A 4D tensor (NCHW) containing batched preprocessed images.
	 * @param num_trials       Number of inference runs for warm-up.
	 * @return A vector of OpenCV Mats, each representing a grayscale segmentation map.
	 */
	static std::vector<cv::Mat> measure_segmentation_trt_performance_mul_concurrent_graph(const std::string& trt_plan, torch::Tensor img_tensor_batch, int num_trials);

	/**
	 * @brief Processes multiple images in parallel using a single-batch TRT model
	 *
	 * This implementation maximizes parallelism for single-batch TensorRT models by:
	 * 1. Creating multiple execution contexts and CUDA streams (one per worker thread)
	 * 2. Processing images independently and concurrently
	 * 3. Using CUDA Graphs for post-processing operations only (argmax kernel)
	 * 4. Properly separating inference streams from post-processing streams
	 * 5. Implementing robust error handling and recovery mechanisms
	 *
	 * @param trt_plan Path to the single-batch TensorRT plan file
	 * @param img_tensors Vector of individual image tensors (each with batch_size=1)
	 * @param num_streams Number of parallel streams to use
	 * @return Vector of segmentation mask images
	 */
	static std::vector<cv::Mat> measure_segmentation_trt_performance_single_batch_parallel(const std::string& trt_plan, const std::vector<torch::Tensor>& img_tensors, int num_streams);

	/**
	 * @brief Processes multiple images in parallel using a pre-loaded single-batch TRT engine
	 *
	 * This optimized version eliminates the engine deserialization overhead by accepting a
	 * previously loaded TensorRT engine. This is especially useful for video processing
	 * where multiple batches of frames need to be processed with the same model.
	 *
	 * Key optimizations:
	 * 1. Accepts a pre-loaded ICudaEngine pointer instead of loading from a plan file
	 * 2. Creates multiple execution contexts from the shared engine (one per worker thread)
	 * 3. Uses CUDA Graphs for post-processing operations to reduce kernel launch overhead
	 * 4. Properly separates inference streams from post-processing streams
	 * 5. Implements robust error handling and recovery mechanisms
	 *
	 * @param engine Pre-loaded TensorRT engine pointer (must remain valid during the function call)
	 * @param img_tensors Vector of individual image tensors (each with batch_size=1)
	 * @param num_streams Number of parallel streams to use
	 * @return Vector of segmentation mask images
	 */
	static std::vector<cv::Mat> measure_segmentation_trt_performance_single_batch_parallel_preloaded(
		nvinfer1::ICudaEngine* engine,
		const std::vector<torch::Tensor>& img_tensors,
		int num_streams);
};

#endif // TRT_INFERENCE_HPP