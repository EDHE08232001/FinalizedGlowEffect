/**
 * @file TRTInference.cpp
 * @brief Implementation of TensorRT inference routines for segmentation and super-resolution.
 *
 * This file provides functions to measure inference performance, execute batched or single-image
 * segmentation, and process super-resolution outputs.
 */

#include "TRTInference.hpp"
#include "ImageProcessingUtil.hpp"
#include "nvToolsExt.h"
#include "helper_cuda.h"  // For checkCudaErrors
#include <future>
#include <thread>
#include <mutex>
#include <iterator>
#include "segmentation_kernels.h"

 // Add these external variable declarations
extern int param_KeyLevel;  // Defined in control_gui.cpp
extern int param_KeyScale;  // Defined in control_gui.cpp 
extern int default_scale;   // Defined in control_gui.cpp

 //--------------------------------------------------------------------------
 // Measure Segmentation Inference (Single Image)
 //--------------------------------------------------------------------------
void TRTInference::measure_segmentation_trt_performance(const string& trt_plan, torch::Tensor img_tensor, int num_trials) {
	std::cout << "STARTING measure_trt_performance" << std::endl;

	TRTGeneration::CustomLogger myLogger;
	IRuntime* runtime = createInferRuntime(myLogger);

	ifstream planFile(trt_plan, ios::binary);
	vector<char> plan((istreambuf_iterator<char>(planFile)), istreambuf_iterator<char>());

	ICudaEngine* engine = runtime->deserializeCudaEngine(plan.data(), plan.size());
	IExecutionContext* context = engine->createExecutionContext();
	if (!engine || !context) {
		cerr << "Failed to deserialize engine or create execution context." << endl;
		exit(EXIT_FAILURE);
	}

	// Use Dims4 for input and output.
	nvinfer1::Dims4 inputDims;
	nvinfer1::Dims4 outputDims;

	int input_size = img_tensor.numel();
	float* h_input;
	cudaMallocHost((void**)&h_input, input_size * sizeof(float));

	// Collect output binding indices.
	int numBindings = engine->getNbBindings();
	std::vector<int> outputBindingIndices;
	std::vector<std::string> outputTensorNames;
	for (int i = 1; i < numBindings; ++i) {
		outputBindingIndices.push_back(i);
		outputTensorNames.push_back(engine->getBindingName(i));
	}

	// Set input dimensions (NCHW).
	inputDims.d[0] = img_tensor.size(0);
	inputDims.d[1] = img_tensor.size(1);
	inputDims.d[2] = img_tensor.size(2);
	inputDims.d[3] = img_tensor.size(3);
	context->setBindingDimensions(0, inputDims);

	std::vector<void*> d_outputs;
	std::vector<float*> h_outputs;
	std::vector<void*> bindings;

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	void* d_input;
	cudaMalloc(&d_input, input_size * sizeof(float));

	std::memcpy(h_input, img_tensor.data_ptr<float>(), input_size * sizeof(float));
	cudaError_t memcpyStatus = cudaMemcpyAsync(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice, stream);
	if (memcpyStatus != cudaSuccess) {
		cerr << "CUDA error (cudaMemcpyAsync): " << cudaGetErrorString(memcpyStatus) << endl;
		exit(EXIT_FAILURE);
	}
	bindings.push_back(d_input);

	// For each output binding, copy dimensions manually.
	for (int i : outputBindingIndices) {
		// Get the raw dims
		nvinfer1::Dims dims = engine->getBindingDimensions(i);
		outputDims.nbDims = dims.nbDims;
		for (int j = 0; j < dims.nbDims; ++j) {
			outputDims.d[j] = dims.d[j];
			if (outputDims.d[j] < 0) {  // Handle dynamic dimensions.
				outputDims.d[j] = inputDims.d[j];
			}
		}
		int outputSize = outputDims.d[0] * outputDims.d[1] * outputDims.d[2] * outputDims.d[3];
		float* h_output = new float[outputSize];
		void* d_output;
		if (cudaMalloc(&d_output, outputSize * sizeof(float)) != cudaSuccess) {
			cerr << "Device memory allocation failed" << endl;
			exit(EXIT_FAILURE);
		}
		h_outputs.push_back(h_output);
		d_outputs.push_back(d_output);
		bindings.push_back(d_output);
	}

	vector<float> latencies;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Warm-up runs.
	for (int i = 0; i < 10; ++i) {
		context->enqueueV2(bindings.data(), stream, nullptr);
	}

	cudaEventRecord(start, stream);
	for (int i = 0; i < num_trials; ++i) {
		char str_buf[100];
		std::sprintf(str_buf, "frame%03d", i);
		nvtxRangePushA(str_buf);
		if (!context->enqueueV2(bindings.data(), stream, nullptr)) {
			cerr << "TensorRT enqueueV2 failed!" << endl;
			exit(EXIT_FAILURE);
		}
		nvtxRangePop();
	}
	cudaEventRecord(stop, stream);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	latencies.push_back(milliseconds);

	float* last_h_output = h_outputs.back();
	int last_output_size = outputDims.d[0] * outputDims.d[1] * outputDims.d[2] * outputDims.d[3];
	cudaMemcpyAsync(last_h_output, d_outputs.back(), last_output_size * sizeof(float), cudaMemcpyDeviceToHost, stream);
	cudaStreamSynchronize(stream);

	float min_val = *std::min_element(last_h_output, last_h_output + last_output_size);
	float max_val = *std::max_element(last_h_output, last_h_output + last_output_size);
	float avg_val = std::accumulate(last_h_output, last_h_output + last_output_size, 0.0f) / last_output_size;
	cout << "Last Output Tensor - Min: " << min_val << ", Max: " << max_val << ", Avg: " << avg_val << endl;

	float average_latency = std::accumulate(latencies.begin(), latencies.end(), 0.0f) / num_trials;
	cout << "TRT - Average Latency over " << num_trials << " trials: " << average_latency << " ms" << endl;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	int batch = outputDims.d[0];
	int num_classes = outputDims.d[1];
	int height = outputDims.d[2];
	int width = outputDims.d[3];
	auto last_output_tensor = torch::from_blob(last_h_output, { batch, num_classes, height, width }, torch::kFloat32);

	cout << "\nLast output tensor dimensions: ";
	for (int i = 0; i < last_output_tensor.dim(); ++i) {
		cout << last_output_tensor.size(i) << " ";
	}
	cout << std::endl;

	auto max_out = torch::max(last_output_tensor, 1);
	auto class_labels = std::get<1>(max_out);
	int scale = 255 / 21;
	auto image_post = class_labels * scale;

	cout << "\nimage_post dimensions: ";
	for (int i = 0; i < image_post.dim(); ++i) {
		cout << image_post.size(i) << " ";
	}
	cout << std::endl;

	auto permuted_img = image_post.permute({ 1, 2, 0 }).to(torch::kU8);
	cout << "permuted_img dimensions: ";
	for (int i = 0; i < permuted_img.dim(); ++i) {
		cout << permuted_img.size(i) << " ";
	}
	cout << std::endl;

	cv::Mat cv_img(permuted_img.size(0), permuted_img.size(1), CV_8UC1, permuted_img.data_ptr<uchar>());
	cout << "Segmentation visualization ready." << endl;

	cudaFreeHost(h_input);
	for (float* h_output : h_outputs) {
		delete[] h_output;
	}
	cudaFree(d_input);
	for (void* d_output : d_outputs) {
		cudaFree(d_output);
	}

	context->destroy();
	engine->destroy();
	runtime->destroy();
	cudaStreamDestroy(stream);
}

//--------------------------------------------------------------------------
// Measure Segmentation Inference (Batch) - Original Version
//--------------------------------------------------------------------------
std::vector<cv::Mat> TRTInference::measure_segmentation_trt_performance_mul(const string& trt_plan, torch::Tensor img_tensor_batch, int num_trials) {
	std::vector<cv::Mat> grayscale_images;
	std::cout << "STARTING measure_segmentation_trt_performance_mul" << std::endl;

	TRTGeneration::CustomLogger myLogger;
	IRuntime* runtime = createInferRuntime(myLogger);

	ifstream planFile(trt_plan, ios::binary);
	vector<char> plan((istreambuf_iterator<char>(planFile)), istreambuf_iterator<char>());

	ICudaEngine* engine = runtime->deserializeCudaEngine(plan.data(), plan.size());
	IExecutionContext* context = engine->createExecutionContext();
	if (!engine || !context) {
		cerr << "Failed to deserialize engine or create execution context." << endl;
		exit(EXIT_FAILURE);
	}

	float* h_input;
	int input_size = img_tensor_batch.numel();
	cudaMallocHost((void**)&h_input, input_size * sizeof(float));

	nvinfer1::Dims4 inputDims;
	nvinfer1::Dims4 outputDims;

	std::vector<int> outputBindingIndices;
	std::vector<std::string> outputTensorNames;
	for (int i = 1; i < engine->getNbBindings(); ++i) {
		outputBindingIndices.push_back(i);
		outputTensorNames.push_back(engine->getBindingName(i));
	}

	inputDims.d[0] = img_tensor_batch.size(0);
	inputDims.d[1] = img_tensor_batch.size(1);
	inputDims.d[2] = img_tensor_batch.size(2);
	inputDims.d[3] = img_tensor_batch.size(3);
	context->setBindingDimensions(0, inputDims);

	std::vector<void*> d_outputs;
	std::vector<float*> h_outputs;
	std::vector<void*> bindings;

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	void* d_input;
	cudaMalloc(&d_input, input_size * sizeof(float));

	std::memcpy(h_input, img_tensor_batch.data_ptr<float>(), input_size * sizeof(float));

	cudaError_t memcpyStatus = cudaMemcpyAsync(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice, stream);
	if (memcpyStatus != cudaSuccess) {
		cerr << "CUDA error (cudaMemcpyAsync): " << cudaGetErrorString(memcpyStatus) << endl;
		exit(EXIT_FAILURE);
	}
	bindings.push_back(d_input);

	for (int i : outputBindingIndices) {
		// Copy dimensions manually from raw dims.
		nvinfer1::Dims dims = context->getBindingDimensions(i);
		outputDims.nbDims = dims.nbDims;
		for (int j = 0; j < dims.nbDims; ++j) {
			outputDims.d[j] = dims.d[j];
			if (outputDims.d[j] < 0)
				outputDims.d[j] = inputDims.d[j];
		}
		int outputSize = outputDims.d[0] * outputDims.d[1] * outputDims.d[2] * outputDims.d[3];
		float* h_output = new float[outputSize];
		void* d_output;
		if (cudaMalloc(&d_output, outputSize * sizeof(float)) != cudaSuccess) {
			cerr << "Device memory allocation failed" << endl;
			exit(EXIT_FAILURE);
		}
		h_outputs.push_back(h_output);
		d_outputs.push_back(d_output);
		bindings.push_back(d_output);
	}

	vector<float> latencies;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	for (int i = 0; i < 10; ++i) {
		context->enqueueV2(bindings.data(), stream, nullptr);
	}

	cudaEventRecord(start, stream);
	for (int i = 0; i < num_trials; ++i) {
		char str_buf[100];
		std::sprintf(str_buf, "frame%03d", i);
		nvtxRangePushA(str_buf);
		if (!context->enqueueV2(bindings.data(), stream, nullptr)) {
			cerr << "TensorRT enqueueV2 failed!" << endl;
			exit(EXIT_FAILURE);
		}
		nvtxRangePop();
	}
	cudaEventRecord(stop, stream);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	latencies.push_back(milliseconds);

	float* last_h_output = h_outputs.back();
	int last_output_size = outputDims.d[0] * outputDims.d[1] * outputDims.d[2] * outputDims.d[3];
	cudaMemcpyAsync(last_h_output, d_outputs.back(), last_output_size * sizeof(float), cudaMemcpyDeviceToHost, stream);
	cudaStreamSynchronize(stream);

	float min_val = *std::min_element(last_h_output, last_h_output + last_output_size);
	float max_val = *std::max_element(last_h_output, last_h_output + last_output_size);
	float avg_val = std::accumulate(last_h_output, last_h_output + last_output_size, 0.0f) / last_output_size;
	cout << "Last Output Tensor - Min: " << min_val << ", Max: " << max_val << ", Avg: " << avg_val << endl;

	float average_latency = std::accumulate(latencies.begin(), latencies.end(), 0.0f) / num_trials;
	cout << "TRT - Average Latency over " << num_trials << " trials: " << average_latency << " ms" << endl;
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	int batch = outputDims.d[0];
	int num_classes = outputDims.d[1];
	int height = outputDims.d[2];
	int width = outputDims.d[3];
	auto last_output_tensor = torch::from_blob(last_h_output, { batch, num_classes, height, width }, torch::kFloat32);

	cout << "\nLast output tensor dimensions: ";
	for (int i = 0; i < last_output_tensor.dim(); ++i) {
		cout << last_output_tensor.size(i) << " ";
	}
	cout << std::endl;

	auto max_out = torch::max(last_output_tensor, 1);
	auto class_labels = std::get<1>(max_out);
	int scale = 255 / 21;
	auto image_post = class_labels * scale;

	for (int i = 0; i < batch; ++i) {
		auto single_image_post = image_post[i].squeeze().to(torch::kU8);
		cv::Mat cv_img(single_image_post.size(0), single_image_post.size(1), CV_8UC1, single_image_post.data_ptr<uchar>());
		grayscale_images.push_back(cv_img.clone());
	}

	cudaFreeHost(h_input);
	for (float* h_output : h_outputs) {
		delete[] h_output;
	}
	cudaFree(d_input);
	for (void* d_output : d_outputs) {
		cudaFree(d_output);
	}
	context->destroy();
	engine->destroy();
	runtime->destroy();
	cudaStreamDestroy(stream);

	return grayscale_images;
}

//--------------------------------------------------------------------------
// New Function: Measure Segmentation Inference (Batch) Concurrent Version
//--------------------------------------------------------------------------
std::vector<cv::Mat> TRTInference::measure_segmentation_trt_performance_mul_concurrent(
	const std::string& trt_plan, torch::Tensor img_tensor_batch, int num_trials) {

	std::cout << "STARTING measure_segmentation_trt_performance_mul_concurrent (multi-stream concurrent version)" << std::endl;

	// -----------------------------
	// Create TensorRT runtime and deserialize the engine from the plan file.
	// -----------------------------
	TRTGeneration::CustomLogger myLogger;
	IRuntime* runtime = createInferRuntime(myLogger);
	ifstream planFile(trt_plan, ios::binary);
	vector<char> plan((istreambuf_iterator<char>(planFile)), istreambuf_iterator<char>());
	ICudaEngine* engine = runtime->deserializeCudaEngine(plan.data(), plan.size());
	if (!engine) {
		std::cerr << "Failed to deserialize engine in concurrent segmentation." << std::endl;
		exit(EXIT_FAILURE);
	}

	// -----------------------------
	// Determine batch and thread parameters.
	// -----------------------------
	int totalBatch = img_tensor_batch.size(0);  // Total number of images.
	int numThreads = 2;                           // Fixed number of threads.
	int subBatch = (totalBatch + numThreads - 1) / numThreads;  // Images per thread.

	// allResults will store the segmentation output for each image.
	std::vector<cv::Mat> allResults(totalBatch);
	std::mutex resultMutex;  // Mutex to protect shared results vector.
	std::vector<std::thread> threads;

	// -----------------------------
	// Launch threads to process sub-batches concurrently.
	// -----------------------------
	for (int t = 0; t < numThreads; ++t) {
		threads.emplace_back([&, t]() {
			// Create execution context for this thread.
			IExecutionContext* context = engine->createExecutionContext();
			if (!context) {
				std::cerr << "Failed to create execution context for thread " << t << std::endl;
				return;
			}

			// Create a CUDA stream with non-blocking flags.
			cudaStream_t stream;
			checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

			// Calculate sub-batch indices for this thread.
			int startIdx = t * subBatch;
			int endIdx = std::min(startIdx + subBatch, totalBatch);
			int validCount = endIdx - startIdx;  // Number of valid images in this sub-batch.
			if (validCount <= 0)
				return;

			// -----------------------------
			// Extract sub-batch from the global image tensor.
			// -----------------------------
			torch::Tensor subTensor = img_tensor_batch.slice(0, startIdx, endIdx);
			// Remove extra singleton dimension if present (e.g., [N,1,3,384,384] -> [N,3,384,384]).
			if ((subTensor.dim() == 5 && subTensor.size(1) == 1) ||
				(subTensor.dim() == 4 && subTensor.size(1) == 1))
			{
				subTensor = subTensor.squeeze(1);
			}
			// Pad the sub-batch to have exactly 4 images if needed.
			if (subTensor.size(0) < 4) {
				int pad = 4 - subTensor.size(0);
				torch::Tensor lastFrame = subTensor[subTensor.size(0) - 1].unsqueeze(0);
				torch::Tensor padTensor = lastFrame.repeat({ pad, 1, 1, 1 });
				subTensor = torch::cat({ subTensor, padTensor }, 0);
			}

			// -----------------------------
			// Allocate host and device memory for input data.
			// -----------------------------
			int inputSize = subTensor.numel();
			float* h_input = nullptr;
			checkCudaErrors(cudaMallocHost((void**)&h_input, inputSize * sizeof(float)));
			std::memcpy(h_input, subTensor.data_ptr<float>(), inputSize * sizeof(float));

			void* d_input = nullptr;
			checkCudaErrors(cudaMalloc(&d_input, inputSize * sizeof(float)));
			checkCudaErrors(cudaMemcpyAsync(d_input, h_input, inputSize * sizeof(float), cudaMemcpyHostToDevice, stream));

			// Set input dimensions for the context.
			nvinfer1::Dims4 inputDims;
			inputDims.d[0] = subTensor.size(0);
			inputDims.d[1] = subTensor.size(1);
			inputDims.d[2] = subTensor.size(2);
			inputDims.d[3] = subTensor.size(3);
			context->setBindingDimensions(0, inputDims);

			// -----------------------------
			// Prepare output buffers.
			// -----------------------------
			std::vector<void*> bindings;
			bindings.push_back(d_input);  // Binding index 0: Input buffer.
			int numBindings = engine->getNbBindings();
			std::vector<float*> h_outputs;  // Host memory for outputs.
			std::vector<void*> d_outputs;   // Device memory for outputs.
			nvinfer1::Dims4 outputDims;
			for (int i = 1; i < numBindings; ++i) {
				// Retrieve binding dimensions for each output.
				nvinfer1::Dims dims = context->getBindingDimensions(i);
				outputDims.nbDims = dims.nbDims;
				for (int j = 0; j < dims.nbDims; ++j) {
					outputDims.d[j] = dims.d[j];
					if (outputDims.d[j] < 0)
						outputDims.d[j] = inputDims.d[j];  // Fallback to input dimensions if negative.
				}
				int outputSize = outputDims.d[0] * outputDims.d[1] * outputDims.d[2] * outputDims.d[3];
				float* h_output = nullptr;
				checkCudaErrors(cudaMallocHost((void**)&h_output, outputSize * sizeof(float)));
				void* d_output = nullptr;
				checkCudaErrors(cudaMalloc(&d_output, outputSize * sizeof(float)));
				h_outputs.push_back(h_output);
				d_outputs.push_back(d_output);
				bindings.push_back(d_output);
			}

			// -----------------------------
			// Perform optional warm-up runs.
			// -----------------------------
			for (int i = 0; i < 3; ++i) {
				context->enqueueV2(bindings.data(), stream, nullptr);
			}

			// -----------------------------
			// Enqueue inference and check for errors.
			// -----------------------------
			if (!context->enqueueV2(bindings.data(), stream, nullptr)) {
				std::cerr << "TensorRT enqueueV2 failed in thread " << t << std::endl;
				exit(EXIT_FAILURE);
			}

			// -----------------------------
			// Copy inference results from device to host.
			// -----------------------------
			int outSize = outputDims.d[0] * outputDims.d[1] * outputDims.d[2] * outputDims.d[3];
			float* lastOutput = h_outputs.back();
			checkCudaErrors(cudaMemcpyAsync(lastOutput, d_outputs.back(), outSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
			cudaStreamSynchronize(stream);  // Ensure all operations complete.

			// -----------------------------
			// Post-process the output tensor.
			// -----------------------------
			auto outputTensor = torch::from_blob(lastOutput, { outputDims.d[0], outputDims.d[1], outputDims.d[2], outputDims.d[3] }, torch::kFloat32);
			auto maxOut = torch::max(outputTensor, 1);  // Find maximum values along the channel dimension.
			auto segMask = std::get<1>(maxOut);  // Expected shape: [4, H, W], where each element is the class index.
			int scaleFactor = 255 / 21;  // Scale factor to map class indices to an 8-bit range.
			auto imagePost = segMask * scaleFactor;

			// -----------------------------
			// Convert tensor outputs to cv::Mat and handle padded outputs.
			// -----------------------------
			std::vector<cv::Mat> localResults;
			for (int i = 0; i < validCount; ++i) {
				auto single = imagePost[i].to(torch::kU8);  // Convert to unsigned 8-bit.
				cv::Mat result(single.size(0), single.size(1), CV_8UC1, single.data_ptr<uchar>());
				localResults.push_back(result.clone());
			}

			// -----------------------------
			// Safely update the global results vector.
			// -----------------------------
			{
				std::lock_guard<std::mutex> lock(resultMutex);
				for (int i = 0; i < validCount; ++i) {
					allResults[startIdx + i] = localResults[i];
				}
			}

			// -----------------------------
			// Free allocated host and device memory, destroy CUDA stream and context.
			// -----------------------------
			cudaFreeHost(h_input);
			cudaFree(d_input);
			for (auto ptr : h_outputs) {
				cudaFreeHost(ptr);
			}
			for (auto dptr : d_outputs) {
				cudaFree(dptr);
			}
			cudaStreamDestroy(stream);
			context->destroy();
			});
	}

	// Wait for all threads to complete execution.
	for (auto& t : threads)
		t.join();

	// Destroy the engine and runtime to free resources.
	engine->destroy();
	runtime->destroy();

	return allResults;
}

//--------------------------------------------------------------------------
// Concurrent Segmentation with CUDA Graph
//--------------------------------------------------------------------------
std::vector<cv::Mat> TRTInference::measure_segmentation_trt_performance_mul_concurrent_graph(const std::string& trt_plan, torch::Tensor img_tensor_batch, int num_trials) {

	std::cout << "STARTING measure_segmentation_trt_performance_mul_concurrent_graph (Hybrid CUDA Graph approach)" << std::endl;

	// Create TensorRT runtime and deserialize the engine
	TRTGeneration::CustomLogger myLogger;
	IRuntime* runtime = createInferRuntime(myLogger);
	ifstream planFile(trt_plan, ios::binary);
	vector<char> plan((istreambuf_iterator<char>(planFile)), istreambuf_iterator<char>());
	std::cout << "Loaded engine size: " << plan.size() / (1024 * 1024) << " MiB" << std::endl;

	auto start_time = std::chrono::high_resolution_clock::now();
	ICudaEngine* engine = runtime->deserializeCudaEngine(plan.data(), plan.size());
	auto end_time = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
	std::cout << "Deserialization required " << duration.count() << " microseconds." << std::endl;

	if (!engine) {
		std::cerr << "Failed to deserialize engine in graph segmentation." << std::endl;
		exit(EXIT_FAILURE);
	}

	// Setup for multi-threaded processing
	int totalBatch = img_tensor_batch.size(0);
	int numThreads = 2;
	int subBatch = (totalBatch + numThreads - 1) / numThreads;
	std::vector<cv::Mat> allResults(totalBatch);
	std::mutex resultMutex;
	std::vector<std::thread> threads;

	// Launch threads to process sub-batches concurrently
	for (int t = 0; t < numThreads; ++t) {
		threads.emplace_back([&, t]() {
			// Create execution context for this thread
			IExecutionContext* context = engine->createExecutionContext();
			if (!context) {
				std::cerr << "Failed to create execution context for thread " << t << std::endl;
				return;
			}

			// Create three streams: one for pre-processing, one for inference, one for post-processing
			cudaStream_t preStream, inferStream, postStream;
			checkCudaErrors(cudaStreamCreateWithFlags(&preStream, cudaStreamNonBlocking));
			checkCudaErrors(cudaStreamCreateWithFlags(&inferStream, cudaStreamNonBlocking));
			checkCudaErrors(cudaStreamCreateWithFlags(&postStream, cudaStreamNonBlocking));

			// Calculate sub-batch indices
			int startIdx = t * subBatch;
			int endIdx = std::min(startIdx + subBatch, totalBatch);
			int validCount = endIdx - startIdx;
			if (validCount <= 0)
				return;

			// Extract sub-batch from the global image tensor
			torch::Tensor subTensor = img_tensor_batch.slice(0, startIdx, endIdx);

			// Remove extra singleton dimensions if needed
			if ((subTensor.dim() == 5 && subTensor.size(1) == 1) ||
				(subTensor.dim() == 4 && subTensor.size(1) == 1))
			{
				subTensor = subTensor.squeeze(1);
			}

			// Pad the sub-batch to have exactly 4 images if needed
			if (subTensor.size(0) < 4) {
				int pad = 4 - subTensor.size(0);
				torch::Tensor lastFrame = subTensor[subTensor.size(0) - 1].unsqueeze(0);
				torch::Tensor padTensor = lastFrame.repeat({ pad, 1, 1, 1 });
				subTensor = torch::cat({ subTensor, padTensor }, 0);
			}

			// Set input dimensions
			nvinfer1::Dims4 inputDims;
			inputDims.d[0] = subTensor.size(0);
			inputDims.d[1] = subTensor.size(1);
			inputDims.d[2] = subTensor.size(2);
			inputDims.d[3] = subTensor.size(3);
			context->setBindingDimensions(0, inputDims);

			// Allocate memory for input and output
			int inputSize = subTensor.numel();
			float* h_input = nullptr;
			void* d_input = nullptr;
			checkCudaErrors(cudaMallocHost((void**)&h_input, inputSize * sizeof(float)));
			checkCudaErrors(cudaMalloc(&d_input, inputSize * sizeof(float)));

			// Copy input data to host pinned memory
			std::memcpy(h_input, subTensor.data_ptr<float>(), inputSize * sizeof(float));

			// Setup bindings and allocate output memory
			std::vector<void*> bindings;
			bindings.push_back(d_input);
			int numBindings = engine->getNbBindings();
			std::vector<float*> h_outputs;
			std::vector<void*> d_outputs;
			nvinfer1::Dims4 outputDims;

			for (int i = 1; i < numBindings; ++i) {
				nvinfer1::Dims dims = context->getBindingDimensions(i);
				outputDims.nbDims = dims.nbDims;
				for (int j = 0; j < dims.nbDims; ++j) {
					outputDims.d[j] = dims.d[j];
					if (outputDims.d[j] < 0)
						outputDims.d[j] = inputDims.d[j];
				}
				int outputSize = outputDims.d[0] * outputDims.d[1] * outputDims.d[2] * outputDims.d[3];
				float* h_output = nullptr;
				void* d_output = nullptr;
				checkCudaErrors(cudaMallocHost((void**)&h_output, outputSize * sizeof(float)));
				checkCudaErrors(cudaMalloc(&d_output, outputSize * sizeof(float)));
				h_outputs.push_back(h_output);
				d_outputs.push_back(d_output);
				bindings.push_back(d_output);
			}

			// Allocate device memory for post-processing
			int batch = outputDims.d[0];
			int height = outputDims.d[2];
			int width = outputDims.d[3];
			unsigned char* d_argmax_output = nullptr;
			checkCudaErrors(cudaMalloc(&d_argmax_output, batch * height * width * sizeof(unsigned char)));

			// Pre-processing: Copy input from host to device (not part of the graph)
			checkCudaErrors(cudaMemcpyAsync(d_input, h_input, inputSize * sizeof(float),
				cudaMemcpyHostToDevice, preStream));
			checkCudaErrors(cudaStreamSynchronize(preStream));

			// Setup timing
			cudaEvent_t start, stop;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);

			// Declare graph objects outside the try block
			cudaGraph_t postprocessGraph = nullptr;
			cudaGraphExec_t postprocessGraphExec = nullptr;
			bool useGraph = false;

			// To fix the compilation error with goto, we'll declare the results vector here
			std::vector<cv::Mat> localResults;
			unsigned char* h_argmax_output = nullptr;

			// First perform a regular inference run for warmup
			// This is always done regardless of graph capture success
			for (int i = 0; i < 2; ++i) {
				// Run TensorRT inference
				if (!context->enqueueV2(bindings.data(), inferStream, nullptr)) {
					std::cerr << "TensorRT enqueueV2 failed during warmup" << std::endl;
					goto cleanup; // Jump to resource cleanup
				}
				checkCudaErrors(cudaStreamSynchronize(inferStream));
			}

			// Now try to create a graph for post-processing operations
			try {
				// Create a graph for post-processing kernels
				checkCudaErrors(cudaStreamBeginCapture(postStream, cudaStreamCaptureModeRelaxed));

				// Call our custom kernel for argmax operation
				launchArgmaxKernel(
					static_cast<float*>(d_outputs.back()),
					d_argmax_output,
					batch,
					outputDims.d[1], // num_classes
					height,
					width,
					postStream
				);

				checkCudaErrors(cudaStreamEndCapture(postStream, &postprocessGraph));
				checkCudaErrors(cudaGraphInstantiate(&postprocessGraphExec, postprocessGraph, nullptr, nullptr, 0));

				// If we get here, graph capture for post-processing was successful
				useGraph = true;
				std::cout << "Thread " << t << " successfully created post-processing graph" << std::endl;
			}
			catch (const std::exception& e) {
				std::cerr << "CUDA Graph capture failed: " << e.what() << std::endl;
				std::cerr << "Falling back to regular execution..." << std::endl;
				useGraph = false;

				// Clean up any partial graph resources
				if (postprocessGraph) {
					cudaGraphDestroy(postprocessGraph);
					postprocessGraph = nullptr;
				}
				if (postprocessGraphExec) {
					cudaGraphExecDestroy(postprocessGraphExec);
					postprocessGraphExec = nullptr;
				}
			}

			// Execute inference timing
			cudaEventRecord(start, inferStream);

			// For TensorRT inference, we always use regular execution since it's not compatible with graph capture
			if (!context->enqueueV2(bindings.data(), inferStream, nullptr)) {
				std::cerr << "TensorRT enqueueV2 failed" << std::endl;
				goto cleanup; // Jump to resource cleanup
			}
			checkCudaErrors(cudaStreamSynchronize(inferStream));

			// Execute post-processing (either with graph or regular method)
			if (useGraph && postprocessGraphExec) {
				checkCudaErrors(cudaGraphLaunch(postprocessGraphExec, postStream));
				checkCudaErrors(cudaStreamSynchronize(postStream));
			}
			else {
				// Fall back to regular kernel launch if graph capture failed
				launchArgmaxKernel(
					static_cast<float*>(d_outputs.back()),
					d_argmax_output,
					batch,
					outputDims.d[1], // num_classes
					height,
					width,
					postStream
				);
				checkCudaErrors(cudaStreamSynchronize(postStream));
			}

			// Copy results from device to host
			h_argmax_output = new unsigned char[batch * height * width];
			checkCudaErrors(cudaMemcpyAsync(
				h_argmax_output,
				d_argmax_output,
				batch * height * width * sizeof(unsigned char),
				cudaMemcpyDeviceToHost,
				postStream
			));
			checkCudaErrors(cudaStreamSynchronize(postStream));

			cudaEventRecord(stop, inferStream);
			checkCudaErrors(cudaStreamSynchronize(inferStream));

			float milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, start, stop);
			std::cout << "Thread " << t << " execution time: " << milliseconds << " ms"
				<< (useGraph ? " (with partial CUDA Graph)" : " (without CUDA Graph)") << std::endl;

			// Convert to OpenCV Mats and update results
			for (int i = 0; i < validCount; ++i) {
				cv::Mat result(height, width, CV_8UC1);
				// Copy the segmentation result for this batch item
				std::memcpy(
					result.data,
					h_argmax_output + (i * height * width),
					height * width * sizeof(unsigned char)
				);
				localResults.push_back(result.clone());
			}

			// Update shared results vector with thread-local results
			{
				std::lock_guard<std::mutex> lock(resultMutex);
				for (int i = 0; i < validCount; ++i) {
					allResults[startIdx + i] = localResults[i];
				}
			}

		cleanup:
			// Clean up temporary host memory
			if (h_argmax_output) {
				delete[] h_argmax_output;
			}

			// Clean up resources
			cudaEventDestroy(start);
			cudaEventDestroy(stop);

			if (postprocessGraphExec) {
				cudaGraphExecDestroy(postprocessGraphExec);
			}
			if (postprocessGraph) {
				cudaGraphDestroy(postprocessGraph);
			}

			cudaFree(d_argmax_output);
			cudaFreeHost(h_input);
			cudaFree(d_input);
			for (auto ptr : h_outputs) {
				cudaFreeHost(ptr);
			}
			for (auto dptr : d_outputs) {
				cudaFree(dptr);
			}
			cudaStreamDestroy(preStream);
			cudaStreamDestroy(inferStream);
			cudaStreamDestroy(postStream);
			context->destroy();
			});
	}

	// Wait for all threads to complete
	for (auto& t : threads) {
		t.join();
	}

	// Clean up global resources
	engine->destroy();
	runtime->destroy();

	return allResults;
}

//--------------------------------------------------------------------------------------
// Processes multiple images in parallel using a single-batch TRT model with CUDA Graph
//--------------------------------------------------------------------------------------
std::vector<cv::Mat> TRTInference::measure_segmentation_trt_performance_single_batch_parallel(const std::string& trt_plan, const std::vector<torch::Tensor>& img_tensors, int num_streams) {

	std::cout << "Starting optimized parallel single-batch segmentation with post-processing CUDA Graph acceleration" << std::endl;

	// Number of images to process
	int num_images = img_tensors.size();
	if (num_images == 0) {
		return {};
	}

	// Create the TensorRT runtime and load the engine
	TRTGeneration::CustomLogger myLogger;
	IRuntime* runtime = createInferRuntime(myLogger);
	ifstream planFile(trt_plan, ios::binary);
	if (!planFile.is_open()) {
		std::cerr << "Error: Could not open plan file: " << trt_plan << std::endl;
		return {};
	}

	vector<char> plan((istreambuf_iterator<char>(planFile)), istreambuf_iterator<char>());
	std::cout << "Loaded single-batch plan file: " << plan.size() / (1024 * 1024) << " MiB" << std::endl;

	ICudaEngine* engine = runtime->deserializeCudaEngine(plan.data(), plan.size());
	if (!engine) {
		std::cerr << "Error: Failed to deserialize CUDA engine" << std::endl;
		runtime->destroy();
		return {};
	}

	// Results container
	std::vector<cv::Mat> results(num_images);
	std::mutex resultMutex;
	std::vector<std::thread> threads;

	// Calculate images per worker thread
	int images_per_thread = (num_images + num_streams - 1) / num_streams;

	// Performance metrics for reporting
	std::vector<double> processing_times(num_streams, 0.0);
	std::vector<int> frames_processed(num_streams, 0);
	std::vector<bool> graph_usage(num_streams, false);

	// Launch parallel worker threads
	for (int t = 0; t < num_streams; ++t) {
		threads.emplace_back([&, t]() {
			// Calculate the range of images for this worker
			int start_idx = t * images_per_thread;
			int end_idx = std::min(start_idx + images_per_thread, num_images);

			if (start_idx >= num_images) {
				return; // No images for this worker
			}

			// Create execution context for this worker
			IExecutionContext* context = engine->createExecutionContext();
			if (!context) {
				std::cerr << "Error: Failed to create execution context for worker " << t << std::endl;
				return;
			}

			// Create CUDA streams for this worker - separate streams for inference and post-processing
			cudaStream_t inferStream, postStream;
			checkCudaErrors(cudaStreamCreateWithFlags(&inferStream, cudaStreamNonBlocking));
			checkCudaErrors(cudaStreamCreateWithFlags(&postStream, cudaStreamNonBlocking));

			// Timing variables
			auto worker_start_time = std::chrono::high_resolution_clock::now();
			int local_frames_processed = 0;

			// Variables for post-processing CUDA graph
			cudaGraph_t postprocessGraph = nullptr;
			cudaGraphExec_t postprocessGraphExec = nullptr;
			bool postGraphCaptured = false;

			// Process each image assigned to this worker
			for (int img_idx = start_idx; img_idx < end_idx; ++img_idx) {
				const torch::Tensor& img_tensor = img_tensors[img_idx];

				// Verify tensor dimensions
				if (img_tensor.dim() != 4 || img_tensor.size(0) != 1) {
					std::cerr << "Error: Invalid tensor dimensions for image " << img_idx
						<< ". Expected 4D tensor with batch size 1." << std::endl;
					continue;
				}

				try {
					// === PRE-ALLOCATION OF ALL MEMORY (BEFORE ANY GRAPH CAPTURE) ===

					// Set input dimensions (always batch size 1 for single-batch model)
					nvinfer1::Dims4 inputDims;
					inputDims.d[0] = 1;
					inputDims.d[1] = img_tensor.size(1);
					inputDims.d[2] = img_tensor.size(2);
					inputDims.d[3] = img_tensor.size(3);
					context->setBindingDimensions(0, inputDims);

					if (!context->allInputDimensionsSpecified()) {
						std::cerr << "Error: Not all input dimensions were specified for image " << img_idx << std::endl;
						continue;
					}

					// Allocate all input/output memory BEFORE any graph capture
					size_t input_size = img_tensor.numel();
					float* h_input = nullptr;
					void* d_input = nullptr;

					cudaError_t cuda_error = cudaMallocHost((void**)&h_input, input_size * sizeof(float));
					if (cuda_error != cudaSuccess) {
						std::cerr << "Error allocating host input memory: " << cudaGetErrorString(cuda_error) << std::endl;
						continue;
					}

					cuda_error = cudaMalloc(&d_input, input_size * sizeof(float));
					if (cuda_error != cudaSuccess) {
						std::cerr << "Error allocating device input memory: " << cudaGetErrorString(cuda_error) << std::endl;
						cudaFreeHost(h_input);
						continue;
					}

					// Copy input data to host buffer
					std::memcpy(h_input, img_tensor.data_ptr<float>(), input_size * sizeof(float));

					// Copy to device (outside any graph capture)
					cuda_error = cudaMemcpyAsync(d_input, h_input, input_size * sizeof(float),
						cudaMemcpyHostToDevice, inferStream);
					if (cuda_error != cudaSuccess) {
						std::cerr << "Error copying input to device: " << cudaGetErrorString(cuda_error) << std::endl;
						cudaFree(d_input);
						cudaFreeHost(h_input);
						continue;
					}

					// Set up bindings and allocate output memory
					std::vector<void*> bindings = { d_input };
					std::vector<void*> d_outputs;
					std::vector<float*> h_outputs;
					nvinfer1::Dims outputDims;

					// Setup output bindings - always get output binding info after setting input dimensions
					for (int i = 1; i < engine->getNbBindings(); ++i) {
						outputDims = context->getBindingDimensions(i);
						int outputSize = 1;
						for (int j = 0; j < outputDims.nbDims; ++j) {
							outputSize *= outputDims.d[j];
						}

						float* h_output = nullptr;
						void* d_output = nullptr;

						cuda_error = cudaMallocHost((void**)&h_output, outputSize * sizeof(float));
						if (cuda_error != cudaSuccess) {
							std::cerr << "Error allocating host output memory: " << cudaGetErrorString(cuda_error) << std::endl;
							for (size_t j = 0; j < h_outputs.size(); j++) {
								cudaFreeHost(h_outputs[j]);
								cudaFree(d_outputs[j]);
							}
							cudaFree(d_input);
							cudaFreeHost(h_input);
							break;
						}

						cuda_error = cudaMalloc(&d_output, outputSize * sizeof(float));
						if (cuda_error != cudaSuccess) {
							std::cerr << "Error allocating device output memory: " << cudaGetErrorString(cuda_error) << std::endl;
							cudaFreeHost(h_output);
							for (size_t j = 0; j < h_outputs.size(); j++) {
								cudaFreeHost(h_outputs[j]);
								cudaFree(d_outputs[j]);
							}
							cudaFree(d_input);
							cudaFreeHost(h_input);
							break;
						}

						h_outputs.push_back(h_output);
						d_outputs.push_back(d_output);
						bindings.push_back(d_output);
					}

					// Check if we hit an error in the binding setup loop
					if (h_outputs.size() != engine->getNbBindings() - 1) {
						continue; // Skip to next image if memory allocation failed
					}

					// Get output dimensions for post-processing
					int batch = outputDims.d[0]; // Should be 1
					int num_classes = outputDims.d[1];
					int height = outputDims.d[2];
					int width = outputDims.d[3];

					// Allocate memory for segmentation mask output
					unsigned char* d_argmax_output = nullptr;
					cuda_error = cudaMalloc(&d_argmax_output, height * width * sizeof(unsigned char));
					if (cuda_error != cudaSuccess) {
						std::cerr << "Error allocating argmax output memory: " << cudaGetErrorString(cuda_error) << std::endl;
						for (size_t j = 0; j < h_outputs.size(); j++) {
							cudaFreeHost(h_outputs[j]);
							cudaFree(d_outputs[j]);
						}
						cudaFree(d_input);
						cudaFreeHost(h_input);
						continue;
					}

					// === RUN TENSORRT INFERENCE (NO GRAPH CAPTURE) ===
					cudaEvent_t start, stop;
					cudaEventCreate(&start);
					cudaEventCreate(&stop);
					cudaEventRecord(start, inferStream);

					// Run TensorRT inference - NOT in a CUDA graph since it's not supported
					if (!context->enqueueV2(bindings.data(), inferStream, nullptr)) {
						std::cerr << "Error: TensorRT enqueueV2 failed for image " << img_idx << std::endl;
						cudaEventDestroy(start);
						cudaEventDestroy(stop);
						cudaFree(d_argmax_output);
						for (size_t j = 0; j < h_outputs.size(); j++) {
							cudaFreeHost(h_outputs[j]);
							cudaFree(d_outputs[j]);
						}
						cudaFree(d_input);
						cudaFreeHost(h_input);
						continue;
					}

					// Wait for inference to complete
					cudaStreamSynchronize(inferStream);

					// === POST-PROCESSING WITH CUDA GRAPH ===
					// Try to create and execute a CUDA graph for post-processing only

					// Only capture the graph on the first successful run
					if (!postGraphCaptured) {
						try {
							// Start graph capture for post-processing only
							cuda_error = cudaStreamBeginCapture(postStream, cudaStreamCaptureModeRelaxed);
							if (cuda_error != cudaSuccess) {
								throw std::runtime_error(std::string("Failed to begin graph capture: ") +
									cudaGetErrorString(cuda_error));
							}

							// Add argmax kernel to the graph
							launchArgmaxKernel(
								static_cast<float*>(d_outputs.back()),
								d_argmax_output,
								1, // batch size is 1
								num_classes,
								height,
								width,
								postStream
							);

							// End capture and instantiate the graph
							cuda_error = cudaStreamEndCapture(postStream, &postprocessGraph);
							if (cuda_error != cudaSuccess) {
								throw std::runtime_error(std::string("Failed to end graph capture: ") +
									cudaGetErrorString(cuda_error));
							}

							cuda_error = cudaGraphInstantiate(&postprocessGraphExec, postprocessGraph, nullptr, nullptr, 0);
							if (cuda_error != cudaSuccess) {
								throw std::runtime_error(std::string("Failed to instantiate graph: ") +
									cudaGetErrorString(cuda_error));
							}

							postGraphCaptured = true;
							graph_usage[t] = true;
							std::cout << "Worker " << t << ": Successfully created post-processing graph" << std::endl;
						}
						catch (const std::exception& e) {
							std::cerr << "Worker " << t << ": CUDA Graph capture for post-processing failed: "
								<< e.what() << std::endl;
							std::cerr << "Falling back to normal execution mode for post-processing" << std::endl;

							// Clean up any partial graph resources
							if (postprocessGraph) {
								cudaGraphDestroy(postprocessGraph);
								postprocessGraph = nullptr;
							}
							if (postprocessGraphExec) {
								cudaGraphExecDestroy(postprocessGraphExec);
								postprocessGraphExec = nullptr;
							}
						}
					}

					// Execute post-processing (with graph if available, or regular execution)
					if (postGraphCaptured && postprocessGraphExec) {
						// Execute the captured post-processing graph
						cuda_error = cudaGraphLaunch(postprocessGraphExec, postStream);
						if (cuda_error != cudaSuccess) {
							std::cerr << "Error launching post-processing graph: "
								<< cudaGetErrorString(cuda_error) << std::endl;
							// Fall back to regular kernel launch
							launchArgmaxKernel(
								static_cast<float*>(d_outputs.back()),
								d_argmax_output,
								1, // batch size is 1
								num_classes,
								height,
								width,
								postStream
							);
						}
					}
					else {
						// Fall back to regular kernel launch if graph not available
						launchArgmaxKernel(
							static_cast<float*>(d_outputs.back()),
							d_argmax_output,
							1, // batch size is 1
							num_classes,
							height,
							width,
							postStream
						);
					}

					// Wait for post-processing to complete
					cudaStreamSynchronize(postStream);

					// Copy results back to host
					unsigned char* h_argmax_output = new unsigned char[height * width];
					cuda_error = cudaMemcpyAsync(h_argmax_output, d_argmax_output,
						height * width * sizeof(unsigned char),
						cudaMemcpyDeviceToHost, postStream);
					if (cuda_error != cudaSuccess) {
						std::cerr << "Error copying results to host: " << cudaGetErrorString(cuda_error) << std::endl;
						delete[] h_argmax_output;
						cudaEventDestroy(start);
						cudaEventDestroy(stop);
						cudaFree(d_argmax_output);
						for (size_t j = 0; j < h_outputs.size(); j++) {
							cudaFreeHost(h_outputs[j]);
							cudaFree(d_outputs[j]);
						}
						cudaFree(d_input);
						cudaFreeHost(h_input);
						continue;
					}
					cudaStreamSynchronize(postStream);

					// Record end timing
					cudaEventRecord(stop, inferStream);
					cudaStreamSynchronize(inferStream);
					float milliseconds = 0;
					cudaEventElapsedTime(&milliseconds, start, stop);

					// Create OpenCV Mat from the segmentation mask
					cv::Mat result(height, width, CV_8UC1);
					std::memcpy(result.data, h_argmax_output, height * width * sizeof(unsigned char));

					// Update the results array
					{
						std::lock_guard<std::mutex> lock(resultMutex);
						results[img_idx] = result.clone();
					}

					// Update local counters
					local_frames_processed++;

					// Cleanup per-image resources
					delete[] h_argmax_output;
					cudaFree(d_argmax_output);
					cudaEventDestroy(start);
					cudaEventDestroy(stop);
					cudaFreeHost(h_input);
					cudaFree(d_input);
					for (auto ptr : h_outputs) {
						cudaFreeHost(ptr);
					}
					for (auto dptr : d_outputs) {
						cudaFree(dptr);
					}
				}
				catch (const std::exception& e) {
					std::cerr << "Error processing image " << img_idx << ": " << e.what() << std::endl;
					// Continue with next image
				}
			}

			// Calculate total processing time for this worker
			auto worker_end_time = std::chrono::high_resolution_clock::now();
			double total_seconds = std::chrono::duration<double>(worker_end_time - worker_start_time).count();

			// Update worker statistics
			{
				std::lock_guard<std::mutex> lock(resultMutex);
				processing_times[t] = total_seconds;
				frames_processed[t] = local_frames_processed;
			}

			// Clean up worker resources
			if (postprocessGraphExec) {
				cudaGraphExecDestroy(postprocessGraphExec);
			}
			if (postprocessGraph) {
				cudaGraphDestroy(postprocessGraph);
			}
			cudaStreamDestroy(inferStream);
			cudaStreamDestroy(postStream);
			context->destroy();
			});
	}

	// Wait for all worker threads to complete
	for (auto& t : threads) {
		t.join();
	}

	// Summarize performance statistics
	std::cout << "\n=== Performance Summary ===" << std::endl;
	std::cout << "Total images processed: " << num_images << std::endl;

	double total_processing_time = 0.0;
	int total_processed = 0;
	int graph_workers = 0;

	for (int t = 0; t < num_streams; ++t) {
		std::cout << "Worker " << t << ": " << frames_processed[t] << " frames in "
			<< processing_times[t] << " seconds";

		if (frames_processed[t] > 0) {
			double fps = frames_processed[t] / processing_times[t];
			std::cout << " (" << fps << " fps)";
		}

		std::cout << (graph_usage[t] ? " [with CUDA Graph]" : " [without CUDA Graph]") << std::endl;

		total_processing_time += processing_times[t];
		total_processed += frames_processed[t];
		if (graph_usage[t]) graph_workers++;
	}

	std::cout << "Average processing time per worker: " << total_processing_time / num_streams << " seconds" << std::endl;
	std::cout << "Effective overall throughput: " << num_images / (total_processing_time / num_streams) << " fps" << std::endl;
	std::cout << "Workers using CUDA Graph: " << graph_workers << " of " << num_streams << std::endl;
	std::cout << "============================" << std::endl;

	// Clean up global resources
	engine->destroy();
	runtime->destroy();

	return results;
}

//--------------------------------------------------------------------------
// Measure Super-Resolution Inference
//--------------------------------------------------------------------------
void TRTInference::measure_trt_performance(const string& trt_plan,
	const string& original_image_path,
	torch::Tensor img_tensor,
	int num_trials,
	bool compare_img_bool) {

	std::cout << "STARTING measure_trt_performance" << std::endl;

	TRTGeneration::CustomLogger myLogger;
	IRuntime* runtime = createInferRuntime(myLogger);

	ifstream planFile(trt_plan, ios::binary);
	vector<char> plan((istreambuf_iterator<char>(planFile)), istreambuf_iterator<char>());

	ICudaEngine* engine = runtime->deserializeCudaEngine(plan.data(), plan.size());
	IExecutionContext* context = engine->createExecutionContext();
	if (!engine || !context) {
		cerr << "Failed to deserialize engine or create execution context." << endl;
		exit(EXIT_FAILURE);
	}

	int input_size = img_tensor.numel();
	float* h_input;
	cudaMallocHost((void**)&h_input, input_size * sizeof(float));

	nvinfer1::Dims4 inputDims;
	nvinfer1::Dims4 outputDims;

	std::vector<int> outputBindingIndices;
	std::vector<std::string> outputTensorNames;
	for (int i = 1; i < engine->getNbBindings(); ++i) {
		outputBindingIndices.push_back(i);
		outputTensorNames.push_back(engine->getBindingName(i));
	}

	inputDims.d[0] = img_tensor.size(0);
	inputDims.d[1] = img_tensor.size(1);
	inputDims.d[2] = img_tensor.size(2);
	inputDims.d[3] = img_tensor.size(3);
	context->setBindingDimensions(0, inputDims);

	std::vector<void*> d_outputs;
	std::vector<float*> h_outputs;
	std::vector<void*> bindings;

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	void* d_input;
	cudaMalloc(&d_input, input_size * sizeof(float));

	std::memcpy(h_input, img_tensor.data_ptr<float>(), input_size * sizeof(float));
	cudaError_t memcpyStatus = cudaMemcpyAsync(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice, stream);
	if (memcpyStatus != cudaSuccess) {
		cerr << "CUDA error (cudaMemcpyAsync): " << cudaGetErrorString(memcpyStatus) << endl;
		exit(EXIT_FAILURE);
	}
	bindings.push_back(d_input);

	for (int i : outputBindingIndices) {
		nvinfer1::Dims dims = context->getBindingDimensions(i);
		outputDims.nbDims = dims.nbDims;
		for (int j = 0; j < dims.nbDims; ++j) {
			outputDims.d[j] = dims.d[j];
			if (outputDims.d[j] < 0) {
				outputDims.d[j] = inputDims.d[j];
			}
		}
		int outputSize = outputDims.d[0] * outputDims.d[1] * outputDims.d[2] * outputDims.d[3];
		float* h_output = new float[outputSize];
		void* d_output;
		if (cudaMalloc(&d_output, outputSize * sizeof(float)) != cudaSuccess) {
			cerr << "Device memory allocation failed" << endl;
			exit(EXIT_FAILURE);
		}
		h_outputs.push_back(h_output);
		d_outputs.push_back(d_output);
		bindings.push_back(d_output);
	}

	vector<float> latencies;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	for (int i = 0; i < 10; ++i) {
		context->enqueueV2(bindings.data(), stream, nullptr);
	}

	cudaEventRecord(start, stream);
	for (int i = 0; i < num_trials; ++i) {
		if (!context->enqueueV2(bindings.data(), stream, nullptr)) {
			cerr << "TensorRT enqueueV2 failed!" << endl;
			exit(EXIT_FAILURE);
		}
	}
	cudaEventRecord(stop, stream);
	cudaEventSynchronize(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	latencies.push_back(milliseconds);

	float* last_h_output = h_outputs.back();
	int last_output_size = outputDims.d[0] * outputDims.d[1] * outputDims.d[2] * outputDims.d[3];
	cudaMemcpyAsync(last_h_output, d_outputs.back(), last_output_size * sizeof(float), cudaMemcpyDeviceToHost, stream);
	cudaStreamSynchronize(stream);

	float min_val = *std::min_element(last_h_output, last_h_output + last_output_size);
	float max_val = *std::max_element(last_h_output, last_h_output + last_output_size);
	float avg_val = std::accumulate(last_h_output, last_h_output + last_output_size, 0.0f) / last_output_size;
	cout << "Last Output Tensor - Min: " << min_val << ", Max: " << max_val << ", Avg: " << avg_val << endl;

	float average_latency = std::accumulate(latencies.begin(), latencies.end(), 0.0f) / num_trials;
	cout << "TRT - Average Latency over " << num_trials << " trials: " << average_latency << " ms" << endl;
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cv::Mat image_data(outputDims.d[2], outputDims.d[3], CV_32F, last_h_output);
	cv::Mat clipped_image_data;
	cv::min(image_data, 1.0, clipped_image_data);
	cv::max(clipped_image_data, 0.0, clipped_image_data);
	clipped_image_data *= 255;
	clipped_image_data.convertTo(clipped_image_data, CV_8U);

	cudaFreeHost(h_input);
	for (float* h_output : h_outputs) {
		delete[] h_output;
	}
	cudaFree(d_input);
	for (void* d_output : d_outputs) {
		cudaFree(d_output);
	}
	context->destroy();
	engine->destroy();
	runtime->destroy();
	cudaStreamDestroy(stream);
}

//--------------------------------------------------------------------------------------
// Processes multiple images in parallel using a preloaded TRT engine
//--------------------------------------------------------------------------------------
std::vector<cv::Mat> TRTInference::measure_segmentation_trt_performance_single_batch_parallel_preloaded(
	nvinfer1::ICudaEngine* engine, const std::vector<torch::Tensor>& img_tensors, int num_streams) {

	if (!engine) {
		std::cerr << "Error: Null engine pointer provided" << std::endl;
		return {};
	}

	std::cout << "Starting optimized parallel inference with preloaded engine" << std::endl;

	// Number of images to process
	int num_images = img_tensors.size();
	if (num_images == 0) {
		return {};
	}

	// Results container
	std::vector<cv::Mat> results(num_images);
	std::mutex resultMutex;
	std::vector<std::thread> threads;

	// Calculate images per worker thread
	int images_per_thread = (num_images + num_streams - 1) / num_streams;

	// Performance metrics for reporting
	std::vector<double> processing_times(num_streams, 0.0);
	std::vector<int> frames_processed(num_streams, 0);
	std::vector<bool> graph_usage(num_streams, false);

	// Launch parallel worker threads
	for (int t = 0; t < num_streams; ++t) {
		threads.emplace_back([&, t]() {
			// Calculate the range of images for this worker
			int start_idx = t * images_per_thread;
			int end_idx = std::min(start_idx + images_per_thread, num_images);

			if (start_idx >= num_images) {
				return; // No images for this worker
			}

			// Create execution context for this worker
			IExecutionContext* context = engine->createExecutionContext();
			if (!context) {
				std::cerr << "Error: Failed to create execution context for worker " << t << std::endl;
				return;
			}

			// Create CUDA streams for this worker - separate streams for inference and post-processing
			cudaStream_t inferStream, postStream;
			checkCudaErrors(cudaStreamCreateWithFlags(&inferStream, cudaStreamNonBlocking));
			checkCudaErrors(cudaStreamCreateWithFlags(&postStream, cudaStreamNonBlocking));

			// Timing variables
			auto worker_start_time = std::chrono::high_resolution_clock::now();
			int local_frames_processed = 0;

			// Variables for post-processing CUDA graph
			cudaGraph_t postprocessGraph = nullptr;
			cudaGraphExec_t postprocessGraphExec = nullptr;
			bool postGraphCaptured = false;

			// Process each image assigned to this worker
			for (int img_idx = start_idx; img_idx < end_idx; ++img_idx) {
				const torch::Tensor& img_tensor = img_tensors[img_idx];

				// Verify tensor dimensions
				if (img_tensor.dim() != 4 || img_tensor.size(0) != 1) {
					std::cerr << "Error: Invalid tensor dimensions for image " << img_idx
						<< ". Expected 4D tensor with batch size 1." << std::endl;
					continue;
				}

				try {
					// === PRE-ALLOCATION OF ALL MEMORY (BEFORE ANY GRAPH CAPTURE) ===

					// Set input dimensions (always batch size 1 for single-batch model)
					nvinfer1::Dims4 inputDims;
					inputDims.d[0] = 1;
					inputDims.d[1] = img_tensor.size(1);
					inputDims.d[2] = img_tensor.size(2);
					inputDims.d[3] = img_tensor.size(3);
					context->setBindingDimensions(0, inputDims);

					if (!context->allInputDimensionsSpecified()) {
						std::cerr << "Error: Not all input dimensions were specified for image " << img_idx << std::endl;
						continue;
					}

					// Allocate all input/output memory BEFORE any graph capture
					size_t input_size = img_tensor.numel();
					float* h_input = nullptr;
					void* d_input = nullptr;

					cudaError_t cuda_error = cudaMallocHost((void**)&h_input, input_size * sizeof(float));
					if (cuda_error != cudaSuccess) {
						std::cerr << "Error allocating host input memory: " << cudaGetErrorString(cuda_error) << std::endl;
						continue;
					}

					cuda_error = cudaMalloc(&d_input, input_size * sizeof(float));
					if (cuda_error != cudaSuccess) {
						std::cerr << "Error allocating device input memory: " << cudaGetErrorString(cuda_error) << std::endl;
						cudaFreeHost(h_input);
						continue;
					}

					// Copy input data to host buffer
					std::memcpy(h_input, img_tensor.data_ptr<float>(), input_size * sizeof(float));

					// Copy to device (outside any graph capture)
					cuda_error = cudaMemcpyAsync(d_input, h_input, input_size * sizeof(float),
						cudaMemcpyHostToDevice, inferStream);
					if (cuda_error != cudaSuccess) {
						std::cerr << "Error copying input to device: " << cudaGetErrorString(cuda_error) << std::endl;
						cudaFree(d_input);
						cudaFreeHost(h_input);
						continue;
					}

					// Set up bindings and allocate output memory
					std::vector<void*> bindings = { d_input };
					std::vector<void*> d_outputs;
					std::vector<float*> h_outputs;
					nvinfer1::Dims outputDims;

					// Setup output bindings - always get output binding info after setting input dimensions
					for (int i = 1; i < engine->getNbBindings(); ++i) {
						outputDims = context->getBindingDimensions(i);
						int outputSize = 1;
						for (int j = 0; j < outputDims.nbDims; ++j) {
							outputSize *= outputDims.d[j];
						}

						float* h_output = nullptr;
						void* d_output = nullptr;

						cuda_error = cudaMallocHost((void**)&h_output, outputSize * sizeof(float));
						if (cuda_error != cudaSuccess) {
							std::cerr << "Error allocating host output memory: " << cudaGetErrorString(cuda_error) << std::endl;
							for (size_t j = 0; j < h_outputs.size(); j++) {
								cudaFreeHost(h_outputs[j]);
								cudaFree(d_outputs[j]);
							}
							cudaFree(d_input);
							cudaFreeHost(h_input);
							break;
						}

						cuda_error = cudaMalloc(&d_output, outputSize * sizeof(float));
						if (cuda_error != cudaSuccess) {
							std::cerr << "Error allocating device output memory: " << cudaGetErrorString(cuda_error) << std::endl;
							cudaFreeHost(h_output);
							for (size_t j = 0; j < h_outputs.size(); j++) {
								cudaFreeHost(h_outputs[j]);
								cudaFree(d_outputs[j]);
							}
							cudaFree(d_input);
							cudaFreeHost(h_input);
							break;
						}

						h_outputs.push_back(h_output);
						d_outputs.push_back(d_output);
						bindings.push_back(d_output);
					}

					// Check if we hit an error in the binding setup loop
					if (h_outputs.size() != engine->getNbBindings() - 1) {
						continue; // Skip to next image if memory allocation failed
					}

					// Get output dimensions for post-processing
					int batch = outputDims.d[0]; // Should be 1
					int num_classes = outputDims.d[1];
					int height = outputDims.d[2];
					int width = outputDims.d[3];

					// Allocate memory for segmentation mask output
					unsigned char* d_argmax_output = nullptr;
					cuda_error = cudaMalloc(&d_argmax_output, height * width * sizeof(unsigned char));
					if (cuda_error != cudaSuccess) {
						std::cerr << "Error allocating argmax output memory: " << cudaGetErrorString(cuda_error) << std::endl;
						for (size_t j = 0; j < h_outputs.size(); j++) {
							cudaFreeHost(h_outputs[j]);
							cudaFree(d_outputs[j]);
						}
						cudaFree(d_input);
						cudaFreeHost(h_input);
						continue;
					}

					// === RUN TENSORRT INFERENCE (NO GRAPH CAPTURE) ===
					cudaEvent_t start, stop;
					cudaEventCreate(&start);
					cudaEventCreate(&stop);
					cudaEventRecord(start, inferStream);

					// Run TensorRT inference - NOT in a CUDA graph since it's not supported
					if (!context->enqueueV2(bindings.data(), inferStream, nullptr)) {
						std::cerr << "Error: TensorRT enqueueV2 failed for image " << img_idx << std::endl;
						cudaEventDestroy(start);
						cudaEventDestroy(stop);
						cudaFree(d_argmax_output);
						for (size_t j = 0; j < h_outputs.size(); j++) {
							cudaFreeHost(h_outputs[j]);
							cudaFree(d_outputs[j]);
						}
						cudaFree(d_input);
						cudaFreeHost(h_input);
						continue;
					}

					// Wait for inference to complete
					cudaStreamSynchronize(inferStream);

					// === POST-PROCESSING WITH CUDA GRAPH ===
					// Try to create and execute a CUDA graph for post-processing only

					// Only capture the graph on the first successful run
					if (!postGraphCaptured) {
						try {
							// Start graph capture for post-processing only
							cuda_error = cudaStreamBeginCapture(postStream, cudaStreamCaptureModeRelaxed);
							if (cuda_error != cudaSuccess) {
								throw std::runtime_error(std::string("Failed to begin graph capture: ") +
									cudaGetErrorString(cuda_error));
							}

							// Add argmax kernel to the graph
							launchArgmaxKernel(
								static_cast<float*>(d_outputs.back()),
								d_argmax_output,
								1, // batch size is 1
								num_classes,
								height,
								width,
								postStream
							);

							// End capture and instantiate the graph
							cuda_error = cudaStreamEndCapture(postStream, &postprocessGraph);
							if (cuda_error != cudaSuccess) {
								throw std::runtime_error(std::string("Failed to end graph capture: ") +
									cudaGetErrorString(cuda_error));
							}

							cuda_error = cudaGraphInstantiate(&postprocessGraphExec, postprocessGraph, nullptr, nullptr, 0);
							if (cuda_error != cudaSuccess) {
								throw std::runtime_error(std::string("Failed to instantiate graph: ") +
									cudaGetErrorString(cuda_error));
							}

							postGraphCaptured = true;
							graph_usage[t] = true;
							std::cout << "Worker " << t << ": Successfully created post-processing graph" << std::endl;
						}
						catch (const std::exception& e) {
							std::cerr << "Worker " << t << ": CUDA Graph capture for post-processing failed: "
								<< e.what() << std::endl;
							std::cerr << "Falling back to normal execution mode for post-processing" << std::endl;

							// Clean up any partial graph resources
							if (postprocessGraph) {
								cudaGraphDestroy(postprocessGraph);
								postprocessGraph = nullptr;
							}
							if (postprocessGraphExec) {
								cudaGraphExecDestroy(postprocessGraphExec);
								postprocessGraphExec = nullptr;
							}
						}
					}

					// Execute post-processing (with graph if available, or regular execution)
					if (postGraphCaptured && postprocessGraphExec) {
						// Execute the captured post-processing graph
						cuda_error = cudaGraphLaunch(postprocessGraphExec, postStream);
						if (cuda_error != cudaSuccess) {
							std::cerr << "Error launching post-processing graph: "
								<< cudaGetErrorString(cuda_error) << std::endl;
							// Fall back to regular kernel launch
							launchArgmaxKernel(
								static_cast<float*>(d_outputs.back()),
								d_argmax_output,
								1, // batch size is 1
								num_classes,
								height,
								width,
								postStream
							);
						}
					}
					else {
						// Fall back to regular kernel launch if graph not available
						launchArgmaxKernel(
							static_cast<float*>(d_outputs.back()),
							d_argmax_output,
							1, // batch size is 1
							num_classes,
							height,
							width,
							postStream
						);
					}

					// Wait for post-processing to complete
					cudaStreamSynchronize(postStream);

					// Copy results back to host
					unsigned char* h_argmax_output = new unsigned char[height * width];
					cuda_error = cudaMemcpyAsync(h_argmax_output, d_argmax_output,
						height * width * sizeof(unsigned char),
						cudaMemcpyDeviceToHost, postStream);
					if (cuda_error != cudaSuccess) {
						std::cerr << "Error copying results to host: " << cudaGetErrorString(cuda_error) << std::endl;
						delete[] h_argmax_output;
						cudaEventDestroy(start);
						cudaEventDestroy(stop);
						cudaFree(d_argmax_output);
						for (size_t j = 0; j < h_outputs.size(); j++) {
							cudaFreeHost(h_outputs[j]);
							cudaFree(d_outputs[j]);
						}
						cudaFree(d_input);
						cudaFreeHost(h_input);
						continue;
					}
					cudaStreamSynchronize(postStream);

					// Record end timing
					cudaEventRecord(stop, inferStream);
					cudaStreamSynchronize(inferStream);
					float milliseconds = 0;
					cudaEventElapsedTime(&milliseconds, start, stop);

					// Create OpenCV Mat from the segmentation mask
					cv::Mat result(height, width, CV_8UC1);
					std::memcpy(result.data, h_argmax_output, height * width * sizeof(unsigned char));

					// Update the results array
					{
						std::lock_guard<std::mutex> lock(resultMutex);
						results[img_idx] = result.clone();
					}

					// Update local counters
					local_frames_processed++;

					// Cleanup per-image resources
					delete[] h_argmax_output;
					cudaFree(d_argmax_output);
					cudaEventDestroy(start);
					cudaEventDestroy(stop);
					cudaFreeHost(h_input);
					cudaFree(d_input);
					for (auto ptr : h_outputs) {
						cudaFreeHost(ptr);
					}
					for (auto dptr : d_outputs) {
						cudaFree(dptr);
					}
				}
				catch (const std::exception& e) {
					std::cerr << "Error processing image " << img_idx << ": " << e.what() << std::endl;
					// Continue with next image
				}
			}

			// Calculate total processing time for this worker
			auto worker_end_time = std::chrono::high_resolution_clock::now();
			double total_seconds = std::chrono::duration<double>(worker_end_time - worker_start_time).count();

			// Update worker statistics
			{
				std::lock_guard<std::mutex> lock(resultMutex);
				processing_times[t] = total_seconds;
				frames_processed[t] = local_frames_processed;
			}

			// Clean up worker resources
			if (postprocessGraphExec) {
				cudaGraphExecDestroy(postprocessGraphExec);
			}
			if (postprocessGraph) {
				cudaGraphDestroy(postprocessGraph);
			}
			cudaStreamDestroy(inferStream);
			cudaStreamDestroy(postStream);
			context->destroy();
			});
	}

	// Wait for all worker threads to complete
	for (auto& t : threads) {
		t.join();
	}

	// Summarize performance statistics
	std::cout << "\n=== Performance Summary ===" << std::endl;
	std::cout << "Total images processed: " << num_images << std::endl;

	double total_processing_time = 0.0;
	int total_processed = 0;
	int graph_workers = 0;

	for (int t = 0; t < num_streams; ++t) {
		std::cout << "Worker " << t << ": " << frames_processed[t] << " frames in "
			<< processing_times[t] << " seconds";

		if (frames_processed[t] > 0) {
			double fps = frames_processed[t] / processing_times[t];
			std::cout << " (" << fps << " fps)";
		}

		std::cout << (graph_usage[t] ? " [with CUDA Graph]" : " [without CUDA Graph]") << std::endl;

		total_processing_time += processing_times[t];
		total_processed += frames_processed[t];
		if (graph_usage[t]) graph_workers++;
	}

	std::cout << "Average processing time per worker: " << total_processing_time / num_streams << " seconds" << std::endl;
	std::cout << "Effective overall throughput: " << num_images / (total_processing_time / num_streams) << " fps" << std::endl;
	std::cout << "Workers using CUDA Graph: " << graph_workers << " of " << num_streams << std::endl;
	std::cout << "============================" << std::endl;

	return results;
}