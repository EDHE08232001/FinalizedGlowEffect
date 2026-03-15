#include "TRTGeneration.hpp"


std::vector<int> TRTGeneration::gen_trt_plan(const std::string& onnxModelPath, const std::string& planFilePath,
                              const std::string& precision_mode, bool dynamic,
                              DynamicShapeRange& dynamicShapeRange)
{
    std::cout << "current dir: " << filesystem::current_path() << std::endl;
    std::vector<int> inputShape;
    // std::string folderPath = "../../../input_imgs/486";
    // std::string folderPath = "../../input_imgs/leftImg8bit/val";
    std::string folderPath = "C:/data_svn/ai_general/code_resource/Mitansh/input_imgs/input_imgs/leftImg8bit/val";
    //std::string folderPath = "E:/data_svn/ai_general/code_resource/Mitansh_work/input_imgs/leftImg8bit/val";
    std::vector<std::string> imagePaths = ImageProcessingUtil::getImagePaths(folderPath);
    
    CustomLogger myLogger;
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(myLogger);    
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    
    const int explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);

    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, myLogger);
    int verbosity = (int) nvinfer1::ILogger::Severity::kERROR;
    if (!parser->parseFromFile(onnxModelPath.c_str(), verbosity)){
        std::cerr << "ERROR: Failed to parse ONNX model" << std::endl;
        return inputShape;
    }

    std::vector<std::string> inputTensorNames;
    std::vector<std::string> outputTensorNames;
    int nbInputs = network->getNbInputs();
    int nbOutputs = network->getNbOutputs();

    if (nbInputs <= 0 || nbOutputs <= 0) {
        std::cerr << "Error: The network does not have any inputs or outputs." << std::endl;
    }

    for (int i = 0; i < nbInputs; ++i) {
        nvinfer1::ITensor* inputTensor = network->getInput(i);
        inputTensorNames.push_back(inputTensor->getName());
    }
    for (int i = 0; i < nbOutputs; ++i) {
        nvinfer1::ITensor* outputTensor = network->getOutput(i);
        outputTensorNames.push_back(outputTensor->getName());
    }

    // Get the dimensions of the first input tensor
    int batchSize, channelNum;
    if (dynamic) {
        // Get the dimensions from the dynamic shape range
        batchSize = dynamicShapeRange.optDims.d[0];
        channelNum = dynamicShapeRange.optDims.d[1];
    } else {
        // Get the dimensions from the input shape for static plans
        nvinfer1::ITensor* inputTensor = network->getInput(0);
        nvinfer1::Dims inputDims = inputTensor->getDimensions();
        batchSize = inputDims.d[0];
        channelNum = inputDims.d[1];
    }
    cout << "Selected INT8 Calibrator BatchSize: " << batchSize << endl;

    Int8EntropyCalibrator* calibrator = new Int8EntropyCalibrator(batchSize, channelNum, imagePaths, "./calibration_cache_file", inputTensorNames);
    
    if (precision_mode == "fp32") {
        config->setFlag(BuilderFlag::kTF32);
    }
    if (precision_mode == "fp16") {
        config->setFlag(BuilderFlag::kFP16);
    }
    else if (precision_mode == "int8") {
        // Check if the platform supports INT8
        if (!builder->platformHasFastInt8()) {
            std::cerr << "ERROR: INT8 not supported on this platform." << std::endl;
            return inputShape;
        }

        // falls back to fp16 or fp32 if some layer cant be converted to int8
        config->setFlag(BuilderFlag::kGPU_FALLBACK);
        config->setFlag(BuilderFlag::kINT8);
        config->setProfilingVerbosity(ProfilingVerbosity::kVERBOSE);

        config->setInt8Calibrator(calibrator);

        for (int i = 0; i < nbInputs; ++i) {
            nvinfer1::ITensor* inputTensor = network->getInput(i);
            nvinfer1::Dims currentDims = inputTensor->getDimensions();

            if (currentDims.d[0] == dynamicShapeRange.optDims.d[0] &&
                currentDims.d[1] == dynamicShapeRange.optDims.d[1] &&
                currentDims.d[2] == dynamicShapeRange.optDims.d[2] &&
                currentDims.d[3] == dynamicShapeRange.optDims.d[3]) 
            {
                inputTensor->setDynamicRange(-2.0f, 2.0f); // Example dynamic range for normalized input [-1,1]
            }
        } 
        for (int i = 0; i < nbOutputs; ++i) {
            nvinfer1::ITensor* outputTensor = network->getOutput(i);
            outputTensor->setDynamicRange(-2.0f, 2.0f); // Example dynamic range for the output [-1,1]
        }
    }
    
    // caching only disabled for benchmarking speed
    config->setFlag(BuilderFlag::kDISABLE_TIMING_CACHE);
    config->setMaxWorkspaceSize(4LL << 30); // 4gb
    config->setFlag(BuilderFlag::kSTRICT_TYPES);

    if (dynamic) {
        for (int i = 0; i < nbInputs; ++i) {
            nvinfer1::ITensor* inputTensor = network->getInput(i);
            nvinfer1::Dims inputDims = inputTensor->getDimensions();

            std::cout << "Dynamic ONNX Input Dims: " << inputDims.d[0] << ", " << inputDims.d[1] << ", " << inputDims.d[2] << ", " << inputDims.d[3] << std::endl;
            auto profile = builder->createOptimizationProfile();

            // Set the dimensions for each input tensor separately
            nvinfer1::Dims minDims = inputDims;
            nvinfer1::Dims optDims = inputDims;
            nvinfer1::Dims maxDims = inputDims;

            // Modify the dimensions based on the provided dynamic shape range
            for (int j = 0; j < inputDims.nbDims; ++j) {
                if (inputDims.d[j] == -1) {
                    minDims.d[j] = dynamicShapeRange.minDims.d[j];
                    optDims.d[j] = dynamicShapeRange.optDims.d[j];
                    maxDims.d[j] = dynamicShapeRange.maxDims.d[j];
                }
            }

            profile->setDimensions(inputTensor->getName(), nvinfer1::OptProfileSelector::kMIN, minDims);
            profile->setDimensions(inputTensor->getName(), nvinfer1::OptProfileSelector::kOPT, optDims);
            profile->setDimensions(inputTensor->getName(), nvinfer1::OptProfileSelector::kMAX, maxDims);
            config->addOptimizationProfile(profile);
        }
    }

    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    IHostMemory* serializedModel = engine->serialize();

    std::ofstream planFile(planFilePath, std::ios::binary);
    planFile.write(reinterpret_cast<const char*>(serializedModel->data()), serializedModel->size());
    planFile.close();

    std::cout << "Serialized plan generated." << std::endl;

    delete calibrator;
    parser->destroy();
    network->destroy();
    config->destroy();
    builder->destroy();
    engine->destroy();
    serializedModel->destroy();
    
    return inputShape;
}