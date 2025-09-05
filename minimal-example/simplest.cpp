#include <array>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include <vulkan/vulkan.h>

// A very simple compute shader

// Does nothing but copies data to the device, scales each element using 
// a compute shader and copies back the result to the host.
// This is a simplified version of https://vulkan-tutorial.com/Compute_Shader.

// Vulkan features introduced:
// * shader printf (debugPrintfEXT extension)
// * enabling validation layers
// * using specialization constants
// * using push constants
// * using shared storage buffers (SSBOs)

// Currently missing:
// * Setting workgroup size on the host
// * Uniform buffers
// * Overlapping data transfers and kernel calls
// * Semaphores
// * Fences
// ...

// We wish to define an input and output SSBO.
// The shader scales each element of the input buffer and writes it to the
// corresponding element of the output buffer.
const uint32_t n_buffers = 2;

// To enable shader printf to stdout either use vkconfig gui or set environment
// variables VK_LAYER_PRINTF_TO_STDOUT=1 VK_LAYER_PRINTF_ONLY_PRESET=1
// VK_LAYER_PRINTF_BUFFER_SIZE=65536

class Simplest {
public:
  void run() {
    initVulkan();
    compute();
    cleanup();
  }

private:
  uint32_t buffer_elems = 64;
  bool enableValidationLayers = true;

  VkInstance instance;
  const std::vector<const char *> validationLayers = {
      "VK_LAYER_KHRONOS_validation"};

  VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
  VkDevice device;
  uint32_t computeQueueFamilyIndex;
  VkQueue computeQueue;

  VkDescriptorSetLayout computeDescriptorSetLayout;
  VkPipelineLayout computePipelineLayout;
  VkPipeline computePipeline;

  VkCommandPool commandPool;
  std::vector<VkBuffer> shaderStorageBuffers;
  std::vector<VkDeviceMemory> shaderStorageBuffersMemory;

  VkDescriptorPool descriptorPool;
  std::vector<VkDescriptorSet> computeDescriptorSets;

  std::vector<VkCommandBuffer> computeCommandBuffers;

  typedef struct PushConstants {
    uint32_t input_length;
    float scale;
  } PushConstants;

  void initVulkan() {
    createInstance();
    pickPhysicalDevice();
    createLogicalDevice();
    createComputeDescriptorSetLayout();
    createComputePipeline();
    createCommandPool();
    createShaderStorageBuffers();
    createDescriptorPool();
    createComputeDescriptorSets();
    createComputeCommandBuffers();
  }

  bool checkValidationLayerSupport() {
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    for (const char *layerName : validationLayers) {
      bool layerFound = false;

      for (const auto &layerProperties : availableLayers) {
        if (strcmp(layerName, layerProperties.layerName) == 0) {
          layerFound = true;
          break;
        }
      }

      if (!layerFound) {
        return false;
      }
    }

    return true;
  }

  void createInstance() {
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Compute shader minimal example";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    // 1.1 needed for shader printf
    appInfo.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    // Manually enable validation
    bool validationLayerSupport = checkValidationLayerSupport();

    if (!validationLayerSupport) {
      std::cout << "Requested validation layers are not supported!"
                << std::endl;
    }

    if (enableValidationLayers && validationLayerSupport) {
      createInfo.enabledLayerCount =
          static_cast<uint32_t>(validationLayers.size());
      createInfo.ppEnabledLayerNames = validationLayers.data();
      createInfo.pNext = nullptr;
    } else {
      createInfo.enabledLayerCount = 0;
      createInfo.pNext = nullptr;
    }

    if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
      throw std::runtime_error("failed to create instance!");
    }
  }

  void pickPhysicalDevice() {
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

    if (deviceCount == 0) {
      throw std::runtime_error("failed to find GPUs with Vulkan support!");
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    for (const auto &device : devices) {
      // Check device properties...
      physicalDevice = device;
      break;
    }

    if (physicalDevice == VK_NULL_HANDLE) {
      throw std::runtime_error("failed to find a suitable GPU!");
    }
  }

  void createLogicalDevice() {
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount,
                                             nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount,
                                             queueFamilies.data());

    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    float queuePriority = 1.0f;
    uint32_t computeQueueFamilyIndex = 0;
    // Get the index of the first queue family that can do compute.
    for (const auto &queueFamily : queueFamilies) {
      if (queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT) {
        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = computeQueueFamilyIndex;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(queueCreateInfo);
        break;
      }
      computeQueueFamilyIndex += 1;
    }

    VkPhysicalDeviceFeatures deviceFeatures{};

    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

    createInfo.queueCreateInfoCount =
        static_cast<uint32_t>(queueCreateInfos.size());
    createInfo.pQueueCreateInfos = queueCreateInfos.data();
    createInfo.pEnabledFeatures = &deviceFeatures;
    createInfo.enabledExtensionCount = 0;

    if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create logical device!");
    }

    vkGetDeviceQueue(device, computeQueueFamilyIndex, 0, &computeQueue);
  }

  void createComputeDescriptorSetLayout() {
    // We wish to create two SSBOs, the input will be bound to the first slot
    // and the output to the second. The actual buffer addresses will be written
    // to the descriptor set in createComputeDescriptorSets().
    std::array<VkDescriptorSetLayoutBinding, n_buffers> layoutBindings{};

    // Input buffer
    layoutBindings[0].binding = 0;
    layoutBindings[0].descriptorCount = 1;
    layoutBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    layoutBindings[0].pImmutableSamplers = nullptr;
    layoutBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    // Output buffer
    layoutBindings[1].binding = 1;
    layoutBindings[1].descriptorCount = 1;
    layoutBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    layoutBindings[1].pImmutableSamplers = nullptr;
    layoutBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = (uint32_t)layoutBindings.size();
    layoutInfo.pBindings = layoutBindings.data();

    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr,
                                    &computeDescriptorSetLayout) !=
        VK_SUCCESS) {
      throw std::runtime_error(
          "failed to create compute descriptor set layout!");
    }
  }

  void createDescriptorPool() {
    uint32_t L = n_buffers; // Total number of SSBOs
    uint32_t M = 1;         // Maximum number of sets.

    // The descriptor pool will have only 2 instances of SSBOs.
    std::array<VkDescriptorPoolSize, 1> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; // Descriptor type
    poolSizes[0].descriptorCount = L; // Number of descriptors of this type

    VkDescriptorPoolCreateInfo descriptorPoolCreateInfo{};
    descriptorPoolCreateInfo.sType =
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptorPoolCreateInfo.poolSizeCount =
        (uint32_t)poolSizes.size(); // Number of deescriptor pool size entries
    descriptorPoolCreateInfo.pPoolSizes = poolSizes.data();
    descriptorPoolCreateInfo.maxSets = M;

    if (vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, nullptr,
                               &descriptorPool) != VK_SUCCESS) {
      throw std::runtime_error("failed to create descriptor pool!");
    }
  }

  void createComputeDescriptorSets() {
    // Number of descriptor sets to allocate (1 set is enough)
    const uint32_t N = 1;
    std::array<VkDescriptorSetLayout, N> descriptorSetLayouts = {
        computeDescriptorSetLayout};
    VkDescriptorSetAllocateInfo descriptorSetAllocateInfo{};
    descriptorSetAllocateInfo.sType =
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptorSetAllocateInfo.descriptorPool =
        descriptorPool; // Which pool is used to allocate from
    descriptorSetAllocateInfo.descriptorSetCount =
        (uint32_t)descriptorSetLayouts.size(); // size of the layouts
    descriptorSetAllocateInfo.pSetLayouts = descriptorSetLayouts.data();

    // The descriptor set will be put into the computeDescriptorSets array.
    computeDescriptorSets.resize(N);
    vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo,
                             computeDescriptorSets.data());

    // We need to write two buffer metadata into the descriptor set
    std::array<VkWriteDescriptorSet, n_buffers> descriptorWrites{};

    // Write the buffer address to the descriptor of the input.
    VkDescriptorBufferInfo bi0{};
    bi0.buffer = shaderStorageBuffers[0];
    bi0.offset = 0;
    bi0.range = VK_WHOLE_SIZE;

    descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[0].dstSet = computeDescriptorSets[0];
    descriptorWrites[0].dstBinding = 0;
    descriptorWrites[0].dstArrayElement = 0;
    descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[0].descriptorCount = 1;
    descriptorWrites[0].pBufferInfo = &bi0;

    // Write the buffer address to the descriptor of the output.
    VkDescriptorBufferInfo bi1{};
    bi1.buffer = shaderStorageBuffers[1];
    bi1.offset = 0;
    bi1.range = VK_WHOLE_SIZE;

    descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[1].dstSet = computeDescriptorSets[0];
    descriptorWrites[1].dstBinding = 1; // Binding id
    descriptorWrites[1].dstArrayElement =
        0; // Update from the first index (important if descriptor array is
           // used)
    descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[1].descriptorCount =
        1; // Update 1 element (important if descriptor array is used)
    descriptorWrites[1].pBufferInfo = &bi1;

    vkUpdateDescriptorSets(device, descriptorWrites.size(),
                           descriptorWrites.data(), 0, nullptr);
  }

  VkShaderModule createShaderModule(const std::vector<char> &code) {
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t *>(code.data());

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create shader module!");
    }

    return shaderModule;
  }

  static std::vector<char> readFile(const std::string &filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
      throw std::runtime_error("failed to open file!");
    }

    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);

    file.close();

    return buffer;
  }

  void createComputePipeline() {
    // Load compiled shader
    auto computeShaderCode = readFile("shaders/simplest.spv");
    VkShaderModule computeShaderModule = createShaderModule(computeShaderCode);

    // Add specialization constants (burned into the pipeline, cannot change
    // between kernel launches)
    uint32_t num_entry = 1;
    int blockSize = 128;

    // Fill with input constants to be passed to the shader.

    VkSpecializationMapEntry entry = {};
    entry.constantID = 0;
    entry.offset = 0;
    entry.size = sizeof(int);

    VkSpecializationInfo specializationInfo = {};
    specializationInfo.mapEntryCount = num_entry;
    specializationInfo.pMapEntries = &entry;
    specializationInfo.dataSize = sizeof(int);
    specializationInfo.pData = &blockSize;

    // Configure shader compilation
    VkPipelineShaderStageCreateInfo computeShaderStageInfo{};
    computeShaderStageInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    computeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    computeShaderStageInfo.module = computeShaderModule;
    computeShaderStageInfo.pSpecializationInfo = &specializationInfo;
    computeShaderStageInfo.pName = "main";

    // Define push constants layout (we can pass a 64-bit integer and a float in
    // each kernel launch)
    VkPushConstantRange pushConstantRange = {};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(uint32_t) + sizeof(float);

    // Build pipeline layout: bind descriptor layouts
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &computeDescriptorSetLayout;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr,
                               &computePipelineLayout) != VK_SUCCESS) {
      throw std::runtime_error("failed to create compute pipeline layout!");
    }

    // Build pipeline: link pipeline layout and shader compilation instructions
    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.layout = computePipelineLayout;
    pipelineInfo.stage = computeShaderStageInfo;

    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo,
                                 nullptr, &computePipeline) != VK_SUCCESS) {
      throw std::runtime_error("failed to create compute pipeline!");
    }

    vkDestroyShaderModule(device, computeShaderModule, nullptr);
  }

  void createCommandPool() {
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    // Useful if we want to reuse the command buffers (e.g. doing multiple
    // computations).
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = computeQueueFamilyIndex;

    if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create graphics command pool!");
    }
  }

  uint32_t findMemoryType(uint32_t typeFilter,
                          VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
      if ((typeFilter & (1 << i)) &&
          (memProperties.memoryTypes[i].propertyFlags & properties) ==
              properties) {
        return i;
      }
    }

    throw std::runtime_error("failed to find suitable memory type!");
  }

  void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                    VkMemoryPropertyFlags properties, VkBuffer &buffer,
                    VkDeviceMemory &bufferMemory) {
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
      throw std::runtime_error("failed to create buffer!");
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex =
        findMemoryType(memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to allocate buffer memory!");
    }

    vkBindBufferMemory(device, buffer, bufferMemory, 0);
  }

  void copyBuffer(VkBuffer dstBuffer, VkBuffer srcBuffer, VkDeviceSize size) {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer copyCommandBuffer;
    vkAllocateCommandBuffers(device, &allocInfo, &copyCommandBuffer);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(copyCommandBuffer, &beginInfo);

    VkBufferCopy copyRegion{};
    copyRegion.size = size;
    vkCmdCopyBuffer(copyCommandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

    vkEndCommandBuffer(copyCommandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &copyCommandBuffer;

    vkQueueSubmit(computeQueue, 1, &submitInfo, VK_NULL_HANDLE);
    // Wait for the memory transfers to be finished before return.
    vkQueueWaitIdle(computeQueue);

    vkFreeCommandBuffers(device, commandPool, 1, &copyCommandBuffer);
  }

  void createShaderStorageBuffers() {
    shaderStorageBuffers.resize(n_buffers);
    shaderStorageBuffersMemory.resize(n_buffers);

    // --- Create input buffer ---

    // Initialize host array
    std::vector<float> array0;
    for (int i = 0; i < buffer_elems; i++) {
      array0.push_back(i);
    }
    VkDeviceSize bufferSize = sizeof(float) * buffer_elems;

    // Create the device buffer
    createBuffer(bufferSize,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, shaderStorageBuffers[0],
                 shaderStorageBuffersMemory[0]);

    // Create a host-mapped staging buffer and copy the array into it
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 stagingBuffer, stagingBufferMemory);

    void *data;
    vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, array0.data(), (size_t)bufferSize);
    vkUnmapMemory(device, stagingBufferMemory);

    // Copy data from staging buffer to the device buffer
    copyBuffer(shaderStorageBuffers[0], stagingBuffer, bufferSize);

    // Destroy the staging buffer
    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);

    // --- Create output buffer of equivalent size ---

    // Create the device buffer
    createBuffer(bufferSize,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, shaderStorageBuffers[1],
                 shaderStorageBuffersMemory[1]);
  }

  void createComputeCommandBuffers() {
    computeCommandBuffers.resize(1);

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = (uint32_t)computeCommandBuffers.size();

    if (vkAllocateCommandBuffers(device, &allocInfo,
                                 computeCommandBuffers.data()) != VK_SUCCESS) {
      throw std::runtime_error("Failed to allocate compute command buffers!");
    }
  }

  void recordComputeCommandBuffer(VkCommandBuffer commandBuffer,
                                  PushConstants pushConstants) {
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
      throw std::runtime_error(
          "Failed to begin recording compute command buffer!");
    }

    vkCmdPushConstants(commandBuffer, computePipelineLayout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants),
                       &pushConstants);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      computePipeline);

    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            computePipelineLayout, 0, 1,
                            &computeDescriptorSets[0], 0, nullptr);

    // Set the number of workgroups here!
    vkCmdDispatch(commandBuffer, 128, 1, 1);

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
      throw std::runtime_error("Failed to record compute command buffer!");
    }
  }

  void compute() {
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    // Compute submission
    recordComputeCommandBuffer(computeCommandBuffers[0], {buffer_elems, 2.0});

    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &computeCommandBuffers[0];
    submitInfo.signalSemaphoreCount = 0;
    submitInfo.pSignalSemaphores = VK_NULL_HANDLE;

    std::cout << "Launching kernel..." << std::endl;
    if (vkQueueSubmit(computeQueue, 1, &submitInfo, VK_NULL_HANDLE) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to submit compute command buffer!");
    };

    // Wait for the computation to be done.
    vkDeviceWaitIdle(device);
    std::cout << "Done!" << std::endl;

    // Create a staging buffer to copy the result from the device to host
    std::vector<float> array1;
    array1.resize(buffer_elems);

    VkDeviceSize bufferSize = sizeof(float) * buffer_elems;
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 stagingBuffer, stagingBufferMemory);

    // Copy data from the device buffer to the staging buffer
    copyBuffer(stagingBuffer, shaderStorageBuffers[1], bufferSize);

    // Copy the data from the staging buffer to the main memory of the host.
    void *data;
    vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
    memcpy(array1.data(), data, (size_t)bufferSize);
    vkUnmapMemory(device, stagingBufferMemory);

    // Destroy the staging buffer
    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);

    std::cout << "Shader output (host):" << std::endl;
    for (int i = 0; i < buffer_elems; i++) {
      printf("%.4f\t", array1[i]);
    }
    std::cout << std::endl;
    vkDeviceWaitIdle(device);
  }

  void cleanup() {
    vkDestroyPipeline(device, computePipeline, nullptr);
    vkDestroyPipelineLayout(device, computePipelineLayout, nullptr);

    // Free the buffers and the underlying resources.
    vkDestroyBuffer(device, shaderStorageBuffers[0], nullptr);
    vkFreeMemory(device, shaderStorageBuffersMemory[0], nullptr);
    vkDestroyBuffer(device, shaderStorageBuffers[1], nullptr);
    vkFreeMemory(device, shaderStorageBuffersMemory[1], nullptr);

    // Also frees the associated descriptor sets and the pool itself.
    vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    
    vkDestroyDescriptorSetLayout(device, computeDescriptorSetLayout, nullptr);

    // Also frees the associated command buffers
    vkDestroyCommandPool(device, commandPool, nullptr);

    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
  }
};

int main(int argc, char **argv) {
  Simplest simplest;
  try {
    simplest.run();
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  return 0;
}