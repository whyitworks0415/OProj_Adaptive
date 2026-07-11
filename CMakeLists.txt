cmake_minimum_required(VERSION 3.20)
project(VulkanRenderer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Vulkan 패키지 찾기
find_package(Vulkan REQUIRED)

# GLFW 패키지 찾기
find_package(glfw3 REQUIRED)

# GLM과 보조 헤더 찾기
find_package(glm REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_path(TINYGLTF_INCLUDE_DIRS "tiny_gltf.h")

# 빌드 대상 소스 파일 (맵 로딩은 GLTFLoader 하나로 통일)
set(SOURCES
    src/main.cpp
    src/VulkanApp.cpp
    src/Camera.cpp
    src/PerformanceStats.cpp
    src/SceneLoader.cpp
    src/Replay.cpp
    src/GLTFLoader.cpp
    src/stb_image_impl.cpp
)

add_executable(VulkanRenderer ${SOURCES})

target_include_directories(VulkanRenderer PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${Vulkan_INCLUDE_DIRS}
    ${TINYGLTF_INCLUDE_DIRS}
)

target_link_libraries(VulkanRenderer PRIVATE
    Vulkan::Vulkan
    glfw
    glm::glm
    imgui::imgui
)

# 컴파일된 셰이더 복사
add_custom_command(TARGET VulkanRenderer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/shaders/spv
        $<TARGET_FILE_DIR:VulkanRenderer>/shaders/spv
    COMMENT "Copying compiled SPIR-V shaders"
)

# 맵 디렉터리 복사
add_custom_command(TARGET VulkanRenderer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/maps
        $<TARGET_FILE_DIR:VulkanRenderer>/maps
    COMMENT "Copying scene maps"
)

# 리플레이 디렉터리 생성
add_custom_command(TARGET VulkanRenderer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
        $<TARGET_FILE_DIR:VulkanRenderer>/replays
    COMMENT "Creating replays directory"
)

# 컴파일 정의
target_compile_definitions(VulkanRenderer PRIVATE
    $<$<CONFIG:Debug>:ENABLE_VALIDATION_LAYERS>
    GLM_FORCE_RADIANS
    GLM_FORCE_DEPTH_ZERO_TO_ONE
)

# MSVC에서도 UTF-8 소스 파일로 컴파일되도록 한다.
if(MSVC)
    target_compile_options(VulkanRenderer PRIVATE /utf-8)
endif()
