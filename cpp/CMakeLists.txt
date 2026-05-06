cmake_minimum_required(VERSION 3.18)
project(mtcnn_cpp_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

function(copy_runtime_dlls target_name)
    if(NOT WIN32)
        return()
    endif()

    # 复制 LibTorch 运行时 DLL，避免运行时提示缺少 torch_cpu.dll、c10.dll 等依赖。
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    if(TORCH_DLLS)
        add_custom_command(TARGET ${target_name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${TORCH_DLLS}
                $<TARGET_FILE_DIR:${target_name}>
            COMMAND_EXPAND_LISTS
        )
    endif()

    # 复制 OpenCV 运行时 DLL。不同安装方式目录略有差异，这里做几种常见路径兼容。
    set(OPENCV_RUNTIME_DIR_CANDIDATES
        "${OpenCV_DIR}/../bin"
        "${OpenCV_DIR}/../../bin"
        "${OpenCV_DIR}/../../../bin"
        "${OpenCV_DIR}/x64/vc16/bin"
        "${OpenCV_DIR}/x64/vc17/bin"
    )

    foreach(candidate_dir IN LISTS OPENCV_RUNTIME_DIR_CANDIDATES)
        if(EXISTS "${candidate_dir}")
            file(GLOB OPENCV_DLLS "${candidate_dir}/*.dll")
            if(OPENCV_DLLS)
                add_custom_command(TARGET ${target_name} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        ${OPENCV_DLLS}
                        $<TARGET_FILE_DIR:${target_name}>
                    COMMAND_EXPAND_LISTS
                )
                break()
            endif()
        endif()
    endforeach()
endfunction()

add_library(mtcnn_detector
    src/mtcnn_detector.cpp
)

target_include_directories(mtcnn_detector
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(mtcnn_detector
    PUBLIC
        ${TORCH_LIBRARIES}
        ${OpenCV_LIBS}
)

add_executable(mtcnn_image_demo
    examples/image_demo.cpp
)

target_link_libraries(mtcnn_image_demo
    PRIVATE
        mtcnn_detector
)
copy_runtime_dlls(mtcnn_image_demo)

add_executable(mtcnn_camera_demo
    examples/camera_demo.cpp
)

target_link_libraries(mtcnn_camera_demo
    PRIVATE
        mtcnn_detector
)
copy_runtime_dlls(mtcnn_camera_demo)

if(MSVC)
    target_compile_options(mtcnn_detector PRIVATE /utf-8)
    target_compile_options(mtcnn_image_demo PRIVATE /utf-8)
    target_compile_options(mtcnn_camera_demo PRIVATE /utf-8)
endif()
