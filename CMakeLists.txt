cmake_minimum_required(VERSION 3.20)
project(reorder_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Vulkan ───────────────────────────────────────────────────────────────────
find_package(Vulkan REQUIRED)

# ── slangc compiler ──────────────────────────────────────────────────────────
find_program(SLANGC slangc REQUIRED
    HINTS ENV SLANG_DIR
          "$ENV{VULKAN_SDK}/bin"
          "$ENV{HOME}/.slang/bin"
          "$ENV{USERPROFILE}/.slang/bin"
    DOC "Path to the Slang shader compiler"
)
message(STATUS "Found slangc: ${SLANGC}")

# ── Compile .slang → .spv ───────────────────────────────────────────────────
set(SHADER_DIR "${CMAKE_SOURCE_DIR}")
set(SPV_DIR    "${CMAKE_BINARY_DIR}/shaders")
file(MAKE_DIRECTORY ${SPV_DIR})

set(SHADER_FILES   reorder.slang        reorder_optimized.slang        reorder_setassoc.slang        reorder_setassoc_v2.slang        reorder_setassoc_v3.slang        reorder_setassoc_v4.slang)
set(SHADER_ENTRIES reorderByMaterial    reorderByMaterialOpt           reorderByMaterialSetAssoc     reorderByMaterialSetAssocV2      reorderByMaterialWaveMerge        reorderByMaterialV4)

set(SPV_OUTPUTS "")
list(LENGTH SHADER_FILES _shader_count)
math(EXPR _shader_last "${_shader_count} - 1")
foreach(_i RANGE ${_shader_last})
    list(GET SHADER_FILES   ${_i} SHADER_FILE)
    list(GET SHADER_ENTRIES ${_i} ENTRY_POINT)
    get_filename_component(SHADER_NAME ${SHADER_FILE} NAME_WE)

    set(SPV_OUTPUT "${SPV_DIR}/${SHADER_NAME}.spv")
    add_custom_command(
        OUTPUT  ${SPV_OUTPUT}
        COMMAND ${SLANGC}
            "${SHADER_DIR}/${SHADER_FILE}"
            -target spirv
            -o "${SPV_OUTPUT}"
            -entry ${ENTRY_POINT}
            -stage compute
        DEPENDS "${SHADER_DIR}/${SHADER_FILE}"
        COMMENT "Compiling ${SHADER_FILE} → ${SHADER_NAME}.spv"
        VERBATIM
    )
    list(APPEND SPV_OUTPUTS ${SPV_OUTPUT})
endforeach()

add_custom_target(shaders ALL DEPENDS ${SPV_OUTPUTS})

# ── RenderDoc (optional) ─────────────────────────────────────────────────────
# Look for renderdoc_app.h in standard install locations.
# If found, enables --capture flag in the benchmark.
find_path(RENDERDOC_INCLUDE_DIR renderdoc_app.h
    HINTS ENV RENDERDOC_DIR
          "C:/Program Files/RenderDoc"
          "$ENV{ProgramFiles}/RenderDoc"
          "/opt/renderdoc"
          "/usr/local/include"
    DOC "Directory containing renderdoc_app.h"
)
if(RENDERDOC_INCLUDE_DIR)
    message(STATUS "Found RenderDoc: ${RENDERDOC_INCLUDE_DIR}")
else()
    message(STATUS "RenderDoc not found — capture support disabled")
endif()

# ── Benchmark executable ────────────────────────────────────────────────────
add_executable(benchmark benchmark.cpp)
target_link_libraries(benchmark PRIVATE Vulkan::Vulkan)
add_dependencies(benchmark shaders)

if(RENDERDOC_INCLUDE_DIR)
    target_include_directories(benchmark PRIVATE ${RENDERDOC_INCLUDE_DIR})
    target_compile_definitions(benchmark PRIVATE HAS_RENDERDOC)
endif()

# Copy .spv files next to the executable so it can find them at runtime.
# On MSVC the exe goes to build/<Config>/, on single-config generators it's build/.
foreach(_i RANGE ${_shader_last})
    list(GET SHADER_FILES ${_i} _sf)
    get_filename_component(_name ${_sf} NAME_WE)
    add_custom_command(TARGET benchmark POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${SPV_DIR}/${_name}.spv"
            "$<TARGET_FILE_DIR:benchmark>/${_name}.spv"
        COMMENT "Copying ${_name}.spv → output directory"
    )
endforeach()
