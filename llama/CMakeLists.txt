# Loader + (later) kernels, built as a library so the tests can link the
# same code the binary runs.
add_library(l3llm_core STATIC
    die.cc
    json.cc
    safetensors.cc
    config.cc
    embed.cc
    rmsnorm.cc
    linear.cc
    rope.cc
    attention.cc
    mlp.cc
    layer.cc
    model.cc
    summary.cc
    trace.cc
)
target_include_directories(l3llm_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(llama main.cc)
target_link_libraries(llama PRIVATE l3llm_core)

add_subdirectory(tests)
