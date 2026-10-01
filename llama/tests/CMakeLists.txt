# Golden-tensor tests. Each one recomputes a step of the forward pass and
# diffs it against the float32 dump from reference/dump_golden.py.
#
# They need the weights, so every test skips rather than fails when
# L3LLM_MODEL_DIR is unset -- a fresh clone still gets a green run.
set(L3LLM_GOLDEN_DIR "${CMAKE_SOURCE_DIR}/reference/golden")
configure_file(golden_dir_config.h.in golden_dir_config.h @ONLY)

# compare.h lives here; golden_dir_config.h is generated into the binary dir.
add_library(l3llm_testing INTERFACE)
target_include_directories(l3llm_testing INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
                                                  ${CMAKE_CURRENT_BINARY_DIR})

function(l3llm_add_test name)
    add_executable(${name} ${name}.cc)
    target_link_libraries(${name} PRIVATE l3llm_core l3llm_testing)
    add_test(NAME ${name} COMMAND ${name})
    # Keep the environment the binary reads in one place.
    set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77)
endfunction()

l3llm_add_test(test_embed)
l3llm_add_test(test_rmsnorm)
l3llm_add_test(test_qkv)
l3llm_add_test(test_rope)
l3llm_add_test(test_attention)
l3llm_add_test(test_mlp)
l3llm_add_test(test_layer)
l3llm_add_test(test_model)
l3llm_add_test(test_trace)
