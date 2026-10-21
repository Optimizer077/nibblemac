
# [pinfer] litert_lm_advanced_main (benchmark CLI: --benchmark, --benchmark_prefill_tokens,
# --benchmark_decode_tokens, --num_iterations ...). Upstream CMake only defines litert_lm_main;
# this reuses its include dirs, link options and link libraries. Appended by bench/litert/build.sh.
if(TARGET litert_lm_main AND NOT TARGET litert_lm_advanced_main)
    add_executable(litert_lm_advanced_main
        "${LITERTLM_GENERATED_SRC_DIR}/runtime/engine/litert_lm_advanced_main.cc")
    if(TARGET litertlm_anchor)
        add_dependencies(litert_lm_advanced_main litertlm_anchor)
    endif()
    target_include_directories(litert_lm_advanced_main PUBLIC ${LITERTLM_INCLUDE_PATHS})
    target_link_options(litert_lm_advanced_main PRIVATE "LINKER:--export-dynamic-symbol=LiteRt*")
    get_target_property(_pinfer_main_libs litert_lm_main LINK_LIBRARIES)
    target_link_libraries(litert_lm_advanced_main PUBLIC ${_pinfer_main_libs})
endif()
# [pinfer] fully static CLIs (Pi glibc 2.41 / GCC 14 runtime are older than the cross sysroot)
# [pinfer] + KleidiAI: XNNPACK (aarch64) references kai_* symbols but the upstream link list omits
# libkleidiai.a; appended last (after --end-group) so it resolves the remaining references.
set(_pinfer_kai "${CMAKE_BINARY_DIR}/external/tensorflow/src/tflite_external-build/kleidiai/libkleidiai.a")
foreach(_pinfer_t litert_lm_main litert_lm_advanced_main)
    if(TARGET ${_pinfer_t})
        target_link_options(${_pinfer_t} PRIVATE "-static")
        target_link_libraries(${_pinfer_t} PUBLIC "${_pinfer_kai}")
    endif()
endforeach()
