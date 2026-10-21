
# [pinfer] Appended to LiteRT-LM's cmake/packages/litert/litert_patcher.cmake by bench/litert/build.sh
# (the LiteRT checkout is git-reset and re-patched on every build, so the fix lives here).
# For an aarch64 target, TFLite's XNNPACK microkernels reference KleidiAI (kai_*) symbols, but
# LiteRT's own links (libLiteRt.so, tools/*) do not list libkleidiai.a -> undefined references.
# Fix: append the archive to CMAKE_{C,CXX}_STANDARD_LIBRARIES (the end of every link line),
# set in LiteRT's root CMakeLists before its subdirectories are added.
get_filename_component(_pinfer_kai
    "${LITERTLM_LITERT_SRC_DIR}/../../../../tensorflow/src/tflite_external-build/kleidiai/libkleidiai.a"
    ABSOLUTE)
set(_pinfer_root "${LITERTLM_LITERT_SRC_DIR}/CMakeLists.txt")
file(READ "${_pinfer_root}" _pinfer_content)
string(FIND "${_pinfer_content}" "[pinfer] KleidiAI" _pinfer_done)
if(_pinfer_done EQUAL -1)
    set(_pinfer_inject "# [pinfer] KleidiAI for XNNPACK microkernels (aarch64 cross build)
if(EXISTS \"${_pinfer_kai}\")
  string(APPEND CMAKE_CXX_STANDARD_LIBRARIES \" \\\"${_pinfer_kai}\\\"\")
  string(APPEND CMAKE_C_STANDARD_LIBRARIES \" \\\"${_pinfer_kai}\\\"\")
endif()
add_subdirectory(c)")
    string(REPLACE "add_subdirectory(c)" "${_pinfer_inject}" _pinfer_content "${_pinfer_content}")
    file(WRITE "${_pinfer_root}" "${_pinfer_content}")
endif()
message(STATUS "[pinfer] KleidiAI (${_pinfer_kai}) appended to LiteRT standard link libraries")
