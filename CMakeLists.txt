cmake_minimum_required(VERSION 3.16)
project(muzero_tictactoe CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(mz
    src/board.cpp
    src/mcts.cpp
    src/minimax.cpp
    src/mlp.cpp
    src/network.cpp
    src/replay_buffer.cpp
    src/selfplay.cpp
    src/targets.cpp
)
target_include_directories(mz PUBLIC include)

enable_testing()

# Test binaries must assert regardless of CMAKE_BUILD_TYPE. The default
# build type is Release, which defines NDEBUG and would otherwise compile
# every assert() in these targets down to a no-op, making the test suite
# vacuous. Strip NDEBUG here without affecting the mz library or app
# targets, which should keep Release optimization.
function(mz_add_test name)
  add_executable(${name} tests/${name}.cpp)
  target_link_libraries(${name} mz)
  target_compile_options(${name} PRIVATE -UNDEBUG)
  add_test(NAME ${name} COMMAND ${name})
endfunction()

mz_add_test(test_board)
mz_add_test(test_minimax)
mz_add_test(test_mlp)
mz_add_test(test_mcts)
mz_add_test(test_network)
mz_add_test(test_replay_buffer)
mz_add_test(test_targets)
mz_add_test(test_selfplay)
