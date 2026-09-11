cmake_minimum_required(VERSION 3.16)
project(muzero_tictactoe CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(mz
    src/board.cpp
    src/eval.cpp
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
mz_add_test(test_eval)

add_executable(train apps/train.cpp)
target_link_libraries(train mz)

add_executable(evaluate apps/evaluate.cpp)
target_link_libraries(evaluate mz)

add_executable(play_cli apps/play_cli.cpp)
target_link_libraries(play_cli mz)

# Plays one game as X and one as O against minimax, printing every move
# and MCTS visit distribution. Useful for tracing exactly where and why a
# checkpoint loses, e.g. `./diag_eval checkpoint.bin`.
add_executable(diag_eval tools/diag_eval.cpp)
target_link_libraries(diag_eval mz)

# Measures whether the learned dynamics tracks the real game. No AlphaZero
# equivalent exists: AlphaZero has no learned model that could be wrong.
add_executable(latent_probe tools/latent_probe.cpp)
target_link_libraries(latent_probe mz)

add_test(NAME integration_smoke
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/integration_smoke.sh ${CMAKE_CURRENT_BINARY_DIR})
