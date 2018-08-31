# NOTE: This file will be copied in the configure script over to LLVM-CLANG. It is not for use in this directory.

set(LLVM_LINK_COMPONENTS support)

add_clang_executable(smc
	smc.cpp
)
target_link_libraries(smc PRIVATE
  clangTooling
  clangBasic
  clangASTMatchers
)
