set(LLVM_LINK_COMPONENTS support)
add_clang_executable(free-launch-2
    T2.cpp
    )
target_link_libraries(free-launch-2 PRIVATE
    clangAnalysis
    clangAST
    clangASTMatchers
    clangBasic
    clangDriver
    clangEdit
    clangFrontend
    clangLex
    clangParse
    clangReorderFields
    clangRewrite
    clangRewriteFrontend
    clangSema
    clangSerialization
    clangStaticAnalyzerFrontend
    clangTooling
    clangToolingCore
    )
