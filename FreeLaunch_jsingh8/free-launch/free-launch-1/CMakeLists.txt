set(LLVM_LINK_COMPONENTS support)
add_clang_executable(free-launch-1
    T1.cpp
    )

target_link_libraries(free-launch-1 PRIVATE
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

