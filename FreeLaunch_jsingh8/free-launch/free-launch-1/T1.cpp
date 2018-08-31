#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Driver/Options.h"
#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"
#include <iostream>
#include <algorithm>
#include <vector>
#include <map>
#include <regex>
#include "../util.h"

using namespace std;
using namespace llvm;
using namespace clang;
using namespace clang::driver;
using namespace clang::tooling;
using namespace clang::ast_matchers;

static llvm::cl::OptionCategory MyToolCategory("my-tool options");
static cl::extrahelp CommonHelp(CommonOptionsParser::HelpMessage);
static cl::extrahelp MoreHelp("\nMore help text...");

class T1Handler : public MatchFinder::MatchCallback {
    private:
        Rewriter *rewriter;
        map <const FunctionDecl *, int> *decls;

    public :
        T1Handler(Rewriter *rewriter, 
                  map <const FunctionDecl *, int> *p_decls) :
            rewriter(rewriter), 
            decls(p_decls) {}

        virtual void run(const MatchFinder::MatchResult &Result) {
            const FunctionDecl *parentKernel = Result.Nodes.getNodeAs<FunctionDecl>("parentKernel");
            if (parentKernel->hasAttr<CUDAGlobalAttr>()) {

                decls->operator[](parentKernel) = 1;
                const CUDAKernelCallExpr *kernel_call = Result.Nodes.getNodeAs<CUDAKernelCallExpr>("kernelCall");
                const DeclRefExpr *childKernel = Result.Nodes.getNodeAs<DeclRefExpr>("childKernel");
                const FunctionDecl *childFunction = dyn_cast<FunctionDecl>(childKernel->getDecl());
                const Stmt *parentFunStmt = childFunction->getBody();

                // Offset taken to ignore the open braces.
                SourceRange range = SourceRange(parentFunStmt->getLocStart().getLocWithOffset(1), 
                                                parentFunStmt->getLocEnd());
                StringRef ref = Lexer::getSourceText(CharSourceRange::getCharRange(range), 
                                                     *Result.SourceManager, 
                                                     LangOptions());

                pair<string, string> dims = getDims(getText("dimExpr", Result));

                const Expr* const* expr_args = kernel_call->getArgs();
                vector<string> callArgs(kernel_call->getNumArgs());
                vector<string> callArgTypes(kernel_call->getNumArgs());
                const Expr *tmp;
                SourceLocation temp;
                for (unsigned i = 0; i < kernel_call->getNumArgs(); i++) {
                    tmp = *(expr_args + i);
                    temp = Lexer::getLocForEndOfToken(tmp->getLocEnd(), 0,
                            *Result.SourceManager, LangOptions());
                    callArgs[i] = Lexer::getSourceText(CharSourceRange::getCharRange(
                                                        tmp->getLocStart(), 
                                                        temp), 
                                                   *Result.SourceManager, LangOptions()).str();
		    LangOptions lo;
		    PrintingPolicy policy(lo);
                    callArgTypes[i] = QualType::getAsString(tmp->getType().split(), policy);
                    if (callArgTypes[i] == "_Bool") {
                        callArgTypes[i] = "bool";
                    }
                    else if (callArgTypes[i] == "_Bool *") {
                        callArgTypes[i] = "bool *";
                    }
                }

                vector<string> kernelArgs(kernel_call->getNumArgs());
                DeclarationName declName;
                for (unsigned i = 0; i < childFunction->getNumParams(); i++) {
                    kernelArgs[i] = childFunction->getParamDecl(i)->getNameAsString();
                }

                // Strings to be used to rewrite.
                string moreArgsForFunction = ", int blocks, char *FL_Args ";
                string startFunctionCode = "\n   FL_T1_Preloop ";
                string endFunctionCode = "\n   FL_postChildLog \n";
                string preChildCode = 
                    "\n int FL_lc = atomicAdd(&FL_count,1); ";
                string postRecordFill =
                    "\n atomicAdd(&FL_totalBlocks,_tmp_childGridSize); \
                    \n FL_check = FL_ttid; \
                    \n goto P;  \
                    \n C:    __threadfence(); \n";
                string unwrapArgs = 
                    "\n FL_T1_Postloop; \
                     \n char * _tmp_p = (char*)((&FL_Args[0])+ckernelSeqNum*FL_childKernelArgSz); \
                     \n _tmp_p+=sizeof(int);";
                string childCode = ref.str();

                // Fill arguments in preChildCode.
                preChildCode += "\nFL_childKernelArgSz = ";
                for (unsigned i = 0 ; i < kernel_call->getNumArgs(); i++) {
                    if (i != 0) {
                        preChildCode += "+ ";
                    }
                    preChildCode += "sizeof(" + callArgTypes[i] + ") ";
                }
                preChildCode += "; \
                     \n    char * _tmp_p = (char *) ((&FL_Args[0])+FL_lc*FL_childKernelArgSz); \
                     \n    int _tmp_childGridSize = (" + dims.first + "); \
                     \n    memcpy((void*)_tmp_p, (void*) &_tmp_childGridSize, sizeof(int)); \
                     \n    _tmp_p+=sizeof(int); \
                     \n    FL_childBlockSize=" + dims.second + ";";

                for (unsigned i = 0 ; i < kernel_call->getNumArgs(); i++) {
                    preChildCode += "\n    memcpy((void*)_tmp_p, (void*) &" + callArgs[i] + ", sizeof(" + callArgTypes[i] + "));";
                    if (i != kernel_call->getNumArgs() - 1) {
                        preChildCode += "\n    _tmp_p+=sizeof(" + callArgTypes[i] + ");";
                    }
                }

                for (unsigned i = 0 ; i < kernel_call->getNumArgs(); i++) {
                    unwrapArgs += 
                        "\n    " + callArgTypes[i] + " " + callArgs[i] + "; \
                        \n   memcpy((void*)&" + callArgs[i] + ", (void*)_tmp_p, sizeof(" + callArgTypes[i] + ")); \
                        \n   _tmp_p+=sizeof(" + callArgTypes[i] + ");";
                }

                findAndReplace(childCode, "return;", "continue;");
                findAndReplace(childCode, "blockDim.x", "FL_childBlockSize");
                findAndReplace(childCode, "blockIdx.x", "logicalChildBlockSeqNum");
                findAndReplace(childCode, "threadIdx.x", "(threadIdx.x%FL_childBlockSize)");
                for (unsigned i = 0; i < kernel_call->getNumArgs(); i++) {
                    findAndReplace(childCode, kernelArgs[i], callArgs[i]);
                }

                //Rewriters
                vector <string> start_find;
                vector <string> start_replace;
                vector <string> end_find;
                vector <string> end_replace;

                start_find.push_back("return;");
                start_find.push_back("blockIdx.x");
                start_replace.push_back("continue;");
                start_replace.push_back("(blockIdx.x*FL_y)");
                findAndReplaceInRange(Result, 
                                      parentKernel->getBody()->getLocStart().getLocWithOffset(1),
                                      kernel_call->getLocStart().getLocWithOffset(-1),
                                      start_find,
                                      start_replace,
                                      rewriter);
                end_find.push_back("return;");
                end_replace.push_back("continue;");
                findAndReplaceInRange(Result, 
                                      kernel_call->getLocEnd().getLocWithOffset(2),
                                      parentKernel->getBody()->getLocEnd().getLocWithOffset(-1),
                                      end_find,
                                      end_replace,
                                      rewriter);
                rewriter->ReplaceText(SourceRange(kernel_call->getLocStart(),
                                                  kernel_call->getLocEnd().getLocWithOffset(1)), 
                                                  preChildCode+postRecordFill);
                rewriter->InsertTextAfter(parentKernel->getBody()->getLocStart().getLocWithOffset(-2),
                                          moreArgsForFunction);
                rewriter->InsertTextAfter(parentKernel->getBody()->getLocStart().getLocWithOffset(1),
                                          startFunctionCode);
                rewriter->InsertTextAfter(parentKernel->getBody()->getLocEnd().getLocWithOffset(-1),
                                          unwrapArgs + childCode + endFunctionCode);
                rewriter->overwriteChangedFiles();
            }
        }
};

class T1ASTConsumer : public ASTConsumer {
    public:
        T1ASTConsumer(Rewriter *rewriter, 
                map <const FunctionDecl *, int> *decls
                ) : 
            FPrinter(rewriter, decls),
            APrinter(rewriter, decls) {

            Finder.addMatcher(
                functionDecl(
                    hasDescendant( 
                        cudaKernelCallExpr(
                            allOf(
                                hasDescendant(
                                    declRefExpr().bind("childKernel")
                                   ),
                                hasDescendant(
                                    callExpr().bind("dimExpr")
                                    )
                                )
                            ).bind("kernelCall")
                        )
                    ).bind("parentKernel"),
                &FPrinter);

            Finder.addMatcher(
                    cudaKernelCallExpr(
                        allOf(
                            hasDescendant(
                                declRefExpr(
                                    to(
                                        functionDecl(
                                            ).bind("funCall")
                                      )
                                    )
                               ),
                            hasDescendant(
                                callExpr().bind("parentDimExpr")
                                )
                            )
                        ).bind("cudaCall"),
                    &APrinter);
        }
  
        void HandleTranslationUnit(ASTContext &Context) override {
            Finder.matchAST(Context);
        }

    private:
        T1Handler FPrinter;
        AllInvocation APrinter;
        MatchFinder Finder;
};

class T1FrontendAction : public ASTFrontendAction {
    public:
        T1FrontendAction() {}
        void EndSourceFileAction() override {
            RewriteBuffer &temp = rewriter.getEditBuffer(rewriter.getSourceMgr().getMainFileID());
            temp.InsertText(0, "#include \"freeLaunch_T1.h\"\n");
            
            std::error_code error_code;
            string fileName = getCurrentFile().str();
            llvm::raw_fd_ostream outFile(fileName, error_code, llvm::sys::fs::F_None);
            temp.write(outFile);
            outFile.close(); 
        }

        std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                       StringRef file) override {
            rewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
            return make_unique<T1ASTConsumer>(&rewriter, &decls);
        }

    private:
        Rewriter rewriter;
        map <const FunctionDecl *, int> decls;
};

int main(int argc, const char **argv) {
	CommonOptionsParser OptionsParser(argc, argv, MyToolCategory);
    ClangTool Tool(OptionsParser.getCompilations(),
				   OptionsParser.getSourcePathList());
    return Tool.run(newFrontendActionFactory<T1FrontendAction>().get());
}
