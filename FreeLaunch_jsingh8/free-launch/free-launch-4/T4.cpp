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

class T4Handler : public MatchFinder::MatchCallback {
    private:
        Rewriter *rewriter;
    public :
        T4Handler(Rewriter *rewriter) : rewriter(rewriter) {}

        virtual void run(const MatchFinder::MatchResult &Result) {
            const FunctionDecl *parent_kernel = Result.Nodes.getNodeAs<FunctionDecl>("parentKernel");
            if (parent_kernel->hasAttr<CUDAGlobalAttr>()) {
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

                const Stmt *grid_val = Result.Nodes.getNodeAs<Stmt>("dimExpr");
                SourceLocation temp = Lexer::getLocForEndOfToken(grid_val->getLocEnd(), 0, 
                        *Result.SourceManager, LangOptions());
                StringRef gridRef = Lexer::getSourceText(CharSourceRange::getCharRange(
                                                       grid_val->getLocStart(),
                                                       temp),
                                                         *Result.SourceManager, 
                                                         LangOptions());
                pair<string, string> dims = getDims(gridRef.str());
                string preChildCode = 
                    "\nfor(int i = 0; i < int((" + dims.first + ") * (" + dims.second + ")); i++) {\n";
                string postChildCode = "}";
                string childCode = ref.str();

                const Expr* const* test = kernel_call->getArgs();
                vector<string> callArgs(kernel_call->getNumArgs());
                const Expr *tmp;
                for (unsigned i = 0; i < kernel_call->getNumArgs(); i++) {
                    tmp = *(test + i);
                    temp = Lexer::getLocForEndOfToken(tmp->getLocEnd(), 0,
                            *Result.SourceManager, LangOptions());
                    gridRef = Lexer::getSourceText(CharSourceRange::getCharRange(
                                                        tmp->getLocStart(), 
                                                        temp), 
                                                   *Result.SourceManager, LangOptions());
                    callArgs[i] = gridRef.str();
                }

                vector<string> kernelArgs(kernel_call->getNumArgs());
                DeclarationName declName;
                for (unsigned i = 0; i < childFunction->getNumParams(); i++) {
                    kernelArgs[i] = childFunction->getParamDecl(i)->getNameAsString();
                }

                findAndReplace(childCode, "return;", "continue;");
                findAndReplace(childCode, "blockIdx.x * blockDim.x + threadIdx.x", "i");
                findAndReplace(childCode, "(blockIdx.x*blockDim.x)+threadIdx.x", "i");
                findAndReplace(childCode, "threadIdx.x + blockIdx.x * blockDim.x", "i");
                findAndReplace(childCode, "threadIdx.x+(blockIdx.x*blockDim.x)", "i");
                findAndReplace(childCode, "threadIdx.x", "(i % (" + dims.second + "))");

                for (unsigned i = 0; i < kernel_call->getNumArgs(); i++) {
                    findAndReplace(childCode, kernelArgs[i], callArgs[i]);
                }

                // Offset taken to ignore the ; at the end of the stmt.
                rewriter->ReplaceText(SourceRange(kernel_call->getLocStart(), 
                                                  kernel_call->getLocEnd().getLocWithOffset(1)), 
                                      preChildCode + childCode + postChildCode);
                rewriter->overwriteChangedFiles();
            }
        }
};

class T4ASTConsumer : public ASTConsumer {
    public:
        T4ASTConsumer(Rewriter *rewriter) : FPrinter(rewriter) {
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

        }
  
        void HandleTranslationUnit(ASTContext &Context) override {
            Finder.matchAST(Context);
        }

    private:
        T4Handler FPrinter;
        MatchFinder Finder;
};

class T4FrontendAction : public ASTFrontendAction {
    public:
        T4FrontendAction() {}
        std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                       StringRef file) override {
            rewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
            return make_unique<T4ASTConsumer>(&rewriter);
        }

    private:
        Rewriter rewriter;
};

int main(int argc, const char **argv) {
	CommonOptionsParser OptionsParser(argc, argv, MyToolCategory);
    ClangTool Tool(OptionsParser.getCompilations(),
				   OptionsParser.getSourcePathList());
    return Tool.run(newFrontendActionFactory<T4FrontendAction>().get());
}
