#include <iostream>
#include <algorithm>
#include <vector>
#include <map>

using namespace std;
using namespace llvm;
using namespace clang;
using namespace clang::driver;
using namespace clang::tooling;
using namespace clang::ast_matchers;

pair<string, string> getDims(string rawString) {
    pair<string, string> ret_pair;

    rawString.erase(remove(rawString.begin(), rawString.end(), '<'), rawString.end());
    rawString.erase(remove(rawString.begin(), rawString.end(), '>'), rawString.end());
    int delim_len = rawString.find(",");
    ret_pair.first = rawString.substr(0, delim_len);
    ret_pair.second = rawString.substr(delim_len + 1, rawString.size() - delim_len - 1);
    return ret_pair;
}

int findAndReplace(string &org, string target, string replacement) {
    int count = 0;
    for(string::size_type i = 0; (i = org.find(target, i)) != string::npos;) {
        org.replace(i, target.size(), replacement);
        i += replacement.size();
        count++;
    }
    return count;
}

string getText(string key, const MatchFinder::MatchResult &Result) {
    const Stmt *st = Result.Nodes.getNodeAs<Stmt>(key);
    SourceLocation temp = Lexer::getLocForEndOfToken(st->getLocEnd(), 0, 
            *Result.SourceManager, LangOptions());
    StringRef ref = Lexer::getSourceText(CharSourceRange::getCharRange(
                                                st->getLocStart(),
                                                temp),
                                             *Result.SourceManager, 
                                             LangOptions());
    return ref.str();
}

void findAndReplaceInRange(const MatchFinder::MatchResult &Result,
                           SourceLocation locStart,
                           SourceLocation locEnd,
                           vector<string> find,
                           vector<string> replace,
                           Rewriter *rewriter) {
    SourceRange range = SourceRange(locStart, locEnd);
    StringRef ref = Lexer::getSourceText(CharSourceRange::getCharRange(range), 
                                         *Result.SourceManager, 
                                         LangOptions());
    string code = ref.str();
    int count = 0;
    for(unsigned i = 0; i < find.size(); i++) {
        count += findAndReplace(code, find[i], replace[i]);
    }
    if (count) {
        rewriter->ReplaceText(range, code);
    }
}

class AllInvocation : public MatchFinder::MatchCallback {
    private:
        Rewriter *rewriter;
        map <const FunctionDecl *, int> *decls;

    public:
        AllInvocation(Rewriter *rewriter, map <const FunctionDecl *, int> *decls) : 
            rewriter(rewriter),
            decls(decls){}

        virtual void run(const MatchFinder::MatchResult &Result) {
            const FunctionDecl *funcDecl = Result.Nodes.getNodeAs<FunctionDecl>("funCall");
            const CUDAKernelCallExpr *cudaCall = Result.Nodes.getNodeAs<CUDAKernelCallExpr>("cudaCall");
            
            // Continue only if it has a subkernel launch
            if (decls->find(funcDecl) != decls->end()) {

                string code = getText("parentDimExpr", Result);
                pair<string, string> dims = getDims(code);
                string after = ", " + dims.first + ", FL_ARG_BUFFER";

                if (decls->operator[](funcDecl) == 1)
                {
                    rewriter->InsertTextBefore(
                            cudaCall->getLocStart(),
                            "\nchar *FL_ARG_BUFFER;\n cudaMalloc((void **)&FL_ARG_BUFFER,MAX_FL_ARGSZ); \n cudaMemset(FL_ARG_BUFFER,0,MAX_FL_ARGSZ); \n");
                    decls->operator[](funcDecl) = 2;
                }
                else {
                    rewriter->InsertTextBefore(
                            cudaCall->getLocStart(),
                            "\ncudaMalloc((void **)&FL_ARG_BUFFER,MAX_FL_ARGSZ); \n cudaMemset(FL_ARG_BUFFER,0,MAX_FL_ARGSZ); \n");
                }
                rewriter->InsertTextAfter(
                        cudaCall->getLocEnd(),
                        after);
                rewriter->InsertTextAfter(
                        cudaCall->getLocEnd().getLocWithOffset(2),
                        "\n cudaFree(FL_ARG_BUFFER);");
                rewriter->overwriteChangedFiles();

            }
        }
};


