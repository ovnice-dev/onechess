#include "misc.h"
#if defined(__linux__)
    #include <sys/mman.h>
#endif
#include <streambuf>
#include <iostream>
#include <fstream>

#include <cstdlib>
#include <cstring>
#include <new>

#if defined(_WIN32)
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #include <windows.h>
#endif

namespace one {

std::string engine_info() { return "Onechess 0.1"; }

#if defined(_WIN32)

namespace {

// Les grandes pages (2 Mo) reduisent fortement les defauts de TLB sur une
// table de transposition de plusieurs centaines de Mo. Windows exige le
// privilege SeLockMemoryPrivilege, souvent absent : on retombe alors
// silencieusement sur une allocation ordinaire.
void* large_pages_alloc(std::size_t allocSize) {
    HANDLE            hToken;
    TOKEN_PRIVILEGES  tp;
    LUID              luid;
    void*             mem  = nullptr;
    const std::size_t page = GetLargePageMinimum();

    if (!page)
        return nullptr;

    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &hToken))
        return nullptr;

    if (LookupPrivilegeValue(nullptr, TEXT("SeLockMemoryPrivilege"), &luid))
    {
        tp.PrivilegeCount           = 1;
        tp.Privileges[0].Luid       = luid;
        tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;

        if (AdjustTokenPrivileges(hToken, FALSE, &tp, sizeof(TOKEN_PRIVILEGES), nullptr, nullptr)
            && GetLastError() == ERROR_SUCCESS)
        {
            // Arrondi a un multiple de la taille de grande page
            const std::size_t size = ((allocSize + page - 1) / page) * page;

            mem = VirtualAlloc(nullptr, size, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES,
                               PAGE_READWRITE);

            // On rend le privilege immediatement : il n'est utile qu'ici
            tp.Privileges[0].Attributes = 0;
            AdjustTokenPrivileges(hToken, FALSE, &tp, sizeof(TOKEN_PRIVILEGES), nullptr, nullptr);
        }
    }

    CloseHandle(hToken);
    return mem;
}

}  // namespace

void* aligned_large_pages_alloc(std::size_t allocSize) {
    if (void* mem = large_pages_alloc(allocSize))
        return mem;

    return VirtualAlloc(nullptr, allocSize, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
}

void aligned_large_pages_free(void* mem) {
    if (mem)
        VirtualFree(mem, 0, MEM_RELEASE);
}

#else

void* aligned_large_pages_alloc(std::size_t allocSize) {
    // Grandes pages transparentes (Linux) : la table et les poids du reseau
    // depassent largement ce que couvrent les pages de 4 Ko dans le TLB.
    constexpr std::size_t alignment = 2 * 1024 * 1024;
    std::size_t           size      = ((allocSize + alignment - 1) / alignment) * alignment;
    void*                 mem       = std::aligned_alloc(alignment, size);
    #if defined(MADV_HUGEPAGE)
    if (mem)
        madvise(mem, size, MADV_HUGEPAGE);
    #endif
    return mem;
}

void aligned_large_pages_free(void* mem) { std::free(mem); }

#endif


// Flux en "T" : tout ce qui passe par cin ou cout est aussi ecrit dans le
// fichier, precede de << (entree) ou >> (sortie). Repris de Stockfish.
namespace {
struct Tie: public std::streambuf {
    Tie(std::streambuf* b, std::streambuf* l) : buf(b), logBuf(l) {}
    int sync() override { return logBuf->pubsync(), buf->pubsync(); }
    int overflow(int c) override { return log(buf->sputc(char(c)), ">> "); }
    int underflow() override { return buf->sgetc(); }
    int uflow() override { return log(buf->sbumpc(), "<< "); }
    std::streambuf *buf, *logBuf;
    int log(int c, const char* prefix) {
        static int last = '\n';
        if (last == '\n')
            logBuf->sputn(prefix, 3);
        return last = logBuf->sputc(char(c));
    }
};

struct Logger {
    std::ofstream file;
    Tie           in, out;
    Logger() : in(std::cin.rdbuf(), file.rdbuf()), out(std::cout.rdbuf(), file.rdbuf()) {}
    ~Logger() { start(""); }
    void start(const std::string& fname) {
        if (file.is_open())
        {
            std::cout.rdbuf(out.buf);
            std::cin.rdbuf(in.buf);
            file.close();
        }
        if (!fname.empty())
        {
            file.open(fname, std::ifstream::out);
            if (!file.is_open())
            {
                std::cerr << "Impossible d'ouvrir " << fname << std::endl;
                return;
            }
            std::cin.rdbuf(&in);
            std::cout.rdbuf(&out);
        }
    }
};
}  // namespace

void start_logger(const std::string& fname) {
    static Logger l;
    l.start(fname);
}

}  // namespace one
