#include "misc.h"

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
    constexpr std::size_t alignment = 4096;
    std::size_t           size      = ((allocSize + alignment - 1) / alignment) * alignment;
    return std::aligned_alloc(alignment, size);
}

void aligned_large_pages_free(void* mem) { std::free(mem); }

#endif

}  // namespace one
