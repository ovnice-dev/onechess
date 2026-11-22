/*
  Onechess, a UCI chess engine.
  Copyright (C) 2026 ovnice

  Onechess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Onechess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

// Outils divers (reecrits pour Onechess le 15-11-2026).

#include "misc.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <streambuf>

#if defined(_WIN32)
    #define NOMINMAX
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
#elif defined(__linux__)
    #include <sys/mman.h>
#endif

namespace one {

std::string engine_info() { return "Onechess 1.2"; }

// ================================================================ memoire
#if defined(_WIN32)

namespace {
// Grandes pages sous Windows : il faut activer le droit « verrouiller des
// pages en memoire » pour le temps de l'allocation ; sans ce droit (cas
// courant), on rend nullptr et l'appelant prend une allocation ordinaire.
void* essai_grandes_pages(std::size_t octets) {
    const std::size_t page = GetLargePageMinimum();
    HANDLE            jeton;
    if (!page || !OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &jeton))
        return nullptr;

    void*            bloc = nullptr;
    TOKEN_PRIVILEGES droit{};
    droit.PrivilegeCount = 1;
    if (LookupPrivilegeValueA(nullptr, "SeLockMemoryPrivilege", &droit.Privileges[0].Luid))
    {
        droit.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
        const bool accorde = AdjustTokenPrivileges(jeton, FALSE, &droit, 0, nullptr, nullptr)
                          && GetLastError() == ERROR_SUCCESS;
        if (accorde)
        {
            const std::size_t taille = (octets + page - 1) / page * page;
            bloc = VirtualAlloc(nullptr, taille, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES, PAGE_READWRITE);
            droit.Privileges[0].Attributes = 0;  // on rend le droit aussitot
            AdjustTokenPrivileges(jeton, FALSE, &droit, 0, nullptr, nullptr);
        }
    }
    CloseHandle(jeton);
    return bloc;
}
}  // namespace

void* aligned_large_pages_alloc(std::size_t octets) {
    void* bloc = essai_grandes_pages(octets);
    return bloc ? bloc : VirtualAlloc(nullptr, octets, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
}

void aligned_large_pages_free(void* bloc) {
    if (bloc)
        VirtualFree(bloc, 0, MEM_RELEASE);
}

#else

// Ailleurs : alignement sur 2 Mo et, sous Linux, demande de grandes pages
// transparentes pour ce bloc.
void* aligned_large_pages_alloc(std::size_t octets) {
    constexpr std::size_t deuxMo = std::size_t(1) << 21;
    const std::size_t     taille = (octets + deuxMo - 1) & ~(deuxMo - 1);
    void*                 bloc   = std::aligned_alloc(deuxMo, taille);
    #if defined(MADV_HUGEPAGE)
    if (bloc)
        madvise(bloc, taille, MADV_HUGEPAGE);
    #endif
    return bloc;
}

void aligned_large_pages_free(void* bloc) { std::free(bloc); }

#endif

// ================================================================ journal UCI
namespace {

// Tampon « miroir » : lit ou ecrit dans le flux d'origine, et recopie chaque
// caractere dans le journal ; chaque ligne y est precedee de « << » (recu) ou
// de « >> » (envoye).
class Miroir : public std::streambuf {
   public:
    Miroir(std::streambuf* origine, std::streambuf* journal, const char* marque) :
        origine(origine), journal(journal), marque(marque) {}
    std::streambuf* source() const { return origine; }

   protected:
    int overflow(int c) override { return copie(origine->sputc(char(c))); }
    int underflow() override { return origine->sgetc(); }
    int uflow() override { return copie(origine->sbumpc()); }
    int sync() override {
        journal->pubsync();
        return origine->pubsync();
    }

   private:
    int copie(int c) {
        if (c == EOF)
            return c;
        if (debutLigne)
            journal->sputn(marque, 3);
        debutLigne = c == '\n';
        journal->sputc(char(c));
        return c;
    }
    std::streambuf* origine;
    std::streambuf* journal;
    const char*     marque;
    bool            debutLigne = true;
};

struct Journal {
    std::ofstream           fichier;
    std::unique_ptr<Miroir> entree, sortie;

    void ouvrir(const std::string& nom) {
        fermer();
        if (nom.empty())
            return;
        fichier.open(nom);
        if (!fichier)
        {
            std::cerr << "Impossible d'ouvrir " << nom << std::endl;
            return;
        }
        entree = std::make_unique<Miroir>(std::cin.rdbuf(), fichier.rdbuf(), "<< ");
        sortie = std::make_unique<Miroir>(std::cout.rdbuf(), fichier.rdbuf(), ">> ");
        std::cin.rdbuf(entree.get());
        std::cout.rdbuf(sortie.get());
    }
    void fermer() {
        if (!fichier.is_open())
            return;
        std::cout.rdbuf(sortie->source());
        std::cin.rdbuf(entree->source());
        fichier.close();
    }
    ~Journal() { fermer(); }
};

}  // namespace

void start_logger(const std::string& fichier) {
    static Journal j;
    j.ouvrir(fichier);
}

}  // namespace one
