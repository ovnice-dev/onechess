#!/usr/bin/env python3
"""Garantit qu'aucun moteur ne survit au script qui l'a lance.

Sous Windows, tuer un processus ne tue pas ses enfants. Une arene interrompue
-- Ctrl+C, fermeture du terminal, redemarrage de l'agent -- laisse donc derriere
elle des moteurs vivants, bloques en lecture sur une entree standard qui ne
viendra jamais. Ils ne se signalent pas : ils tiennent un fichier ouvert, et
surtout ils occupent un coeur.

C'est une menace directe sur la seule chose qui compte ici, la validite des
mesures. Un orphelin oublie pendant un SPRT vole du temps a l'un des deux
moteurs et decale le resultat sans que rien ne le signale.

La parade propre est le Job Object : on inscrit le processus courant dans un
"travail" marque KILL_ON_JOB_CLOSE. Toute la descendance en herite. Quand le
dernier descripteur du travail se ferme -- c'est-a-dire quand le script meurt,
de quelque facon que ce soit -- Windows tue tout ce qui reste dedans.
"""

import ctypes
import os
import sys

JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JobObjectExtendedLimitInformation  = 9

_HANDLE = None      # garde le descripteur en vie pour toute la duree du script


class _IoCounters(ctypes.Structure):
    _fields_ = [(n, ctypes.c_ulonglong) for n in
                ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                 "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class _BasicLimits(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", ctypes.c_uint32),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", ctypes.c_uint32),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", ctypes.c_uint32),
                ("SchedulingClass", ctypes.c_uint32)]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", _BasicLimits),
                ("IoInfo", _IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t)]


def install(quiet=False):
    """A appeler une fois, dans le processus principal. Rend True si la
    protection est active. Un echec n'est jamais fatal : on le signale et le
    script continue, simplement sans filet."""
    global _HANDLE

    if os.name != "nt":
        return False
    if _HANDLE is not None:
        return True

    try:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)

        # Les types doivent etre declares : un HANDLE est un pointeur, et
        # GetCurrentProcess rend le pseudo-descripteur -1, que ctypes refuse de
        # convertir s'il le prend pour un entier C ordinaire.
        k32.CreateJobObjectW.restype        = ctypes.c_void_p
        k32.CreateJobObjectW.argtypes       = [ctypes.c_void_p, ctypes.c_wchar_p]
        k32.GetCurrentProcess.restype       = ctypes.c_void_p
        k32.GetCurrentProcess.argtypes      = []
        k32.SetInformationJobObject.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                                ctypes.c_void_p, ctypes.c_uint32]
        k32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

        job = k32.CreateJobObjectW(None, None)
        if not job:
            raise OSError(ctypes.get_last_error(), "CreateJobObject")

        info = _ExtendedLimits()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not k32.SetInformationJobObject(ctypes.c_void_p(job),
                                           JobObjectExtendedLimitInformation,
                                           ctypes.byref(info), ctypes.sizeof(info)):
            raise OSError(ctypes.get_last_error(), "SetInformationJobObject")

        # Les travaux imbriques existent depuis Windows 8 : etre deja dans un
        # travail (lance par un terminal, un agent, un service) n'empeche rien.
        if not k32.AssignProcessToJobObject(ctypes.c_void_p(job),
                                            ctypes.c_void_p(k32.GetCurrentProcess())):
            raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject")

        _HANDLE = job
        return True

    except OSError as e:
        if not quiet:
            print(f"[avertissement] protection anti-orphelin indisponible : {e}",
                  file=sys.stderr)
        return False
