"""Patch NATTEN for cuda-wheels build compatibility.

NATTEN's setup.py reads its own env vars (NATTEN_CUDA_ARCH, NATTEN_N_WORKERS)
and ignores PyTorch/cuda-wheels conventions (TORCH_CUDA_ARCH_LIST, MAX_JOBS).
This patch injects a small shim near the top of setup.py that translates
between the two so cuda-wheels' standard build env "just works":

  1. If NATTEN_CUDA_ARCH is unset, fall back to TORCH_CUDA_ARCH_LIST. Strip
     the '+PTX' suffix (NATTEN's parser doesn't accept it) and normalize
     space-separated to semicolon-separated (NATTEN's expected separator,
     per get_cuda_arch_list() at setup.py).
  2. If NATTEN_N_WORKERS is unset, fall back to MAX_JOBS. Without this,
     NATTEN defaults to cpu_count()//4 which is unrelated to the cuda-wheels
     max_jobs cap — and CUTLASS template instantiations need the cap.
  3. (REMOVED 2026-08-26.) This patch used to strip 10.0 / 10.3 from the arch
     list on Windows, because NATTEN enables -DNATTEN_WITH_BLACKWELL_FNA=1 when
     those archs are present and MSVC rejects the resulting
     sm100_fmha_bwd_kernel_tma_warpspecialized.hpp with C2061. The limitation
     is real; deciding it HERE was the mistake -- the resolver and the verify
     gate both read the YAML, so a patch that disagrees with them is invisible
     until C7 fails. It now lives in arch_override.yml's
     arch_list_by_cuda_windows, next to the reason. See the note at the shim.

Also patches pyproject.toml: setuptools.packages.find.where = ["src/"] has
a trailing slash that newer setuptools' convert_path rejects on Windows
("path 'src/' cannot end with '/'"), failing metadata generation before
the build even starts. Linux's convert_path is permissive and ignores it.
"""
from pathlib import Path

# pyproject.toml fixes:
#   1. strip trailing slash from packages.find.where (Windows fix).
#   2. lowercase the [project] name "NATTEN" -> "natten" so the wheel
#      distribution field is deterministic across setuptools/wheel
#      versions. Upstream has `name = "NATTEN"` in pyproject.toml but
#      `name="natten"` in setup.py -- modern setuptools prefers
#      pyproject.toml -> uppercase wheels; older falls back to setup.py
#      -> lowercase. Result: same matrix dispatch produces some
#      `natten-*.whl` and some `NATTEN-*.whl`, which our release-upload
#      step then assigns to two SEPARATE GitHub releases. PEP 503 says
#      project names should be normalized lowercase anyway. Fix both
#      sources to lowercase.
pyproject_file = Path("pyproject.toml")
pyproject_text = pyproject_file.read_text()
old_where = 'where = ["src/"]'
new_where = 'where = ["src"]'
if old_where in pyproject_text:
    pyproject_text = pyproject_text.replace(old_where, new_where, 1)
    print("Patched pyproject.toml: packages.find.where 'src/' -> 'src' (Windows fix)")
else:
    print("NOTE: pyproject.toml didn't contain 'where = [\"src/\"]' -- skipping (may already be fixed upstream)")
old_name = 'name = "NATTEN"'
new_name = 'name = "natten"'
if old_name in pyproject_text:
    pyproject_text = pyproject_text.replace(old_name, new_name, 1)
    print("Patched pyproject.toml: [project] name 'NATTEN' -> 'natten' (PEP 503 normalization)")
else:
    print("NOTE: pyproject.toml didn't contain 'name = \"NATTEN\"' -- skipping (may already be lowercase upstream)")
pyproject_file.write_text(pyproject_text)

# csrc/CMakeLists.txt: strip GCC-only flags forwarded to host compiler that
# MSVC chokes on. `-Wconversion` triggers cl error D8021 ("invalid numeric
# argument '/Wconversion'") because MSVC parses '-W<digit>' as warning level.
# `-fno-strict-aliasing` is a GCC aliasing knob with no MSVC equivalent;
# MSVC errors out the same way. Neither is load-bearing for libnatten
# correctness — they're a non-critical warning + a GCC-specific optimizer
# safety hint. Strip both unconditionally; behavior on Linux is unchanged
# in any way that matters for the built kernels.
cmake_file = Path("csrc/CMakeLists.txt")
cmake_text = cmake_file.read_text()
patched_cmake = cmake_text
for line in (
    'set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Wconversion")',
    'set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fno-strict-aliasing")',
):
    if line in patched_cmake:
        patched_cmake = patched_cmake.replace(line + "\n", "", 1)
        print(f"Stripped from csrc/CMakeLists.txt: {line}")
    else:
        print(f"NOTE: csrc/CMakeLists.txt didn't contain {line!r} -- skipping (may already be removed upstream)")
if patched_cmake != cmake_text:
    cmake_file.write_text(patched_cmake)

# Windows shard mode: MSVC linker fails the partial-link in each shard
# with "LNK1120: 2709 unresolved externals" because only ~1/16 of the .o
# files are present per shard. Linux's ld is permissive enough that the
# partial .so still gets emitted (we discard it anyway -- only the .o
# files matter from a shard). MSVC needs /FORCE:UNRESOLVED to tell it
# to produce the .pyd despite unresolved symbols.
#
# In the link job, CUDA_WHEELS_SHARD_COUNT is not exported (the
# build-wheel action only sets it in compile-shard mode), so this
# conditional is a no-op there -- the link runs strict-linked with all
# .o files present and no unresolved externals.
cmake_text = cmake_file.read_text()
shard_link_block = '''

# --- cuda-wheels Windows shard mode (injected) ---
# Tell MSVC's linker to ignore unresolved externals during a compile-shard
# build (env var CUDA_WHEELS_SHARD_COUNT set). The resulting .pyd is
# discarded; only the .o files are uploaded to the link job.
if(NATTEN_IS_WINDOWS AND DEFINED ENV{CUDA_WHEELS_SHARD_COUNT})
    target_link_options(natten PRIVATE "/FORCE:UNRESOLVED")
    message(STATUS "cuda-wheels Windows shard mode: /FORCE:UNRESOLVED enabled")
endif()
# --- end cuda-wheels Windows shard mode ---
'''
if 'cuda-wheels Windows shard mode' not in cmake_text:
    cmake_file.write_text(cmake_text + shard_link_block)
    print("Appended Windows-shard-mode /FORCE:UNRESOLVED block to csrc/CMakeLists.txt")
else:
    print("NOTE: /FORCE:UNRESOLVED block already present in csrc/CMakeLists.txt -- skipping")

# Restrict Blackwell + Hopper autogen .cu files to their target archs by
# moving them into per-arch OBJECT libraries. NATTEN gates Blackwell with
# NATTEN_WITH_BLACKWELL_FNA and Hopper with NATTEN_WITH_HOPPER_FNA (single
# global flags), so when on, nvcc compiles each .cu against the target's
# full CUDA_ARCHITECTURES list -- ~5x wasted nvcc work per Blackwell file
# and ~4-7x per Hopper file. Earlier attempts used `set_source_files_properties
# (... PROPERTIES CUDA_ARCHITECTURES ...)` but cmake does NOT recognize that
# at source-file scope (target-only property, per cmake gitlab #25105 and
# absence of any prop_sf/CUDA_ARCHITECTURES doc); the call silently no-ops.
# OBJECT libraries with per-target CUDA_ARCHITECTURES is the idiomatic and
# actually-working approach.
#
# Arch suffix discipline: cmake values are "100a-real;103a-real" / "90a-real"
# (NOT bare "100;103" / "90"). The `a` is required to enable CUTLASS Hopper/
# Blackwell features (wgmma, TMA, cluster, mbarrier); without it, nvcc would
# either fail to compile architecture-specific intrinsics or silently emit
# non-functional SASS. The `-real` suffix prevents PTX bloat (binary-only).
#
# Runtime safety verified in checks.py + *_forward.cu: NATTEN's dispatcher
# rejects any device_cc outside the target arch range before the kernel host
# entry is called.
cmake_text = cmake_file.read_text()
old_add_lib = "add_library(natten SHARED ${ALL_SOURCES})"
new_add_lib = '''# --- cuda-wheels arch-specific OBJECT libraries (injected) ---
# Blackwell DC: compile only for sm_100a (B200). User decision: drop
# sm_103a (B300/Blackwell Ultra) -- those users fall through to the
# regular multi-arch CUTLASS-FNA path. Bonus: cu12.8 nvcc doesn't know
# compute_103a, so dropping it makes the patch work across the whole
# matrix without per-cuda conditionals.
# Hopper: sm_90a only (the only Hopper data-center arch).
# Both OBJECT libs need torch/cutlass/natten includes propagated --
# NATTEN's target_include_directories(natten ...) call later in the
# file is target-scoped to `natten` and doesn't reach OBJECT libs.
# A compile-shard may legitimately receive ZERO files of this family: the
# shard filter partitions the ~144 autogen .cu files by GLOBAL sorted index
# (i %% shard_count), and the Hopper family has only ~22 of them -- so at
# sharding: 40 most shards get none, and add_library() with an empty source
# list is a hard cmake CONFIGURE error:
#   CMake Error at CMakeLists.txt:234 (add_library):
#     No SOURCES given to target: natten_hopper
# Skipping the target in that shard is correct -- another shard compiles
# those TUs, and the link job (shard_index 0, no deletion) builds the full
# target with every source as a cache hit. An empty CMake list is false.
#
# NOTE: bare variable names, never ${VAR}. In an if(), ${VAR} is expanded
# BEFORE parsing, so an undefined variable leaves `if(AND (...))` and cmake
# dies with "Unknown arguments specified" -- which is exactly what killed
# every Windows cu13.0 shard on 2026-08-26 (NATTEN_WITH_BLACKWELL_FNA is
# not defined when the arch list has no sm_100). A bare name is evaluated
# as a variable and an undefined one is simply false.
if(NATTEN_WITH_BLACKWELL_FNA AND (AUTOGEN_BLACKWELL_FNA OR AUTOGEN_BLACKWELL_FMHA))
    list(REMOVE_ITEM ALL_SOURCES ${AUTOGEN_BLACKWELL_FNA} ${AUTOGEN_BLACKWELL_FMHA})
    add_library(natten_blackwell OBJECT
        ${AUTOGEN_BLACKWELL_FNA} ${AUTOGEN_BLACKWELL_FMHA})
    set_target_properties(natten_blackwell PROPERTIES
        CUDA_ARCHITECTURES "100a-real"
        POSITION_INDEPENDENT_CODE ON)
    target_include_directories(natten_blackwell SYSTEM PRIVATE ${TORCH_INCLUDE_DIRS})
    target_include_directories(natten_blackwell PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/cutlass/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/autogen/include
    )
    list(LENGTH AUTOGEN_BLACKWELL_FNA  _cuw_n_bw_fna)
    list(LENGTH AUTOGEN_BLACKWELL_FMHA _cuw_n_bw_fmha)
    math(EXPR _cuw_n_bw "${_cuw_n_bw_fna} + ${_cuw_n_bw_fmha}")
    message(STATUS "cuda-wheels: ${_cuw_n_bw} Blackwell sources -> natten_blackwell OBJECT (CUDA_ARCHITECTURES=100a-real, ${_cuw_n_bw_fna} FNA + ${_cuw_n_bw_fmha} FMHA)")
    foreach(_cuw_f ${AUTOGEN_BLACKWELL_FNA} ${AUTOGEN_BLACKWELL_FMHA})
        get_filename_component(_cuw_bn ${_cuw_f} NAME)
        get_filename_component(_cuw_pd ${_cuw_f} DIRECTORY)
        get_filename_component(_cuw_pd ${_cuw_pd} NAME)
        message(STATUS "  ${_cuw_pd}/${_cuw_bn}")
    endforeach()
endif()
# A compile-shard may legitimately receive ZERO files of this family: the
# shard filter partitions the ~144 autogen .cu files by GLOBAL sorted index
# (i %% shard_count), and the Hopper family has only ~22 of them -- so at
# sharding: 40 most shards get none, and add_library() with an empty source
# list is a hard cmake CONFIGURE error:
#   CMake Error at CMakeLists.txt:234 (add_library):
#     No SOURCES given to target: natten_hopper
# Skipping the target in that shard is correct -- another shard compiles
# those TUs, and the link job (shard_index 0, no deletion) builds the full
# target with every source as a cache hit. An empty CMake list is false.
if(NATTEN_WITH_HOPPER_FNA AND (AUTOGEN_HOPPER_FNA OR AUTOGEN_HOPPER_FMHA))
    list(REMOVE_ITEM ALL_SOURCES ${AUTOGEN_HOPPER_FNA} ${AUTOGEN_HOPPER_FMHA})
    add_library(natten_hopper OBJECT
        ${AUTOGEN_HOPPER_FNA} ${AUTOGEN_HOPPER_FMHA})
    set_target_properties(natten_hopper PROPERTIES
        CUDA_ARCHITECTURES "90a-real"
        POSITION_INDEPENDENT_CODE ON)
    target_include_directories(natten_hopper SYSTEM PRIVATE ${TORCH_INCLUDE_DIRS})
    target_include_directories(natten_hopper PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/cutlass/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/autogen/include
    )
    list(LENGTH AUTOGEN_HOPPER_FNA  _cuw_n_hp_fna)
    list(LENGTH AUTOGEN_HOPPER_FMHA _cuw_n_hp_fmha)
    math(EXPR _cuw_n_hp "${_cuw_n_hp_fna} + ${_cuw_n_hp_fmha}")
    message(STATUS "cuda-wheels: ${_cuw_n_hp} Hopper sources -> natten_hopper OBJECT (CUDA_ARCHITECTURES=90a-real, ${_cuw_n_hp_fna} FNA + ${_cuw_n_hp_fmha} FMHA)")
    foreach(_cuw_f ${AUTOGEN_HOPPER_FNA} ${AUTOGEN_HOPPER_FMHA})
        get_filename_component(_cuw_bn ${_cuw_f} NAME)
        get_filename_component(_cuw_pd ${_cuw_f} DIRECTORY)
        get_filename_component(_cuw_pd ${_cuw_pd} NAME)
        message(STATUS "  ${_cuw_pd}/${_cuw_bn}")
    endforeach()
endif()
# --- end cuda-wheels arch-specific OBJECT libraries ---

add_library(natten SHARED ${ALL_SOURCES})

# if(TARGET ...), NOT if(NATTEN_WITH_*_FNA). The two are not the same
# condition and conflating them is what broke natten at sharding: 23.
# NATTEN_WITH_HOPPER_FNA is set from the ARCH LIST, so it is true in every
# shard; the target only exists if THIS shard received Hopper sources. The
# add_library() above is guarded on the source list, so a shard with zero
# Hopper files correctly skips the target -- and then this line referenced a
# target that does not exist:
#   CMake Error at CMakeLists.txt:291 (target_link_libraries):
#     Error evaluating generator expression: $<TARGET_OBJECTS:natten_hopper>
#     Objects of target "natten_hopper" referenced but no such target exists.
# Guarding the producer without guarding the consumer buys nothing. Note this
# is a CONFIGURE/GENERATE error, so the shard produces almost no objects and
# the tolerant-shard-link gate correctly refuses it ("failed without undefined
# references") rather than passing it off as the expected link failure.
if(TARGET natten_blackwell)
    target_link_libraries(natten PRIVATE $<TARGET_OBJECTS:natten_blackwell>)
endif()
if(TARGET natten_hopper)
    target_link_libraries(natten PRIVATE $<TARGET_OBJECTS:natten_hopper>)
endif()'''
if 'cuda-wheels arch-specific OBJECT libraries' in cmake_text:
    print("NOTE: arch-specific OBJECT libraries block already present in csrc/CMakeLists.txt -- skipping")
elif old_add_lib in cmake_text:
    cmake_text = cmake_text.replace(old_add_lib, new_add_lib, 1)
    cmake_file.write_text(cmake_text)
    print("Patched csrc/CMakeLists.txt: split Blackwell/Hopper into OBJECT libs around add_library(natten SHARED ...)")
else:
    raise SystemExit(
        "FATAL: anchor 'add_library(natten SHARED ${ALL_SOURCES})' not found in "
        "csrc/CMakeLists.txt -- upstream may have changed."
    )

# MSVC noise suppression. Job 76637988762's Build wheel step ran to 82k log
# lines on Windows; histogram of warnings:
#   27483 C4514 unreferenced inline function removed
#    4842 C4100 unreferenced parameter
#    1107 C4623 default constructor implicitly defined as deleted
#     486 C4577 likely mismatch, popping warning state pushed in different file
#     ...
# All from CUTLASS template instantiations under third_party/cutlass; none
# are actionable in NATTEN. Suppressing them on Windows via target_compile_options
# cuts the log to a fraction of its prior size without losing real diagnostics
# (NATTEN's own warnings stay at /W3 default).
cmake_text = cmake_file.read_text()
msvc_noise_block = '''

# --- cuda-wheels MSVC noise suppression (injected) ---
# Suppress high-volume MSVC warnings from CUTLASS template instantiations.
# Applied via target_compile_options so order vs add_library() doesn't matter.
if(NATTEN_IS_WINDOWS)
    set(_cuw_msvc_wd_codes
        # First batch (from initial 82k-line log)
        4514  # unreferenced inline function has been removed
        4100  # unreferenced parameter
        4623  # default constructor implicitly defined as deleted
        4624  # destructor implicitly defined as deleted
        4577  # likely mismatch, popping warning state pushed in different file
        4067  # unexpected tokens following preprocessor directive
        4068  # unknown pragma
        4505  # unreferenced local function has been removed
        4127  # conditional expression is constant
        # Second batch (from b535f14 follow-up Windows log job 76682941969)
        4711  # 9.5k: function selected for automatic inline expansion
        4820  # 9.2k: N bytes padding added after data member
        4061  # 7k:   enumerator not explicitly handled in switch
        4251  # 5k:   needs to have dll-interface (STL members)
        4710  # 5k:   function not inlined
        4365  # 3k:   sign-conversion (CUTLASS templates)
        4626  # 2.8k: assignment operator implicitly defined as deleted
        5027  # 2.5k: move assignment operator implicitly defined as deleted
        4996  # 1.9k: deprecated function (torch internals)
        4244  # 1k:   data-loss conversion
        4668  # 711:  not defined as preprocessor macro
        4625  # 576:  copy constructor implicitly defined as deleted
        5039  # 402:  extern-C function pointer
        4619  # 315:  no warning number 'XXXX'
        4324  # 297:  structure padded due to alignment specifier
        4267  # size_t -> 32-bit conversion
        # Third batch (from job 76714... on f4f0b66 - torch headers + CUTLASS leftovers)
        4275  # non dll-interface class used as base for dll-interface class
        4686  # possible change in behavior, change in UDT return calling convention
        4355  # 'this': used in base member initializer list
        4800  # implicit conversion to bool, possible information loss
        5031  # pragma warning(pop): likely mismatch, popping in different file
        5246  # initialization of subobject should be wrapped in braces
        5026  # move constructor implicitly defined as deleted
        4582  # constructor is not implicitly called
        4583  # destructor is not implicitly called
        4018  # signed/unsigned mismatch
        4242  # conversion, possible loss of data
        4310  # cast truncates constant value
        4459  # declaration of X hides global declaration
        4201  # nonstandard extension used: nameless struct/union
        4189  # local variable initialized but not referenced
        4191  # unsafe function pointer conversion
        # Fourth batch (from 6fd6738 follow-up)
        5219  # implicit conversion, possible loss of data (torch TypeSafeSignMath, MASSIVE)
        5045  # Compiler will insert Spectre mitigation
        4702  # unreachable code
        4868  # compiler may not enforce left-to-right evaluation order
        4388  # signed/unsigned mismatch in comparison
        4296  # expression is always true
        4464  # relative include path contains '..'
    )
    foreach(_cuw_c ${_cuw_msvc_wd_codes})
        target_compile_options(natten PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/wd${_cuw_c}>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd${_cuw_c}>
        )
        # Apply to the per-arch OBJECT libs too -- they're separate targets,
        # so target_compile_options on `natten` doesn't reach them. Without
        # this, Hopper sources on Windows emit the full CUTLASS C4514 torrent
        # (millions of lines, 100+ MB logs, hours of I/O cost -- enough to
        # blow the 6h GH cap on Windows runners).
        if(TARGET natten_blackwell)
            target_compile_options(natten_blackwell PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:/wd${_cuw_c}>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd${_cuw_c}>
            )
        endif()
        if(TARGET natten_hopper)
            target_compile_options(natten_hopper PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:/wd${_cuw_c}>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd${_cuw_c}>
            )
        endif()
    endforeach()
    list(JOIN _cuw_msvc_wd_codes "," _cuw_msvc_wd_label)
    message(STATUS "cuda-wheels: suppressed MSVC warnings C${_cuw_msvc_wd_label} on natten + per-arch OBJECT libs")
endif()
# --- end cuda-wheels MSVC noise suppression ---
'''
if 'cuda-wheels MSVC noise suppression' not in cmake_text:
    cmake_file.write_text(cmake_text + msvc_noise_block)
    print("Appended MSVC noise-suppression block to csrc/CMakeLists.txt")
else:
    print("NOTE: MSVC noise-suppression block already present in csrc/CMakeLists.txt -- skipping")

# nvcc / cudafe1 diagnostic suppression. Separate from MSVC: nvcc warnings
# use the #NNN-D form and are controlled by --diag-suppress=NNN rather than
# /wd<num>. The MSVC block above doesn't touch them. Applies on all platforms
# because nvcc runs on both Linux and Windows.
cmake_text = cmake_file.read_text()
nvcc_diag_block = '''

# --- cuda-wheels nvcc diagnostic suppression (injected) ---
set(_cuw_nvcc_diag_codes
    221    # floating-point value does not fit in required floating-point type
           # (NATTEN's fna_collective_softmax.hpp uses -((float)(1e+300)) as a
           # sentinel; the double->float cast overflows. Pure noise.)
    20011  # calling a __host__ function from __host__ __device__ context
           # (CUTLASS template noise; harmless when only host path is taken)
    # Third batch (from job 76714... follow-up Windows log)
    1394   # field of class type without a DLL interface used in a class
           # with a DLL interface (torch headers; massive volume on Windows)
    1388   # base class dllexport/dllimport spec differs from derived class
    1390   # dllexport/dllimport conflict (torch's TypeMeta::_typeMetaData)
    550    # variable was set but never used (CUTLASS cute::nullspace etc.)
)
foreach(_cuw_d ${_cuw_nvcc_diag_codes})
    target_compile_options(natten PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=${_cuw_d}>
    )
    if(TARGET natten_blackwell)
        target_compile_options(natten_blackwell PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=${_cuw_d}>
        )
    endif()
    if(TARGET natten_hopper)
        target_compile_options(natten_hopper PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=${_cuw_d}>
        )
    endif()
endforeach()
list(JOIN _cuw_nvcc_diag_codes "," _cuw_nvcc_diag_label)
message(STATUS "cuda-wheels: nvcc --diag-suppress=${_cuw_nvcc_diag_label} on natten + per-arch OBJECT libs")
# --- end cuda-wheels nvcc diagnostic suppression ---
'''
if 'cuda-wheels nvcc diagnostic suppression' not in cmake_text:
    cmake_file.write_text(cmake_text + nvcc_diag_block)
    print("Appended nvcc diag-suppression block to csrc/CMakeLists.txt")
else:
    print("NOTE: nvcc diag-suppression block already present in csrc/CMakeLists.txt -- skipping")

# csrc/include/natten/helpers.h: CHECK_CONTIGUOUS uses the C++ alternative
# token `not` (`TORCH_CHECK(not x.is_sparse(), ...)`). GCC/Clang accept this
# without <ciso646>; MSVC errors with `identifier "not" is undefined` unless
# /permissive- or /Za is set. Replace with the standard `!` operator — this
# is a one-liner in the macro definition, but the macro expands inside other
# helpers.h checks (lines ~325, ~366 in the build error), so every consumer
# is fixed by the single substitution.
helpers_file = Path("csrc/include/natten/helpers.h")
helpers_text = helpers_file.read_text()
old_check = '(not x.is_sparse(),'
new_check = '(!x.is_sparse(),'
if old_check in helpers_text:
    helpers_file.write_text(helpers_text.replace(old_check, new_check))
    print(f"Patched csrc/include/natten/helpers.h: replaced {old_check!r} with {new_check!r} (MSVC fix)")
else:
    print("NOTE: csrc/include/natten/helpers.h didn't contain 'not x.is_sparse(),' -- skipping")

setup_file = Path("setup.py")
content = setup_file.read_text()

anchor = 'CUDA_ARCH = os.getenv("NATTEN_CUDA_ARCH", "")'
shim = '''# cuda-wheels shim: bridge TORCH_CUDA_ARCH_LIST -> NATTEN_CUDA_ARCH
# and MAX_JOBS -> NATTEN_N_WORKERS so the cuda-wheels build harness can
# drive NATTEN with its standard env vars. See patches/natten.py.
if not os.getenv("NATTEN_CUDA_ARCH"):
    _torch_arch = os.getenv("TORCH_CUDA_ARCH_LIST", "")
    _raw = [p.strip() for p in _torch_arch.replace(";", " ").split() if p.strip()]
    _parts = [p.replace("+PTX", "").strip() for p in _raw]
    _parts = [p for p in _parts if p]
    if _parts:
        os.environ["NATTEN_CUDA_ARCH"] = ";".join(_parts)
    # Remember WHICH archs carried +PTX before we strip it. NATTEN's parser
    # rejects the suffix, so it has to come off -- but dropping it on the
    # floor is why every natten wheel shipped SASS only and was rejected by
    # C7 with "declared +PTX for ['sm_120'] but shipped NO PTX ... no JIT
    # path onto newer GPUs". Upstream's own knob (NATTEN_BUILD_WITH_PTX=1)
    # is all-or-nothing: it emits PTX for EVERY arch in the list, which is
    # both bloat and, for 90/100/103, the useless `a` form. Carry the exact
    # set instead and let arch_list_to_cmake_tags below honour it.
    # Stored in NATTEN's own int form (8.6 -> 86, per _check_cuda_arch).
    if not os.getenv("CUW_NATTEN_PTX_ARCH"):
        _ptx = [p.replace("+PTX", "").strip() for p in _raw if "+PTX" in p]
        _ptx_i = []
        for _p in _ptx:
            try:
                _ptx_i.append(str(int(float(_p) * 10)))
            except ValueError:
                pass
        if _ptx_i:
            os.environ["CUW_NATTEN_PTX_ARCH"] = ";".join(_ptx_i)
            print(f"[cuda-wheels] PTX tail requested for sm_{_ptx_i}")
if not os.getenv("NATTEN_N_WORKERS"):
    _mj = os.getenv("MAX_JOBS", "")
    if _mj.isdigit() and int(_mj) > 0:
        os.environ["NATTEN_N_WORKERS"] = _mj
# The Windows Blackwell-DC strip that used to live here is GONE. It read:
#     if platform.system() == "Windows":
#         drop 10.0 / 10.3 from NATTEN_CUDA_ARCH
# because MSVC's strict mode rejects
# sm100_fmha_bwd_kernel_tma_warpspecialized.hpp with C2061.
#
# The limitation is real. Expressing it HERE was the mistake. An arch list is
# configuration: the resolver reads the YAML, the gate resolves the same YAML,
# and a human reading arch_override.yml is entitled to believe it. This patch
# silently disagreed with all three, so C7 failed every natten Windows wheel
# with `missing arch families sm_[10]` and there was no honest way to fix it --
# narrowing the shared x86 row to match would have stripped Blackwell from
# Linux too, and Windows had no arch field of its own.
#
# It does now: `arch_list_by_cuda_windows` in arch_override.yml carries the
# Windows list, with the C2061 reason written down next to it. A patch's job is
# to PLUG THE LIST IN -- translate the farm's arch list into whatever env var
# or gencode form upstream wants (the shim above does exactly that) -- never to
# decide its contents.
# Pin NATTEN_BUILD_DIR to a predictable in-source location so the cuda-wheels
# shard/link harness can find the .o files. Default is a temporary directory
# whose name changes per run, which doesn't survive the upload/restore
# handoff between shard compile jobs and the downstream link job.
# IMPORTANT: NATTEN's setup.py at line 67-68 falls back to the tempdir if
# the directory doesn't exist:
#   if not os.path.isdir(BUILD_DIR):
#       BUILD_DIR = tmp_dir.name
# so we must create the directory before the env var is read.
if not os.getenv("NATTEN_BUILD_DIR"):
    _cuw_natten_build_dir = os.path.abspath("build/natten_cmake")
    os.makedirs(_cuw_natten_build_dir, exist_ok=True)
    os.environ["NATTEN_BUILD_DIR"] = _cuw_natten_build_dir
    print(f"[cuda-wheels] NATTEN_BUILD_DIR set to {_cuw_natten_build_dir}")
''' + anchor

if anchor in content:
    content = content.replace(anchor, shim, 1)
    print("Patched setup.py: TORCH_CUDA_ARCH_LIST -> NATTEN_CUDA_ARCH and MAX_JOBS -> NATTEN_N_WORKERS shim inserted")
else:
    raise SystemExit(
        "FATAL: anchor 'CUDA_ARCH = os.getenv(\"NATTEN_CUDA_ARCH\", \"\")' "
        "not found in setup.py -- upstream may have changed. Re-check the "
        "patch against the pinned source_tag."
    )

# --- PTX tail (pairs with the CUW_NATTEN_PTX_ARCH capture in the shim) -------
# Teach arch_list_to_cmake_tags to emit `-virtual` for exactly the archs the
# farm marked +PTX. Without this natten builds `-real` only, ships no PTX at
# all, and C7 rejects every wheel -- which is why the release currently holds
# zero natten assets despite ~70 runner-hours per pass.
ptx_anchor = """    if WITH_PTX:
        ptx_tags = (
            "-virtual;".join(
                [str(x) if x not in [90, 100, 103] else f"{x}a" for x in arch_list]
            )
            + "-virtual"
        )

        return real_tags + ";" + ptx_tags
    return real_tags"""

ptx_replacement = """    # --- cuda-wheels: farm-directed PTX tail (see patches/natten.py) ---
    # Upstream's WITH_PTX is all-or-nothing and maps 90/100/103 to the
    # arch-conditional `90a`/`100a`/`103a` form. Arch-conditional PTX can only
    # ever JIT back onto that same arch, so it is not a forward-compat tail --
    # and verify_wheel.py's _cuobjdump_archs drops `a` targets from the PTX set
    # for precisely that reason, so emitting it would satisfy nobody.
    # Emit BASE-arch virtual tags for the farm's +PTX archs only.
    _cuw_ptx = []
    for _a in os.getenv("CUW_NATTEN_PTX_ARCH", "").replace(";", " ").split():
        try:
            _v = int(_a)
        except ValueError:
            continue
        if _v in arch_list and _v not in _cuw_ptx:
            _cuw_ptx.append(_v)
    if _cuw_ptx:
        _cuw_tags = "-virtual;".join(str(x) for x in _cuw_ptx) + "-virtual"
        print(f"[cuda-wheels] CUDA_ARCHITECTURES PTX tail: {_cuw_tags}")
        return real_tags + ";" + _cuw_tags
    # --- end cuda-wheels ---
    if WITH_PTX:
        ptx_tags = (
            "-virtual;".join(
                [str(x) if x not in [90, 100, 103] else f"{x}a" for x in arch_list]
            )
            + "-virtual"
        )

        return real_tags + ";" + ptx_tags
    return real_tags"""

if ptx_anchor in content:
    content = content.replace(ptx_anchor, ptx_replacement, 1)
    print("Patched setup.py: arch_list_to_cmake_tags honours CUW_NATTEN_PTX_ARCH")
else:
    raise SystemExit(
        "FATAL: arch_list_to_cmake_tags' WITH_PTX branch not found in setup.py. "
        "Without it natten ships SASS-only wheels that C7 rejects for every "
        "cell. Re-check the patch against the pinned source_tag."
    )

# Shard filter for cmake-style build:
# Inject a post-autogen filter that deletes .cu files NOT in this shard's
# slice, gated on CUDA_WHEELS_SHARD_INDEX/COUNT env vars (set by the
# build-wheel action in compile-shard mode). The shard then compiles only
# its slice (1/N of the 144 autogen .cu files). When the env vars are
# unset (link-only mode, full-mode), this is a no-op and natten builds
# normally.
#
# Sorted round-robin partition: each .cu file at sorted-index i goes to
# shard (i % shard_count) + 1. Deterministic across shards (each shard sees
# the same sort order from autogen's deterministic output).
autogen_anchor = """            autogen_kernel_instantitations(
                this_dir=this_dir,
                autogen_dir=autogen_dir,
                scripts_dir=scripts_dir,
                policy=AUTOGEN_POLICY,
                cuda_arch_list=cuda_arch_list,
            )"""

autogen_filter = autogen_anchor + """

            # cuda-wheels shard filter: when CUDA_WHEELS_SHARD_COUNT > 0,
            # delete .cu files not in this shard's slice so cmake only
            # builds 1/N of the autogen output. See patches/natten.py.
            _cuw_shard_count = int(os.environ.get('CUDA_WHEELS_SHARD_COUNT', '0'))
            if _cuw_shard_count > 0:
                import glob
                _cuw_shard_index = int(os.environ.get('CUDA_WHEELS_SHARD_INDEX', '1'))
                _cuw_pattern = path.join(autogen_dir, 'src', 'cuda', '**', '*.cu')
                _cuw_all = sorted(glob.glob(_cuw_pattern, recursive=True))
                # csrc/src/*.cu -- the hand-written torch-API dispatch layer,
                # picked up by cmake's TORCH_APIS glob (csrc/CMakeLists.txt:164)
                # and NOT under autogen/, so the pattern above never saw them.
                # All 14 were therefore compiled by EVERY shard: they pull in
                # the same CUTLASS headers as the autogen kernels, cost ~27
                # minutes, and put a floor under the shard wall clock that no
                # `sharding:` value could lower. Partition them with the same
                # round-robin so each is compiled once across the fleet; the
                # link job rebuilds the full set from the merged cache.
                _cuw_shared = sorted(glob.glob(
                    path.join(path.dirname(autogen_dir), 'src', '*.cu')))
                _cuw_all = _cuw_all + _cuw_shared
                _cuw_kept = [f for i, f in enumerate(_cuw_all)
                             if i % _cuw_shard_count == _cuw_shard_index - 1]
                _cuw_to_delete = set(_cuw_all) - set(_cuw_kept)
                for _f in _cuw_to_delete:
                    os.remove(_f)
                _cuw_kept_shared = [f for f in _cuw_kept if f in set(_cuw_shared)]
                print(f'[cuda-wheels natten shard {_cuw_shard_index}/{_cuw_shard_count}] '
                      f'kept {len(_cuw_kept)}/{len(_cuw_all)} .cu files '
                      f'({len(_cuw_kept_shared)}/{len(_cuw_shared)} shared torch-API); '
                      f'deleted {len(_cuw_to_delete)}')
"""

if autogen_anchor in content:
    content = content.replace(autogen_anchor, autogen_filter, 1)
    print("Patched setup.py: autogen shard filter injected after autogen_kernel_instantitations() call")
else:
    raise SystemExit(
        "FATAL: anchor for autogen call not found in setup.py -- upstream may have changed. "
        "Re-check the patch against the pinned source_tag."
    )

# Sequential-checkpoint: skip cmake configure on resume. Proven on
# natten_sequential (run 26052223468) -- combined with CMAKE_SUPPRESS_REGENERATION
# below and the action.yml's sudo-touch of CUDA targets/ headers, ninja
# correctly resumes the build mid-chain. Without this patch, NATTEN's setup.py
# always runs cmake configure, which regenerates build.ninja with subtly
# different command strings and trips ninja's command_hash check.
old_cmake_pair = '''            # Config and build the extension
            subprocess.check_call(
                ["cmake", cmake_lists_dir] + cmake_args, cwd=build_dir
            )
            cmake_build_args = [
                "--build",
                build_dir,
                "-j",
                str(N_WORKERS),
            ]
            if VERBOSE:
                cmake_build_args.append("--verbose")
            subprocess.check_call(["cmake", *cmake_build_args])'''

new_cmake_pair = '''            # cuda-wheels sequential-checkpoint: skip the cmake configure call
            # on resume so the restored build.ninja is honored byte-identically
            # by ninja. A fresh configure re-emits build.ninja with subtly
            # different compile-command strings, forcing a full rebuild via
            # the command_hash check.
            _cuw_cmake_cache = os.path.join(build_dir, "CMakeCache.txt")
            _cuw_build_ninja = os.path.join(build_dir, "build.ninja")
            if os.path.isfile(_cuw_cmake_cache) and os.path.isfile(_cuw_build_ninja):
                print(f"[cuda-wheels] reusing CMakeCache.txt + build.ninja in {build_dir}; skipping configure")
            else:
                subprocess.check_call(
                    ["cmake", cmake_lists_dir] + cmake_args, cwd=build_dir
                )
            cmake_build_args = [
                "--build",
                build_dir,
                "-j",
                str(N_WORKERS),
            ]
            if VERBOSE:
                cmake_build_args.append("--verbose")
            subprocess.check_call(["cmake", *cmake_build_args])'''

if old_cmake_pair in content:
    content = content.replace(old_cmake_pair, new_cmake_pair, 1)
    print("Patched setup.py: skip cmake configure on resume when CMakeCache.txt + build.ninja exist")
else:
    raise SystemExit(
        "FATAL: cmake configure block not found in NATTEN setup.py -- upstream may have changed."
    )

# Inject -DCMAKE_SUPPRESS_REGENERATION=TRUE into cmake_args. Without it,
# cmake's Ninja generator emits a RERUN_CMAKE edge that ninja itself fires
# on chain-link resume (visible as "[0/1] Re-running CMake..."), re-running
# cmake configure as a ninja edge -- defeating the setup.py-level skip above.
old_cmake_args = '''            cmake_args = [
                f"-DPYTHON_PATH={sys.executable}",'''

new_cmake_args = '''            cmake_args = [
                "-DCMAKE_SUPPRESS_REGENERATION=TRUE",
                f"-DPYTHON_PATH={sys.executable}",'''

if old_cmake_args in content:
    content = content.replace(old_cmake_args, new_cmake_args, 1)
    print("Patched setup.py: injected -DCMAKE_SUPPRESS_REGENERATION=TRUE into cmake_args")
else:
    raise SystemExit(
        "FATAL: cmake_args initializer not found in NATTEN setup.py -- upstream may have changed."
    )

setup_file.write_text(content)
