import os
import shlex
import subprocess
from pathlib import Path
from benchmarks.args import parse_build_args


OPERATIONS = ("spgemm",)


def main():
    args = parse_build_args(OPERATIONS)

    compiler = get_nacho_compiler_exe()
    output_dir = Path("generated/nacho")
    command = [str(compiler), "--output-dir", str(output_dir)]
    if args.operation:
        command.extend(["--kernels", args.operation])
    subprocess.run(command, check=True)
    print(output_dir)

    backend_dir = Path("systems/nacho")
    library = output_dir / "libnacho_spgemm.so"
    subprocess.run(
        [
            *shlex.split(os.environ.get("NVCC", "nvcc")),
            f"-arch={args.gpu_arch}",
            "-O3",
            "-std=c++17",
            "--expt-extended-lambda",
            "-shared",
            "-Xcompiler=-fPIC",
            "-I",
            str(output_dir),
            "-I",
            str(Path("third_party/nacho") / "runtime"),
            str(backend_dir / "benchmark.cu"),
            str(output_dir / "spgemm_gpu.cu"),
            "-o",
            str(library),
        ],
        check=True,
    )
    print(library)


def get_nacho_compiler_exe():
    backend_dir = Path("systems/nacho")
    compiler = backend_dir / "compiler"
    if not compiler.is_file():
        print("--- Building Nacho compiler")
        nacho_dir = Path("third_party/nacho")
        build_dir = nacho_dir / "build"
        library = build_dir / "libnacho_library.a"
        if not library.is_file():
            subprocess.run(
                ["cmake", "-S", str(nacho_dir), "-B", str(build_dir)], check=True
            )
            subprocess.run(
                ["cmake", "--build", str(build_dir), "--target", "nacho_library"],
                check=True,
            )
        subprocess.run(
            [
                *shlex.split(os.environ.get("CXX", "c++")),
                "-O2",
                "-std=c++20",
                "-I",
                str(nacho_dir / "include"),
                str(backend_dir / "compiler.cpp"),
                str(library),
                "-o",
                str(compiler),
            ],
            check=True,
        )
    return compiler


if __name__ == "__main__":
    main()
