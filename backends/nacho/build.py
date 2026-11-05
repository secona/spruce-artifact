import argparse
import os
import shlex
import subprocess
from pathlib import Path


def get_nacho_compiler_exe():
    backend_dir = Path(__file__).resolve().parent
    compiler = backend_dir / "compiler"
    if not compiler.is_file():
        print("--- Building Nacho compiler")
        nacho_dir = backend_dir.parents[1] / "third_party" / "nacho"
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
                "-I", str(nacho_dir / "include"),
                str(backend_dir / "compiler.cpp"),
                str(library),
                "-o", str(compiler),
            ],
            check=True,
        )
    return compiler

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", choices=("spgemm",))
    args = parser.parse_args()

    compiler = get_nacho_compiler_exe()
    output_dir = Path(__file__).resolve().parents[2] / "generated" / "nacho"
    command = [str(compiler), "--output-dir", str(output_dir)]
    if args.operation:
        command.extend(["--kernels", args.operation])
    subprocess.run(command, check=True)
    print(output_dir)
