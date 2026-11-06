{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  inputs.nixpkgs-llvm14.url = "github:NixOS/nixpkgs/nixos-24.11";

  outputs = { nixpkgs, nixpkgs-llvm14, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
      legacyPkgs = import nixpkgs-llvm14 { inherit system; };
      llvm14 = legacyPkgs.llvmPackages_14.llvm;
      cuda = pkgs.cudaPackages_13;
      cudaToolkit = pkgs.symlinkJoin {
        name = "spruce-cuda-toolkit";
        paths = [
          cuda.cuda_nvcc
          cuda.cuda_crt
          cuda.cuda_cudart
          (pkgs.lib.getOutput "include" cuda.cuda_nvrtc)
          (pkgs.lib.getLib cuda.cuda_nvrtc)
          (pkgs.lib.getDev cuda.cccl)
          (pkgs.lib.getOutput "include" cuda.libcusparse)
          (pkgs.lib.getLib cuda.libcusparse)
        ];
      };
    in
    {
      devShells.${system}.default = (pkgs.mkShell.override { stdenv = cuda.backendStdenv; }) {
        packages = with pkgs; [
          python313
          cmake
          ninja
          pkg-config
          git
        ];
        buildInputs = [ cudaToolkit pkgs.tbb pkgs.zlib pkgs.libxml2 ];

        CUDA_ROOT = "${cudaToolkit}";
        CUDA_TOOLKIT_ROOT = "${cudaToolkit}";
        CUDA_PATH = "${cudaToolkit}";
        CUDAToolkit_ROOT = "${cudaToolkit}";
        CUDACXX = "${cudaToolkit}/bin/nvcc";
        LLVM_14_BUILD_DIR = "${llvm14.dev}";

        shellHook = ''
          export CMAKE_LIBRARY_PATH="/run/opengl-driver/lib:${cudaToolkit}/lib''${CMAKE_LIBRARY_PATH:+:$CMAKE_LIBRARY_PATH}"
          export LD_LIBRARY_PATH="/run/opengl-driver/lib:${pkgs.lib.makeLibraryPath [ cuda.backendStdenv.cc.cc.lib pkgs.zlib pkgs.libxml2 pkgs.tbb cudaToolkit ]}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

          if [ ! -d .venv ]; then
            ${pkgs.python313}/bin/python -m venv .venv
          fi
          source .venv/bin/activate
        '';
      };
    };
}
