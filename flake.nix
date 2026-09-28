{
  description = "Rust Raytracer";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { nixpkgs, ... }:
    let
      systems = [
        # No other systems were tested
        "x86_64-linux"
      ];

      forEachSystem = f:
        nixpkgs.lib.genAttrs systems (system:
          f (import nixpkgs {
            inherit system;
          })
        );
    in
      {
        devShells = forEachSystem (pkgs: {
          default =
            pkgs.mkShell rec {
              nativeBuildInputs = with pkgs; [
                pkg-config
                cargo
                rustc
                rust-analyzer
              ];

              buildInputs = with pkgs; [
                libGL
                libxkbcommon
                wayland
              ];

              LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath buildInputs;
            };
        });

        formatter = forEachSystem (pkgs: pkgs.nixfmt);
      };
}
