{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs";
    poetry2nix = {
      url = "github:nix-community/poetry2nix";
      inputs = {
        nixpkgs.follows = "nixpkgs";
        flake-utils.follows = "flake-utils";
      };
    };
    pre-commit-hooks = {
      url = "github:cachix/git-hooks.nix";
      inputs = {
        nixpkgs.follows = "nixpkgs";
      };
    };
  };

  outputs = { self, nixpkgs, flake-utils, poetry2nix, pre-commit-hooks, ... }:
    flake-utils.lib.eachSystem [ "x86_64-linux" "aarch64-linux" ] (system:
      let
        projectConfig = {
          python = pkgs.python311;
          dependencyOverrides = (final: prev: {
            docutils = prev.docutils.overridePythonAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ [ final.flit-core ];
            });
            lgpio = prev.lgpio.overridePythonAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ [
                final.setuptools
                (pkgs.stdenv.mkDerivation rec {
                  pname = "lgpio";
                  version = "0.2.2";
                  src = pkgs.fetchFromGitHub {
                    owner = "joan2937";
                    repo = "lg";
                    rev = "v${version}";
                    hash = "sha256-92lLV+EMuJj4Ul89KIFHkpPxVMr/VvKGEocYSW2tFiE=";
                  };
                  makeFlags = [ "prefix=$(out)" ];
                  postConfigure = "substituteInPlace Makefile --replace ldconfig 'echo ldconfig'";
                })
              ];
              nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ pkgs.swig ];
            });
            mypy = prev.mypy.override {
              preferWheel = true;
            };
            jiter = prev.jiter.override {
              preferWheel = true;
            };
            fsspec = prev.fsspec.overridePythonAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ [ final.hatchling final.hatch-vcs ];
            });
            to-file-like-obj = prev.to-file-like-obj.overridePythonAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ [ final.hatchling ];
            });
            tokenizers = prev.tokenizers.override {
              preferWheel = true;
            };
            numpy = prev.numpy.override {
              preferWheel = true;
            };
            pasimple = prev.pasimple.overridePythonAttrs (old: {
              postPatch = (old.postPatch or "") + ''
                substituteInPlace pasimple/pa_simple.py --replace \
                  "_libpulse_simple = ctypes.CDLL('libpulse-simple.so.0')" \
                  "_libpulse_simple = ctypes.CDLL('${pkgs.lib.getLib pkgs.pulseaudio}/lib/libpulse-simple.so.0')"
              '';
            });
            av = prev.av.override {
              preferWheel = true;
            };
            elevenlabs = prev.elevenlabs.overridePythonAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ [ final.poetry-core ];
            });
            streamp3 = prev.streamp3.overridePythonAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ [ final.setuptools pkgs.lame ];
              nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ final.cython_0 ];
            });
            openai = prev.openai.overridePythonAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ [ final.hatch-fancy-pypi-readme ];
            });
            anthropic = prev.anthropic.overridePythonAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ [ final.hatch-fancy-pypi-readme ];
            });
            sounddevice = prev.sounddevice.overridePythonAttrs (old: {
              patches = (old.patches or [ ]) ++ pkgs.python3Packages.sounddevice.patches;
            });
          });
        };
        pkgs = nixpkgs.legacyPackages.${system};
        inherit (poetry2nix.lib.mkPoetry2Nix { pkgs = pkgs; }) mkPoetryApplication mkPoetryEnv defaultPoetryOverrides cleanPythonSources cli;
        pyProject = builtins.fromTOML (builtins.readFile (./. + "/pyproject.toml"));
        mkPoetryArgs = {
          overrides = [ projectConfig.dependencyOverrides defaultPoetryOverrides ];
          python = projectConfig.python;
          projectDir = ./.;
        };
        nixpkgsAttrMap = attrnames: builtins.map (p: pkgs.${p}) attrnames;
        pyProjectNixpkgsDeps = (nixpkgsAttrMap (pyProject.tool.nixpkgs.dependencies or [ ])) ++ [
          (pkgs.libcamera.override { python3 = projectConfig.python; })
        ];
        pyProjectNixpkgsDevDeps = pyProjectNixpkgsDeps ++ nixpkgsAttrMap (pyProject.tool.nixpkgs.dev-dependencies or [ ]);
        # HACK work around a bug in poetry2nix where the .egg-info is named incorrectly
        # https://github.com/nix-community/poetry2nix/issues/616
        pkgInfoFields = {
          Metadata-Version = "2.1";
          Name = pyProject.tool.poetry.name;
          Version = pyProject.tool.poetry.version;
          Summary = pyProject.tool.poetry.description;
        };
        pkgInfoFile = with pkgs.lib.generators; (
          builtins.toFile "${pyProject.tool.poetry.name}-PKG-INFO"
            (toKeyValue { mkKeyValue = mkKeyValueDefault { } ": "; } pkgInfoFields)
        );
        moduleNames = (
          pkgs.lib.attrNames
            (
              pkgs.lib.filterAttrs
                (n: v: v == "directory")
                (builtins.readDir srcDir)
            )
        );
        editableEggInfoFix = ps:
          (ps.toPythonModule (
            pkgs.runCommand "editable-egg-info-fix" { } ''
              mkdir -p "$out/${ps.python.sitePackages}"
              cd "$out/${ps.python.sitePackages}"
              ${
                pkgs.lib.concatMapStringsSep
                "\n"
                (pkg: (
                    if (pkg != pyProject.tool.poetry.name)
                    then ''mkdir "${pkg}.egg-info"; ln -s "${pkgInfoFile}" "${pkg}.egg-info/PKG-INFO"''
                    else ""
                ))
                moduleNames
              }
            ''));
        # use impure flake in direnv to get live editing for mkPoetryEnv
        envProjectDir = builtins.getEnv "PROJECT_DIR";
        srcDir = (if envProjectDir == "" then ./src else "${envProjectDir}/src");
        mkPoetryEnvEditableArgs = {
          editablePackageSources.${pyProject.tool.poetry.name} = srcDir;
          extraPackages = (ps: [ (editableEggInfoFix ps) ]);
        };
      in
      rec {
        packages = {
          default = (mkPoetryApplication mkPoetryArgs).overrideAttrs (old: {
            propagatedBuildInputs = (old.propagatedBuildInputs or [ ]) ++ pyProjectNixpkgsDeps;
          });
          containerImage = (pkgs.dockerTools.streamLayeredImage {
            name = pyProject.tool.poetry.name;
            # add deps to contents so they can be invoked directly if needed (e.g. /bin/someprogram)
            contents = [ packages.default ] ++ pyProjectNixpkgsDeps;
            config.Cmd = [ "/bin/${pyProject.tool.poetry.name}" ];
          }).overrideAttrs (old: { passthru.exePath = ""; });
        };
        apps = pkgs.lib.mapAttrs (k: v: flake-utils.lib.mkApp { drv = v; }) packages;
        devShells =
          let
            baseShellPackages = [
              pkgs.act
              (pkgs.poetry.withPlugins (ps: with ps; [ poetry-plugin-up ]))
              cli
            ];
          in
          {
            default = (
              (mkPoetryEnv (mkPoetryArgs // mkPoetryEnvEditableArgs)).env.overrideAttrs (
                oldAttrs: {
                  buildInputs = baseShellPackages ++ pyProjectNixpkgsDevDeps;
                  shellHook = ''
                    ${checks.pre-commit-hooks.shellHook}
                  '';
                  "SHELL_LOADED_${builtins.hashString "sha256" envProjectDir}" = "1";
                }
              )
            );
            poetryFallback = pkgs.mkShell {
              packages = baseShellPackages ++ [ projectConfig.python ];
            };
          };
        checks.pre-commit-hooks = pre-commit-hooks.lib.${pkgs.system}.run {
          src = ./.;
          hooks =
            let
              poetryPreCommit = name: text: (pkgs.writeShellApplication
                {
                  name = "pre-commit-${name}";
                  runtimeInputs = with devShells.default; (nativeBuildInputs ++ buildInputs);
                  text = text;
                } + "/bin/pre-commit-${name}");
            in
            {
              shellcheck.enable = true;
              black = {
                enable = true;
                entry = pkgs.lib.mkForce (poetryPreCommit "black" ''black "$@"'');
              };
              nixpkgs-fmt.enable = true;
              prettier = {
                enable = true;
                types_or = [ "markdown" "json" "yaml" ];
                excludes = [ "^\\.template/.+/\\.cruft\\.json$" ];
              };
              isort = {
                enable = true;
                entry = pkgs.lib.mkForce (poetryPreCommit "isort" ''isort "$@"'');
              };
              mypy = {
                enable = true;
                name = "mypy";
                entry = pkgs.lib.mkForce (poetryPreCommit "mypy" ''mypy "$@"'');
                pass_filenames = false;
              };
              pytest = {
                enable = true;
                name = "pytest";
                entry = poetryPreCommit "pytest" ''
                  # HACK force path to be in scope for flake evaluation
                  # ${ cleanPythonSources { src = ./.; } + "/src" }
                  pytest "$@"
                '';
                pass_filenames = false;
              };
              sphinx = {
                enable = true;
                name = "sphinx";
                entry = poetryPreCommit "sphinx" "sphinx-build docs/ docs/generated/";
                pass_filenames = false;
              };
              no-merge-rejects = {
                enable = true;
                name = "no-merge-rejects";
                entry = poetryPreCommit "no-merge-rejects" ''
                  for filename in "$@"; do
                    echo "Rejected merge file exists: $filename"
                  done
                  exit 1
                '';
                files = "\\.rej$";
              };
            };
        };
      });
}


