## PR 59
### 1. Changed
- feat: Refactor dxmz setup and update dx_com version
  - [setup.sh] Adds color-coded logging and common utility scripts.
- docs: Improve documentation for benchmark command (remove releated content)
  - This commit updates the Benchmark documentation, correcting a typo and removing the integrated benchmarking section. The integrated benchmarking section was removed(temporary).
- docs: Update Docker installation instructions in README.md
### 2. Fixed
### 3. Added
## PR 58
### 1. Changed
- chore: remove Performance data table in 'docs/source/index.md' [TFT-102](https://deepx.atlassian.net/browse/TFT-102)
### 2. Fixed
- fix: Differentiate venv paths for local and container installations [TFT-102](https://deepx.atlassian.net/browse/TFT-102)
  - When  is executed for a local (host) installation after a Docker-based setup has already created a virtual environment, a conflict can arise. Both installation methods would attempt to use the same venv path (), potentially leading to a corrupted environment.
    **Container Mode Path:** 
    **Local Mode Path:** 
### 3. Added
