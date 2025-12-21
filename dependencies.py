import os
import pkg_resources
import shutil
import subprocess
import sys
from pathlib import Path


add_on_path = Path(__file__).parent                     # assuming this file is at root of add-on
os.environ["ADDON_PATH"] = str(add_on_path)
requirements_txt = add_on_path / 'requirements.txt'     # assuming requirements.txt is at root of add-on
requirements_for_check_txt = add_on_path / 'requirements_for_check.txt'     # assuming requirements.txt is at root of add-on
DA3_DIR = add_on_path / "da3_repo"

deps_path = add_on_path / 'deps_public'                 # might not exist until install_deps is called
deps_path_da3 = add_on_path / 'deps_da3'
# Append dependencies folder to system path so we can import
# (important for Windows machines, but less so for Linux)
sys.path.insert(0, os.fspath(deps_path))
sys.path.insert(0, os.fspath(deps_path_da3))
sys.path.insert(0, os.fspath(DA3_DIR))
OPENCV_PINNED = "opencv-python==4.11.0.86"


class Dependencies:
    # cache variables used to eliminate unnecessary computations
    _checked = None
    _requirements = None

    @staticmethod
    def install():
        if Dependencies.check():
            return True

        # Create folder into which pip will install dependencies
        if not os.path.exists(DA3_DIR):
            try:
                subprocess.check_call([
                    'git',
                    'clone',
                    '--recursive',
                    'https://github.com/ByteDance-Seed/Depth-Anything-3.git',
                    os.fspath(DA3_DIR),
                ])
            except subprocess.CalledProcessError as e:
                print(f'Caught Exception while trying to git clone da3')
                print(f'  Exception: {e}')
                return False

        # Ensure submodules are present (covers previous non-recursive clones)
        if not Dependencies._update_submodules():
            return False
        
        try:
            deps_path.mkdir(exist_ok=True)
        except Exception as e:
            print(f'Caught Exception while trying to create dependencies folder')
            print(f'  Exception: {e}')
            print(f'  Folder: {deps_path}')
            return False
        try:
            deps_path_da3.mkdir(exist_ok=True)
        except Exception as e:
            print(f'Caught Exception while trying to create dependencies folder')
            print(f'  Exception: {e}')
            print(f'  Folder: {deps_path_da3}')
            return False
        # Ensure pip is installed
        try:
            subprocess.check_call([sys.executable, "-m", "ensurepip", "--upgrade"])
        except subprocess.CalledProcessError as e:
            print(f'Caught CalledProcessError while trying to ensure pip is installed')
            print(f'  Exception: {e}')
            print(f'  {sys.executable=}')
            return False

        # Install dependencies from requirements.txt
        try:
            cmd = [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-r",
                os.fspath(requirements_txt),
                "--target",
                os.fspath(deps_path)
            ]
            print(f'Installing: {cmd}')
            subprocess.check_call(cmd)
            # Force the pinned OpenCV to avoid newer wheels pulling numpy>=2
            Dependencies._ensure_pinned_opencv()
        except subprocess.CalledProcessError as e:
            print(f'Caught CalledProcessError while trying to install dependencies')
            print(f'  Exception: {e}')
            print(f'  Requirements: {requirements_txt}')
            print(f'  Folder: {deps_path}')
            return False
        # Install dependencies from requirements.txt

        try:
            cmd = [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                os.fspath(DA3_DIR),
                "--target",
                os.fspath(deps_path_da3)
            ]
            print(f'Installing: {cmd}')
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError as e:
            print(f'Caught CalledProcessError while trying to install DA3')
            print(f'  Exception: {e}')
            print(f'  Requirements: {DA3_DIR}')
            return False
        # Install streaming dependencies
        if not Dependencies.install_streaming_deps():
            print("Warning: Streaming dependencies installation failed, but continuing.")
        return Dependencies.check(force=True)

    @staticmethod
    def update_da3_repo():
        """Pull the latest DA3 repo (with submodules) and reinstall into deps_da3."""
        try:
            if not DA3_DIR.exists() or not (DA3_DIR / ".git").exists():
                print('DA3 repo missing, cloning recursively...')
                subprocess.check_call([
                    'git',
                    'clone',
                    '--recursive',
                    'https://github.com/ByteDance-Seed/Depth-Anything-3.git',
                    os.fspath(DA3_DIR),
                ])
            else:
                subprocess.check_call(['git', '-C', os.fspath(DA3_DIR), 'pull', '--ff-only'])
                if not Dependencies._update_submodules():
                    return False

            # Reinstall DA3 into deps_da3 to keep it in sync with the working tree
            if deps_path_da3.exists():
                shutil.rmtree(deps_path_da3)
            deps_path_da3.mkdir(exist_ok=True)

            cmd = [
                sys.executable,
                '-m',
                'pip',
                'install',
                '--no-deps',
                os.fspath(DA3_DIR),
                '--target',
                os.fspath(deps_path_da3),
            ]
            print(f'Installing updated DA3 into deps_da3: {cmd}')
            subprocess.check_call(cmd)

            Dependencies._checked = None
            return True
        except subprocess.CalledProcessError as e:
            print(f'Caught CalledProcessError while updating DA3 repo')
            print(f'  Exception: {e}')
            return False
        except Exception as e:
            print(f'Caught Exception while updating DA3 repo')
            print(f'  Exception: {e}')
            return False

    @staticmethod
    def install_streaming_deps():
        """Install DA3-Streaming extra requirements into deps_public with fallbacks."""
        streaming_reqs = DA3_DIR / 'da3_streaming' / 'requirements.txt'
        if not streaming_reqs.exists():
            print(f"Streaming requirements not found: {streaming_reqs}")
            return False

        try:
            # Primary attempt: full requirements
            cmd = [
                sys.executable,
                '-m',
                'pip',
                'install',
                '-r',
                os.fspath(streaming_reqs),
                '--target',
                os.fspath(deps_path),
            ]
            print(f'Installing streaming deps: {cmd}')
            subprocess.check_call(cmd)
            Dependencies._ensure_pinned_opencv()
            Dependencies._ensure_pypose_installed()
            Dependencies._ensure_sklearn_installed()
            return True
        except subprocess.CalledProcessError as e:
            print('Streaming deps install failed, retrying with faiss-cpu fallback and without pypose...')
            try:
                # Fallback: install without faiss-gpu/pypose, add faiss-cpu
                # Remove or override problematic packages by installing the desired ones explicitly
                fallback_cmds = [
                    [
                        sys.executable,
                        '-m',
                        'pip',
                        'install',
                        'faiss-cpu',
                        '--target',
                        os.fspath(deps_path),
                    ],
                    [
                        sys.executable,
                        '-m',
                        'pip',
                        'install',
                        'pandas',
                        'prettytable',
                        'einops',
                        'safetensors',
                        'numba',
                        'scikit-learn',
                        '--target',
                        os.fspath(deps_path),
                    ],
                ]
                for fc in fallback_cmds:
                    print(f'Installing streaming fallback chunk: {fc}')
                    subprocess.check_call(fc)
                Dependencies._ensure_pinned_opencv()
                Dependencies._ensure_pypose_installed()
                Dependencies._ensure_sklearn_installed()
                return True
            except subprocess.CalledProcessError as e2:
                print(f'Fallback streaming deps install failed: {e2}')
                return False
        except Exception as e:
            print(f'Caught Exception while installing streaming deps: {e}')
            return False

    @staticmethod
    def _ensure_pinned_opencv():
        """Force reinstall of the pinned OpenCV version to avoid numpy>=2 pulls from newer wheels."""
        try:
            cmd = [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--force-reinstall",
                OPENCV_PINNED,
                "--target",
                os.fspath(deps_path),
            ]
            print(f'Ensuring pinned OpenCV: {cmd}')
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError as e:
            print(f'Warning: failed to enforce pinned OpenCV ({OPENCV_PINNED}): {e}')

    @staticmethod
    def _ensure_pypose_installed():
        """Best-effort install of pypose into deps_public; ignore failure on unsupported platforms."""
        try:
            cmd = [
                sys.executable,
                "-m",
                "pip",
                "install",
                "pypose",
                "--target",
                os.fspath(deps_path),
            ]
            print(f'Ensuring pypose (best-effort): {cmd}')
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError as e:
            print(f'Warning: pypose install failed (may be unsupported on this platform): {e}')

    @staticmethod
    def _ensure_sklearn_installed():
        """Best-effort install of scikit-learn into deps_public for loop closure utilities."""
        try:
            cmd = [
                sys.executable,
                "-m",
                "pip",
                "install",
                "scikit-learn",
                "--target",
                os.fspath(deps_path),
            ]
            print(f'Ensuring scikit-learn (best-effort): {cmd}')
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError as e:
            print(f'Warning: scikit-learn install failed: {e}')

    @staticmethod
    def _update_submodules():
        """Update all submodules including nested .gitmodules under da3_streaming."""
        try:
            # Root-level submodules
            subprocess.check_call([
                'git', '-C', os.fspath(DA3_DIR), 'submodule', 'update', '--init', '--recursive'
            ])
            # Handle nested .gitmodules under da3_streaming if present
            streaming_dir = DA3_DIR / 'da3_streaming'
            gitmodules_path = streaming_dir / '.gitmodules'
            if gitmodules_path.exists():
                subprocess.check_call([
                    'git', '-C', os.fspath(streaming_dir), 'submodule', 'update', '--init', '--recursive'
                ])
                # Fallback: ensure salad submodule exists even if nested update didn’t populate
                salad_path = streaming_dir / 'loop_utils' / 'salad'
                if not salad_path.exists():
                    subprocess.check_call([
                        'git', 'clone', 'https://github.com/serizba/salad.git', os.fspath(salad_path)
                    ])
            return True
        except subprocess.CalledProcessError as e:
            print('Caught Exception while trying to update submodules (including da3_streaming)')
            print(f'  Exception: {e}')
            return False

    @staticmethod
    def check(*, force=False):
        if force:
            Dependencies._checked = None
        elif Dependencies._checked is not None:
            # Assume everything is installed
            return Dependencies._checked

        Dependencies._checked = False

        if deps_path.exists() and os.path.exists(DA3_DIR):
            try:
                # Ensure all required dependencies are installed in dependencies folder
                ws = pkg_resources.WorkingSet(entries=[ os.fspath(deps_path) ])
                for dep in Dependencies.requirements(force=force):
                    ws.require(dep)

                # If we get here, we found all required dependencies
                Dependencies._checked = True

            except Exception as e:
                print(f'Caught Exception while trying to check dependencies')
                print(f'  Exception: {e}')
                Dependencies._checked = False

        return Dependencies._checked

    @staticmethod
    def requirements(*, force=False):
        if force:
            Dependencies._requirements = None
        elif Dependencies._requirements is not None:
            return Dependencies._requirements

        # load and cache requirements
        with requirements_for_check_txt.open() as requirements:
            dependencies = pkg_resources.parse_requirements(requirements)
            Dependencies._requirements = [ dep.project_name for dep in dependencies ]
        return Dependencies._requirements
