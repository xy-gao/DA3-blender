import os
import pkg_resources
import subprocess
import sys
from pathlib import Path


add_on_path = Path(__file__).parent                     # assuming this file is at root of add-on
os.environ["ADDON_PATH"] = str(add_on_path)
requirements_txt = add_on_path / 'requirements.txt'     # assuming requirements.txt is at root of add-on
requirements_for_check_txt = add_on_path / 'requirements_for_check.txt'     # assuming requirements.txt is at root of add-on
DA3_DIR = add_on_path / "da3_repo"

deps_path = add_on_path / 'deps_public'                 # might not exist until install_deps is called
# Append dependencies folder to system path so we can import
# (important for Windows machines, but less so for Linux)
sys.path.append(os.fspath(deps_path))
sys.path.append(os.fspath(DA3_DIR))

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
                subprocess.check_call(['git', 'clone', 'https://github.com/ByteDance-Seed/Depth-Anything-3.git', DA3_DIR])
            except subprocess.CalledProcessError as e:
                print(f'Caught Exception while trying to git clone da3')
                print(f'  Exception: {e}')
                return False
        
        try:
            deps_path.mkdir(exist_ok=True)
        except Exception as e:
            print(f'Caught Exception while trying to create dependencies folder')
            print(f'  Exception: {e}')
            print(f'  Folder: {deps_path}')
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
                "-e",
                os.fspath(DA3_DIR)
            ]
            print(f'Installing: {cmd}')
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError as e:
            print(f'Caught CalledProcessError while trying to install DA3')
            print(f'  Exception: {e}')
            print(f'  Requirements: {DA3_DIR}')
            return False
        return Dependencies.check(force=True)

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
