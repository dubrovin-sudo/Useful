#!/usr/bin/env python

import subprocess

from .. import export, reconstruct


def git_checkout(commit_hash):
    try:
        # Checkout the specific commit hash for reconstruct.py and export.py
        subprocess.run(
            ["git", "checkout", commit_hash, "--", "reconstruct.py"], check=True
        )
        subprocess.run(["git", "checkout", commit_hash, "--", "export.py"], check=True)
        print(f"Checked out reconstruct.py and export.py to commit {commit_hash}")
    except subprocess.CalledProcessError as e:
        print(f"Error during git checkout: {e}")


def main():
    print("1st")
    commit_hash = "53e1539fbdca4d14680bdf919ac46a40acf3a7f7"  # 1st
    git_checkout(commit_hash)  # Checkout the commit
    reconstruct()  # Execute reconstruct code
    export()  # Execute export code


if __name__ == "__main__":
    main()
