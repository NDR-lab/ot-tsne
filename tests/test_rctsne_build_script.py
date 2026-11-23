"""Check build setup with local compiler and micromamba stubs."""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from ne_aligned_utils.config import ROOT


@pytest.fixture
def build_setup(tmp_path):
    repo = tmp_path / "repo"
    script = repo / "scripts" / "setup" / "build_rctsne.sh"
    script.parent.mkdir(parents=True)
    shutil.copy2(ROOT / "scripts" / "setup" / script.name, script)
    specification = repo / "envs" / "rctsne" / "environment.yml"
    specification.parent.mkdir(parents=True)
    shutil.copy2(ROOT / "envs" / "rctsne" / "environment.yml", specification)
    compiler = tmp_path / "compiler"
    compiler.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$1" == "--version" ]]; then echo "test compiler"; exit 0; fi\n'
        'printf "%s\\n" "$@" > "$COMPILER_CAPTURE"\n'
    )
    compiler.chmod(0o755)
    mamba = tmp_path / "micromamba"
    mamba.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$@" > "$MAMBA_CAPTURE"\n'
        'mkdir -p "$MAMBA_ROOT_PREFIX/envs/rctsne/include"\n'
        'touch "$MAMBA_ROOT_PREFIX/envs/rctsne/include/fftw3.h"\n'
    )
    mamba.chmod(0o755)
    env = {
        **os.environ,
        "CXX": str(compiler),
        "COMPILER_CAPTURE": str(tmp_path / "compiler.txt"),
        "NE_ALIGNED_MICROMAMBA": str(mamba),
        "MAMBA_CAPTURE": str(tmp_path / "micromamba.txt"),
        "MAMBA_ROOT_PREFIX": str(tmp_path / "mamba-root"),
    }
    env.pop("FFTW_PREFIX", None)
    return script, env


def run_build(script, env, tmp_path):
    return subprocess.run(
        ["bash", str(script)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_explicit_fftw_prefix_needs_no_micromamba(build_setup, tmp_path):
    script, env = build_setup
    prefix = tmp_path / "custom-fftw"
    (prefix / "include").mkdir(parents=True)
    (prefix / "include" / "fftw3.h").touch()
    env["FFTW_PREFIX"] = str(prefix)
    env["NE_ALIGNED_MICROMAMBA"] = str(tmp_path / "absent-micromamba")

    completed = run_build(script, env, tmp_path)

    assert completed.returncode == 0, completed.stderr
    arguments = (tmp_path / "compiler.txt").read_text().splitlines()
    assert f"-I{prefix}/include" in arguments
    assert f"-L{prefix}/lib" in arguments
    assert not (tmp_path / "micromamba.txt").exists()


def test_default_fftw_install_uses_pinned_specification(build_setup, tmp_path):
    script, env = build_setup

    completed = run_build(script, env, tmp_path)

    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "micromamba.txt").read_text().splitlines() == [
        "create",
        "-y",
        "-f",
        "envs/rctsne/environment.yml",
    ]
    arguments = (tmp_path / "compiler.txt").read_text().splitlines()
    assert f"-I{env['MAMBA_ROOT_PREFIX']}/envs/rctsne/include" in arguments


def test_invalid_explicit_prefix_fails_without_installing(build_setup, tmp_path):
    script, env = build_setup
    env["FFTW_PREFIX"] = str(tmp_path / "missing-fftw")

    completed = run_build(script, env, tmp_path)

    assert completed.returncode != 0
    assert "no fftw3.h" in completed.stderr
    assert not (tmp_path / "micromamba.txt").exists()
    assert not (tmp_path / "compiler.txt").exists()
