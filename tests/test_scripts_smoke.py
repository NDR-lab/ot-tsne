"""The small conference-facing command surface must remain importable."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import tomllib

import pytest

from ne_aligned_utils.config import ROOT

PUBLIC_SCRIPTS = (
    ROOT / "scripts" / "prepare_data.py",
    ROOT / "scripts" / "run_benchmark.py",
    ROOT / "scripts" / "make_figures.py",
    ROOT / "scripts" / "check_environment.py",
)

SETUP_SCRIPT = ROOT / "scripts" / "setup_external_envs.sh"
SUBMIT_SCRIPT = ROOT / "scripts" / "submit_benchmark.sh"
SHELL_SCRIPTS = (SETUP_SCRIPT, SUBMIT_SCRIPT)


@pytest.mark.parametrize("script", PUBLIC_SCRIPTS, ids=lambda path: path.name)
def test_public_script_builds_argument_parser(script):
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout.lower()


@pytest.mark.parametrize("script", SHELL_SCRIPTS, ids=lambda path: path.name)
def test_shell_script_syntax(script):
    completed = subprocess.run(
        ["bash", "-n", str(script)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr


def test_external_setup_help_does_not_install_anything():
    completed = subprocess.run(
        ["bash", str(SETUP_SCRIPT), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Creates the pinned R, LIGER, MultiMAP, and rc-tSNE environments" in completed.stdout


def test_submission_script_only_makes_final_job_dependent(tmp_path):
    capture = tmp_path / "submissions.txt"
    fake_sbatch = tmp_path / "sbatch"
    fake_sbatch.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$*" >> "$SBATCH_CAPTURE"\n'
        'if [[ " $* " == *" --parsable "* ]]; then printf "123\\n"; fi\n'
    )
    fake_sbatch.chmod(0o755)
    fake_uv = tmp_path / "uv"
    fake_uv.write_text('#!/usr/bin/env bash\nshift 2\nexec "$TEST_PYTHON" "$@"\n')
    fake_uv.chmod(0o755)
    env = dict(os.environ)
    env["PATH"] = f"{tmp_path}{os.pathsep}{env['PATH']}"
    env["SBATCH_CAPTURE"] = str(capture)
    env["NE_ALIGNED_LAUNCHER_FILE"] = str(tmp_path / "missing-launchers.sh")
    env["TEST_PYTHON"] = sys.executable

    completed = subprocess.run(
        ["bash", str(SUBMIT_SCRIPT)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
    submissions = capture.read_text().splitlines()
    # Jobs come from the manifest: 6 enabled non-Yao datasets x 3 enabled experiments,
    # 23 Yao method jobs, and the finalizer.
    assert len(submissions) == 42
    run_jobs = submissions[:-1]
    assert all("run-aggregate" in command for command in run_jobs)
    assert all("--dependency" not in command for command in run_jobs)
    yao = [command for command in run_jobs if "--dataset yao_10x_ent_l2l3" in command]
    assert len(yao) == 23
    assert all("--method" in command for command in yao)
    with (ROOT / "configs/benchmark.toml").open("rb") as handle:
        manifest = tomllib.load(handle)
    expected_yao = {
        (experiment["name"], method)
        for experiment in manifest["experiments"]
        if experiment.get("enabled", True) and "yao_10x_ent_l2l3" in experiment["datasets"]
        for method in experiment["methods"]
    }
    actual_yao = []
    for command in yao:
        tokens = shlex.split(command)
        actual_yao.append(
            (tokens[tokens.index("--experiment") + 1], tokens[tokens.index("--method") + 1])
        )
    assert set(actual_yao) == expected_yao
    assert len(actual_yao) == len(expected_yao)
    assert "--dependency=afterok:" in submissions[-1]
    assert "run_benchmark.py finalize" in submissions[-1]


def test_gpu_submission_runs_only_ot_plan_methods_on_cuda(tmp_path):
    capture = tmp_path / "submissions.txt"
    fake_sbatch = tmp_path / "sbatch"
    fake_sbatch.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$*" >> "$SBATCH_CAPTURE"\n'
        'if [[ " $* " == *" --parsable "* ]]; then printf "123\\n"; fi\n'
    )
    fake_sbatch.chmod(0o755)
    fake_uv = tmp_path / "uv"
    fake_uv.write_text('#!/usr/bin/env bash\nshift 2\nexec "$TEST_PYTHON" "$@"\n')
    fake_uv.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        "SBATCH_CAPTURE": str(capture),
        "NE_ALIGNED_LAUNCHER_FILE": str(tmp_path / "missing-launchers.sh"),
        "TEST_PYTHON": sys.executable,
    }
    completed = subprocess.run(
        ["bash", str(SUBMIT_SCRIPT), "--device", "cuda"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    submissions = capture.read_text().splitlines()
    assert all("--device cuda" in command for command in submissions)
    yao_methods = {
        shlex.split(command)[shlex.split(command).index("--method") + 1]
        for command in submissions
        if "--dataset yao_10x_ent_l2l3" in command
    }
    assert yao_methods == {
        "eot-50", "eot-t0", "ot", "ot-gramdiag", "ot-knndiag", "ot-selfotmedian", "ot-zerodiag"
    }


def test_gpu_submission_can_be_limited_to_default_ot_tsne_on_yao(tmp_path):
    capture = tmp_path / "submissions.txt"
    fake_sbatch = tmp_path / "sbatch"
    fake_sbatch.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$*" >> "$SBATCH_CAPTURE"\n'
        'if [[ " $* " == *" --parsable "* ]]; then printf "123\\n"; fi\n'
    )
    fake_sbatch.chmod(0o755)
    fake_uv = tmp_path / "uv"
    fake_uv.write_text('#!/usr/bin/env bash\nshift 2\nexec "$TEST_PYTHON" "$@"\n')
    fake_uv.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        "SBATCH_CAPTURE": str(capture),
        "NE_ALIGNED_LAUNCHER_FILE": str(tmp_path / "missing-launchers.sh"),
        "TEST_PYTHON": sys.executable,
    }
    completed = subprocess.run(
        [
            "bash",
            str(SUBMIT_SCRIPT),
            *("--device", "cuda", "--dataset", "yao_10x_ent_l2l3", "--method", "ot"),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    submissions = capture.read_text().splitlines()
    # One default OT-tSNE job on Yao, then one finalizer for that selection only.
    assert len(submissions) == 2
    assert "--experiment ot_diagonal_cpu --method ot --device cuda" in submissions[0]
    assert "--dataset yao_10x_ent_l2l3" in submissions[0]
    assert "finalize --dataset yao_10x_ent_l2l3 --method ot --device cuda" in submissions[-1]

    unknown = subprocess.run(
        ["bash", str(SUBMIT_SCRIPT), "--dataset", "no-such-dataset"],
        cwd=tmp_path,
        env={**env, "SBATCH_CAPTURE": str(tmp_path / "unused.txt")},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert unknown.returncode != 0
    assert not (tmp_path / "unused.txt").exists()


@pytest.mark.parametrize("manifest", ["invalid TOML", "experiments = []"])
def test_invalid_yao_configuration_stops_before_submission(tmp_path, manifest):
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "configs").mkdir()
    shutil.copy2(SUBMIT_SCRIPT, repo / "scripts" / SUBMIT_SCRIPT.name)
    (repo / "configs" / "benchmark.toml").write_text(manifest)
    fake_uv = tmp_path / "uv"
    fake_uv.write_text('#!/usr/bin/env bash\nshift 2\nexec "$TEST_PYTHON" "$@"\n')
    fake_uv.chmod(0o755)
    fake_sbatch = tmp_path / "sbatch"
    fake_sbatch.write_text('#!/usr/bin/env bash\ntouch "$SBATCH_CAPTURE"\n')
    fake_sbatch.chmod(0o755)
    capture = tmp_path / "submissions.txt"
    env = {
        **os.environ,
        "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        "TEST_PYTHON": sys.executable,
        "SBATCH_CAPTURE": str(capture),
        "NE_ALIGNED_LAUNCHER_FILE": str(tmp_path / "missing-launchers.sh"),
    }

    completed = subprocess.run(
        ["bash", str(repo / "scripts" / SUBMIT_SCRIPT.name)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode != 0
    assert not capture.exists()
