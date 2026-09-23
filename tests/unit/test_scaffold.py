# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests that the package itself is well formed.

These are not tests of the method. They are tests that the scaffold holds
together: every module imports, the version is consistent, the layering rules
are actually respected, and the license header is present everywhere. They are
cheap and they catch the class of mistake that is otherwise noticed late.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import pkgutil
import platform
import sys
from pathlib import Path

import pytest

import mround
from mround.cli.main import build_parser, main

PACKAGE_ROOT = Path(mround.__file__).parent
REPO_ROOT = PACKAGE_ROOT.parent


def _module_names() -> list[str]:
    """Every importable module in the package."""
    return [
        name for _, name, _ in pkgutil.walk_packages(mround.__path__, prefix=f"{mround.__name__}.")
    ]


# Every directory whose Python this repository authors, as glob patterns so the
# Linux machine's folder is found without its name appearing here (the public
# export forbids that name in the tree it builds). The vendored reference trees
# and the generated public tree are not listed: the first is not ours to edit
# and the second is a build output of files already checked here.
AUTHORED_DIRECTORY_PATTERNS = (
    "mround",
    "tests",
    "scripts",
    "examples",
    "paper",
    "analysis",
    "ubuntu-*",
)

# The directories that exist in every checkout, private or published.
ALWAYS_AUTHORED = ("mround", "tests", "scripts", "examples")

HEADER = ("# Copyright 2026 The MRound Authors", "# SPDX-License-Identifier: Apache-2.0")


def _authored_directories() -> list[Path]:
    """The authored directories present in this checkout."""
    return sorted(
        {
            path
            for pattern in AUTHORED_DIRECTORY_PATTERNS
            for path in REPO_ROOT.glob(pattern)
            if path.is_dir()
        }
    )


def _source_files() -> list[Path]:
    """Every Python source file this repository authors."""
    return sorted(
        path
        for directory in _authored_directories()
        for path in directory.rglob("*.py")
        if "__pycache__" not in path.parts
    )


# The layers that must run on a machine with no MLX at all. Directories and
# single modules, both spelled relative to the package root.
FRAMEWORK_FREE_LAYERS = ("reference", "planner", "schemes.py", "exceptions.py")

FRAMEWORK_FREE_FILES = sorted(
    {
        path
        for entry in FRAMEWORK_FREE_LAYERS
        for path in (
            (PACKAGE_ROOT / entry).rglob("*.py")
            if (PACKAGE_ROOT / entry).is_dir()
            else [PACKAGE_ROOT / entry]
        )
    }
)

# A floor, not a count, so adding a module to a covered layer does not fail the
# suite. It exists to catch a layer disappearing from coverage entirely.
EXPECTED_FRAMEWORK_FREE_FILES = 8


def _declares_unfinished(path: Path) -> bool:
    """Whether this module's docstring admits it is not finished.

    The convention is a ``Status:`` line saying what is implemented and what is
    not. Three states occur and all three matter: finished modules carry no
    line, pure stubs say ``Not implemented``, and partly implemented modules say
    which half is which. An earlier version of this check looked for the literal
    ``Status: Phase``, which fitted only the first two and started failing the
    moment a module became half-finished, which is a normal thing to be.

    The marker is load-bearing rather than decorative: it grants the exemption
    below, so a separate test checks it against which modules actually raise.
    """
    docstring = ast.get_docstring(ast.parse(path.read_text())) or ""
    status = [line for line in docstring.splitlines() if line.startswith("Status:")]
    if not status:
        return False
    # A marker that does not say so is not an admission. The earlier version
    # granted the exemption for any ``Status:`` line at all, so a module saying
    # ``Status: fully implemented`` while still raising would have passed.
    assert all("not implemented" in line.lower() for line in status), (
        f"{path.name}: a Status line must say what is not implemented: {status}"
    )
    return True


def _raises_not_implemented(path: Path) -> bool:
    """Whether any function in this module raises ``NotImplementedError``."""
    return any(
        isinstance(node, ast.Raise)
        and node.exc is not None
        and "NotImplementedError" in ast.dump(node.exc)
        for node in ast.walk(ast.parse(path.read_text()))
    )


def _imports_package(path: Path, package: str) -> bool:
    """Whether this module imports ``package`` anywhere, parsed rather than grepped."""
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import) and any(
            alias.name == package or alias.name.startswith(package + ".") for alias in node.names
        ):
            return True
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == package or module.startswith(package + "."):
                return True
    return False


def _is_matrix_product(node: ast.expr) -> bool:
    """Whether an expression is ``a @ b`` or a call to a ``matmul``."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
        return True
    if isinstance(node, ast.Call):
        function = node.func
        name = function.attr if isinstance(function, ast.Attribute) else getattr(function, "id", "")
        return name == "matmul"
    return False


def _imports_mlx_at_runtime(path: Path) -> bool:
    """Whether this module imports MLX outside an ``if TYPE_CHECKING`` block.

    Parsed rather than grepped. The previous version of this check tested
    whether the file contained the string ``TYPE_CHECKING`` anywhere, which
    meant a single type-only import elsewhere in the file excused a genuine
    runtime dependency. That is exactly the kind of check that passes forever
    and means nothing.
    """
    tree = ast.parse(path.read_text())
    guarded: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.dump(node.test):
            guarded.update(id(child) for child in ast.walk(node) if child is not node)
    return any(
        id(node) not in guarded
        and (
            (isinstance(node, ast.Import) and any(a.name.startswith("mlx") for a in node.names))
            or (isinstance(node, ast.ImportFrom) and (node.module or "").startswith("mlx"))
        )
        for node in ast.walk(tree)
    )


class TestPackage:
    def test_version_is_a_string(self) -> None:
        assert isinstance(mround.__version__, str)
        assert mround.__version__

    def test_every_module_imports(self) -> None:
        # A stub that raises NotImplementedError when called is fine. A stub
        # that cannot even be imported means the skeleton is broken.
        #
        # One exception is legitimate: `mround.core` is the MLX layer and imports
        # MLX at module level, which is correct, so on a machine without MLX
        # those modules cannot import. That specific failure is tolerated and
        # every other one is not, which keeps the check meaningful here while
        # remaining complete on Apple Silicon.
        broken: list[str] = []
        for name in _module_names():
            try:
                importlib.import_module(name)
            except ModuleNotFoundError as exc:
                if exc.name is not None and exc.name.split(".")[0] == "mlx":
                    continue
                broken.append(f"{name}: {exc!r}")
            except Exception as exc:
                broken.append(f"{name}: {exc!r}")
        assert not broken, f"modules failed to import: {broken}"

    def test_the_framework_free_layers_do_not_import_mlx(self) -> None:
        # The rule is not "MLX only in core/". The formats layer writes MLX
        # checkpoints and the pipeline layer runs MLX models, so both need it,
        # and saying otherwise would make this test something people route
        # around rather than something they respect.
        #
        # What must hold is that the deliberately framework-free layers stay
        # that way: the reference implementation, because it is the oracle the
        # MLX code is checked against and it has to run where MLX does not; the
        # planner, because it does wide accumulation on the CPU and knows
        # nothing about arrays; and the two root modules, because every layer
        # imports them and a dependency there would be a dependency everywhere.
        # See CLAUDE.md, "Match the layering".
        offenders = [path for path in sorted(FRAMEWORK_FREE_FILES) if _imports_mlx_at_runtime(path)]
        assert not offenders, (
            "these layers must run without MLX and no longer do: "
            f"{[p.relative_to(PACKAGE_ROOT).as_posix() for p in offenders]}"
        )

    def test_the_framework_free_set_is_not_silently_shrinking(self) -> None:
        # Without this, deleting a directory from FRAMEWORK_FREE_LAYERS would
        # make the test above pass by covering nothing, which is the failure
        # mode of every allowlist.
        covered = {path.relative_to(PACKAGE_ROOT).parts[0] for path in FRAMEWORK_FREE_FILES}
        assert {"reference", "planner", "schemes.py", "exceptions.py"} <= covered
        assert len(FRAMEWORK_FREE_FILES) >= EXPECTED_FRAMEWORK_FREE_FILES

    def test_the_mlx_layers_really_do_import_it(self) -> None:
        # The mirror of the rule above. `core/` exists to be the MLX
        # implementation, so an implemented module there that does not import
        # MLX is either dead or has drifted into the reference's job. Modules
        # that declare themselves unfinished are exempt, since a stub has
        # nothing to implement it with yet.
        silent = [
            path
            for path in sorted((PACKAGE_ROOT / "core").rglob("*.py"))
            if path.name != "__init__.py"
            and not _declares_unfinished(path)
            and not _imports_mlx_at_runtime(path)
        ]
        assert not silent, (
            f"implemented core/ modules that do not import MLX: "
            f"{[p.relative_to(PACKAGE_ROOT).as_posix() for p in silent]}"
        )

    def test_the_status_marker_agrees_with_the_code(self) -> None:
        # The `Status:` line is used by the test above to grant an exemption,
        # which makes it load-bearing. A marker that drifts out of step with the
        # code hands out exemptions to modules no longer entitled to them, and
        # tells a reader a module is finished when it still raises. So the two
        # are checked against each other rather than one being trusted.
        declared = {
            path.relative_to(PACKAGE_ROOT).as_posix()
            for path in PACKAGE_ROOT.rglob("*.py")
            if _declares_unfinished(path)
        }
        raising = {
            path.relative_to(PACKAGE_ROOT).as_posix()
            for path in PACKAGE_ROOT.rglob("*.py")
            if _raises_not_implemented(path)
        }
        assert declared == raising, (
            f"declares itself unfinished but implements everything: "
            f"{sorted(declared - raising)}; "
            f"still raises NotImplementedError but claims to be done: "
            f"{sorted(raising - declared)}"
        )

    def test_package_is_typed(self) -> None:
        assert (PACKAGE_ROOT / "py.typed").is_file()


class TestLicenseHeaders:
    def test_every_source_file_carries_the_header(self) -> None:
        # Both lines, in order, as the first two lines of the file: CLAUDE.md
        # says "exactly this header, before the module docstring", and a check
        # for the SPDX substring anywhere in the first 400 characters would
        # have accepted a header buried under a shebang or inside a docstring.
        missing = [
            path.relative_to(REPO_ROOT)
            for path in _source_files()
            if tuple(path.read_text().splitlines()[:2]) != HEADER
        ]
        assert not missing, f"missing or misplaced license header: {missing}"

    def test_the_header_check_covers_every_authored_directory(self) -> None:
        # The directories every checkout has are found, so a directory renamed
        # or moved does not quietly leave the check, and the walk reaches
        # beyond the package itself.
        found = {path.name for path in _authored_directories()}
        assert set(ALWAYS_AUTHORED) <= found, found
        assert len(_source_files()) > len(list(PACKAGE_ROOT.rglob("*.py")))


class TestLayering:
    """The layering rules from DOCUMENTATION.md section 3, enforced.

    CLAUDE.md makes layer violations a rejection criterion, so they are checked
    rather than trusted.
    """

    def test_planner_does_not_import_mlx_at_runtime(self) -> None:
        # The planner consumes scores and costs and produces an allocation.
        # Keeping MLX out of its runtime path is what makes it testable
        # without a model. Type-only imports under TYPE_CHECKING are fine.
        offenders = [
            path.relative_to(REPO_ROOT)
            for path in (PACKAGE_ROOT / "planner").rglob("*.py")
            if _imports_mlx_at_runtime(path)
        ]
        assert not offenders, f"planner imports MLX at runtime: {offenders}"

    def test_allocator_has_no_mlx_reference_at_all(self) -> None:
        # The allocator is pure discrete optimization over numbers. Even a
        # type-only MLX reference would signal the layering has slipped.
        source = (PACKAGE_ROOT / "planner" / "allocator.py").read_text()
        assert "mlx" not in source

    def test_schemes_imports_only_the_error_types_from_the_package(self) -> None:
        # Every layer depends on schemes, so schemes may depend on none of them.
        # The one exception is the error module, itself a leaf that imports
        # nothing, so that a malformed scheme is refused with the package's
        # own SchemeError (also a ValueError) rather than a bare ValueError.
        source = (PACKAGE_ROOT / "schemes.py").read_text()
        imports = [
            line.strip()
            for line in source.splitlines()
            if line.startswith(("from mround", "import mround"))
        ]
        assert imports == ["from mround.exceptions import SchemeError"]
        errors = (PACKAGE_ROOT / "exceptions.py").read_text()
        assert "from mround" not in errors
        assert "import mround" not in errors

    def test_reference_layer_never_imports_mlx(self) -> None:
        # The reference exists to be an oracle the MLX implementation can be
        # checked against, and an oracle that shares a framework with the thing
        # it judges is worth much less. It also has to run on machines with no
        # MLX at all, which is what keeps the numerical core in CI.
        offenders = [
            path.relative_to(REPO_ROOT)
            for path in (PACKAGE_ROOT / "reference").rglob("*.py")
            if "mlx" in path.read_text()
        ]
        assert not offenders, f"reference layer references MLX: {offenders}"

    def test_no_runtime_torch_dependency(self) -> None:
        # PyTorch is a parity-fixture tool only. See MEMORY.md D-002.
        offenders = [
            path.relative_to(REPO_ROOT)
            for path in PACKAGE_ROOT.rglob("*.py")
            if _imports_package(path, "torch") or _imports_package(path, "auto_round")
        ]
        assert not offenders, f"package imports torch or auto_round: {offenders}"

    def test_no_residual_is_formed_by_subtracting_two_products(self) -> None:
        # MEMORY.md D-014: ``mx.matmul`` on float32 loses four orders of
        # magnitude on Metal, so a small residual is formed by multiplying the
        # difference of the operands, never by subtracting two large products.
        # A numerical test cannot pin this, because on the CPU backend both
        # forms agree to float32 and on Metal the difference sits under the
        # parity tolerance that D-014 itself set. So the shape of the
        # expression is what gets checked: no subtraction anywhere in the
        # numerical layers may have a matrix product as either operand.
        offenders = []
        for directory in ("core", "reference"):
            for path in (PACKAGE_ROOT / directory).rglob("*.py"):
                for node in ast.walk(ast.parse(path.read_text())):
                    if (
                        isinstance(node, ast.BinOp)
                        and isinstance(node.op, ast.Sub)
                        and any(_is_matrix_product(side) for side in (node.left, node.right))
                    ):
                        offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
        assert not offenders, f"a residual is formed from two products: {offenders}"


class TestEnvironment:
    """Checks that only mean something on the development machine.

    These skip everywhere else rather than failing, because the framework-free
    layers are meant to run on any host and continuous integration exercises
    them on Linux.
    """

    @pytest.mark.skipif(
        sys.platform != "darwin" or platform.machine() != "arm64",
        reason="Apple Silicon only",
    )
    def test_mlx_is_installed_where_it_is_the_gate(self) -> None:
        # On Apple Silicon a missing MLX must fail, not skip. Every MLX gated
        # test collapses to a skip when the framework is absent, so without
        # this check a Mac with a broken environment reports a green suite of
        # about two hundred skips, which is the one outcome the gate exists to
        # rule out. Linux without MLX is the framework-free path and is not
        # gated here.
        assert importlib.util.find_spec("mlx.core") is not None, (
            "MLX is not importable on this Mac: the MLX suite has silently "
            "become skips. Install it with pip install -e '.[dev]'."
        )

    @pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")
    def test_python_is_native_arm64(self) -> None:
        # MLX requires a native arm64 Python and cannot install under Rosetta.
        # The failure mode without this check is a confusing install error or a
        # cryptic crash much later, so it is worth catching at the point where
        # someone can still fix it cheaply by rebuilding the conda environment.
        processor = platform.processor()
        assert processor == "arm", (
            f"this Python reports processor={processor!r}, which means it is "
            "running under Rosetta rather than natively. MLX cannot install "
            "here. Recreate the environment with "
            "`CONDA_SUBDIR=osx-arm64 conda env create -f environment.yml`."
        )

    @pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")
    def test_macos_is_new_enough_for_mlx(self) -> None:
        # MLX requires macOS 14 or later; MRound targets 26 (Tahoe) or later.
        # Reported as a warning-shaped assertion because someone running only
        # the reference layer on an older machine is doing something reasonable.
        release = platform.mac_ver()[0]
        if not release:
            pytest.skip("could not determine the macOS version")
        major = int(release.split(".")[0])
        assert major >= 14, (
            f"macOS {release} is below the 14.0 that MLX requires; the "
            "framework-free reference layer still works, but nothing on Metal will"
        )


class TestCLI:
    def test_parser_builds(self) -> None:
        assert build_parser().prog == "mround"

    def test_bare_invocation_prints_help_and_fails(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main([]) == 1
        assert "usage: mround" in capsys.readouterr().out

    def test_version_flag(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as excinfo:
            main(["--version"])
        assert excinfo.value.code == 0
        assert mround.__version__ in capsys.readouterr().out

    @pytest.mark.parametrize("command", ["version", "doctor", "quantize", "eval"])
    def test_subcommands_are_registered(self, command: str) -> None:
        parser = build_parser()
        registered: set[str] = set()
        for action in parser._actions:
            choices = getattr(action, "choices", None)
            if isinstance(choices, dict):
                registered.update(choices)
        assert command in registered

    @pytest.mark.parametrize("command", ["quantize", "eval"])
    def test_the_working_subcommands_report_rather_than_traceback(
        self, command: str, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # These two used to raise NotImplementedError and this test checked
        # that they said where the plan lived. They are implemented now, and
        # what matters has moved: whatever stops them, a missing model or a
        # platform that cannot run MLX, has to arrive as a message and an exit
        # status rather than as a traceback, because every one of those errors
        # is something the person can act on.
        argv = [command, "definitely-not-a-model"] + (
            ["-o", "out"] if command == "quantize" else []
        )
        assert main(argv) == 1
        assert capsys.readouterr().err.startswith("mround: ")

    def test_the_zero_shot_suite_says_it_is_not_here_yet(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # The one thing `eval` advertises and cannot do. It names what it
        # cannot do, where the plan is, and what it can do instead, rather
        # than failing somewhere inside a stub.
        assert main(["eval", "a-checkpoint", "--zeroshot"]) == 1
        message = capsys.readouterr().err
        assert "zero-shot" in message
        assert "ROADMAP.md" in message
        assert "--perplexity" in message

    def test_version_subcommand_matches_the_flag(self, capsys: pytest.CaptureFixture[str]) -> None:
        # `mround --version` worked while `mround version` raised, which is a
        # papercut on the one command a person tries first.
        assert main(["version"]) == 0
        assert capsys.readouterr().out.strip() == f"mround {mround.__version__}"

    def test_doctor_reports_and_its_status_is_its_verdict(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Doctor exists so that a machine that cannot run MRound says so here
        # rather than an hour into a tuning run, and so that a script can gate
        # on the exit status without parsing anything.
        code = main(["doctor"])
        printed = capsys.readouterr().out
        assert "mround doctor" in printed
        assert "checks" in printed
        ready = "quantization      ready" in printed
        assert code == (0 if ready else 1)
        # The reference layer needs only numpy, which this suite imports, so on
        # any machine that runs this test the verdict is ready. The earlier
        # form of this assertion also accepted the word "numpy", which doctor
        # prints unconditionally, and so could not fail.
        assert "reference layer   ready" in printed

    def test_doctor_json_carries_the_same_verdict(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = main(["doctor", "--json"])
        report = json.loads(capsys.readouterr().out)
        assert set(report) == {
            "mround",
            "python",
            "system",
            "machine",
            "processor",
            "conda_env",
            "packages",
            "device",
            "checks",
            "ready",
        }
        assert {check["name"] for check in report["checks"]} == {
            "apple silicon",
            "mlx",
            "mlx-lm",
            "numpy",
            "datasets",
        }
        assert all(set(check) == {"name", "ok", "detail"} for check in report["checks"])
        assert set(report["ready"]) == {"quantize", "reference"}
        # Every failing check says what to do about it, and the exit status is
        # the quantize verdict rather than an unrelated success.
        assert all(check["detail"] for check in report["checks"])
        assert code == (0 if report["ready"]["quantize"] else 1)

    def test_doctor_agrees_with_the_loader_about_this_machine(self) -> None:
        # The verdict is the loader's own function, not a second copy of the
        # rule living in the CLI, so the two cannot drift apart.
        from mround.cli.main import _doctor_report  # noqa: PLC0415

        report = _doctor_report()
        verdict = next(c for c in report["checks"] if c["name"] == "apple silicon")
        assert verdict["ok"] == (sys.platform == "darwin" and platform.machine() == "arm64")
