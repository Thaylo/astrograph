"""Global pytest fixtures for deterministic test behavior."""

import json
import sys

import pytest


@pytest.fixture(autouse=True, scope="session")
def _session_test_workspace(tmp_path_factory):
    """Create a session workspace with a Python LSP binding.

    The binding-only architecture (D-009) requires an explicit binding for
    each language before ``create_lsp_client`` returns a real client.
    Without this fixture, every test that indexes Python code would get a
    NullLSPClient and zero extracted code units.
    """
    workspace = tmp_path_factory.mktemp("astrograph_test")
    meta_dir = workspace / ".metadata_astrograph"
    meta_dir.mkdir()
    bindings = {"python": [sys.executable, "-m", "pylsp"]}
    (meta_dir / "lsp_bindings.json").write_text(json.dumps(bindings))

    yield workspace

    from astrograph.languages.registry import LanguageRegistry

    LanguageRegistry.reset()


@pytest.fixture(autouse=True)
def _disable_startup_autoindex(monkeypatch, _session_test_workspace):
    """Point ASTROGRAPH_WORKSPACE at the session workspace.

    The workspace directory is empty (except for the binding file), so
    startup auto-indexing will find nothing and finish instantly.
    """
    monkeypatch.setenv("ASTROGRAPH_WORKSPACE", str(_session_test_workspace))
