"""End-to-end test for C++ attach-based LSP setup and indexing."""

from __future__ import annotations

import contextlib
import json
import socket
import threading

from astrograph.tools import CodeStructureTools
from tests.languages.test_lsp_client import _run_fake_socket_lsp


class _EphemeralSocketLSP:
    """Minimal socket LSP endpoint used when no host C++ endpoint is running."""

    def __init__(self, host: str = "127.0.0.1") -> None:
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.bind((host, 0))
        self._server.listen(2)
        self._server.settimeout(0.2)
        self._endpoint = f"tcp://{host}:{self._server.getsockname()[1]}"

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve_loop, daemon=True)

    @property
    def endpoint(self) -> str:
        return self._endpoint

    def start(self) -> None:
        self._thread.start()

    def _serve_loop(self) -> None:
        while not self._stop.is_set():
            try:
                _run_fake_socket_lsp(self._server)
            except TimeoutError:
                continue
            except OSError:
                break

    def close(self) -> None:
        self._stop.set()
        with contextlib.suppress(OSError):
            self._server.close()
        self._thread.join(timeout=1.0)


class TestCppAttachE2E:
    def test_cpp_attach_discovery_and_indexing(self, tmp_path, monkeypatch):
        """Discover/provision a C++ endpoint, bind it, then index C++ code."""
        fallback_endpoint = _EphemeralSocketLSP()
        fallback_endpoint.start()
        endpoint = fallback_endpoint.endpoint

        source_file = tmp_path / "sample.cpp"
        source_file.write_text(
            """
class Greeter {
public:
  int greet(int value) {
    return value + 1;
  }
};

// function helper
int helper(int value) {
  return value * 2;
}
""".strip()
            + "\n"
        )
        build_dir = tmp_path / "build"
        build_dir.mkdir(parents=True, exist_ok=True)
        (build_dir / "compile_commands.json").write_text(
            json.dumps(
                [
                    {
                        "directory": str(tmp_path),
                        "command": "clang++ -std=c++20 -I. -c sample.cpp",
                        "file": "sample.cpp",
                    }
                ]
            )
            + "\n"
        )

        # Disable startup auto-index to avoid races; then point lsp_setup workspace explicitly.
        monkeypatch.setenv("ASTROGRAPH_WORKSPACE", "")
        tools = CodeStructureTools()
        monkeypatch.setenv("ASTROGRAPH_WORKSPACE", str(tmp_path))

        try:
            inspect_before = json.loads(tools.lsp_setup(mode="inspect").text)
            assert "cpp_lsp" in inspect_before["supported_languages"]

            auto_bind = json.loads(
                tools.lsp_setup(
                    mode="auto_bind",
                    observations=[
                        {
                            "language": "cpp_lsp",
                            "command": endpoint,
                        }
                    ],
                ).text
            )
            assert any(change["language"] == "cpp_lsp" for change in auto_bind["changes"])

            inspect_after = json.loads(tools.lsp_setup(mode="inspect").text)
            cpp_status = next(s for s in inspect_after["servers"] if s["language"] == "cpp_lsp")
            assert cpp_status["available"] is True
            assert cpp_status["transport"] == "tcp"

            index_result = tools.index_codebase(str(tmp_path))
            assert "Indexed" in index_result.text
            assert len(tools.index.entries) > 0
            assert any(
                entry.code_unit.language == "cpp_lsp"
                and entry.code_unit.file_path.endswith("sample.cpp")
                for entry in tools.index.entries.values()
            )
        finally:
            tools.close()
            fallback_endpoint.close()
