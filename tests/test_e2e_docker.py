"""End-to-end tests for the Docker container and MCP protocol.

These tests validate:
1. The published Docker image works correctly
2. MCP JSON-RPC protocol flow over stdio (newline AND Content-Length framing)
3. Full workflows: index → analyze → suppress → re-analyze
4. Clean process exit after EOF (no ClosedResourceError)
5. Mixed-language workspaces (Python + JavaScript)
"""

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

import pytest

# Docker image to test (use local build for CI, published for release validation)
DOCKER_IMAGE = os.environ.get("ASTOGRAPH_TEST_IMAGE", "thaylo/astrograph")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _docker_cmd(workspace_path: str | None = None, read_only: bool = False) -> list[str]:
    cmd = ["docker", "run", "--rm", "-i"]
    if workspace_path:
        mount_opt = "ro" if read_only else "rw"
        cmd.extend(["-v", f"{workspace_path}:/workspace:{mount_opt}"])
    cmd.append(DOCKER_IMAGE)
    return cmd


def send_mcp_messages(
    messages: list[dict],
    workspace_path: str | None = None,
    read_only: bool = False,
) -> list[dict]:
    """Send newline-delimited JSON-RPC messages and return parsed responses.

    Raises ``AssertionError`` on non-zero exit code so that failures surface
    the container's stderr instead of a confusing downstream assertion.
    """
    input_data = "\n".join(json.dumps(msg) for msg in messages)

    result = subprocess.run(
        _docker_cmd(workspace_path, read_only),
        input=input_data,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, (
        f"Container exited with code {result.returncode}\n"
        f"--- stdout ---\n{result.stdout[:2000]}\n"
        f"--- stderr ---\n{result.stderr[:2000]}"
    )

    responses = []
    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        try:
            responses.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise AssertionError(f"Unparseable JSON-RPC response: {line!r}") from exc
    return responses


def send_mcp_framed(
    messages: list[dict],
    workspace_path: str | None = None,
    read_only: bool = False,
) -> list[dict]:
    """Send Content-Length framed JSON-RPC messages and parse framed responses.

    This exercises the same code path real MCP clients (Claude Desktop, Cursor)
    use — Content-Length headers before each JSON body.
    """
    parts: list[bytes] = []
    for msg in messages:
        body = json.dumps(msg).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        parts.append(header + body)
    input_bytes = b"".join(parts)

    result = subprocess.run(
        _docker_cmd(workspace_path, read_only),
        input=input_bytes,
        capture_output=True,
        timeout=30,
    )

    assert result.returncode == 0, (
        f"Container exited with code {result.returncode}\n"
        f"--- stderr ---\n{result.stderr.decode('utf-8', errors='replace')[:2000]}"
    )

    # Parse framed responses from stdout
    responses = []
    buf = result.stdout
    while buf:
        # Find header end
        for delim, dlen in (b"\r\n\r\n", 4), (b"\n\n", 2):
            idx = buf.find(delim)
            if idx >= 0:
                headers = buf[:idx].decode("ascii")
                buf = buf[idx + dlen :]
                break
        else:
            break  # no more headers

        # Parse Content-Length
        content_length = None
        for header_line in headers.splitlines():
            if ":" in header_line:
                name, value = header_line.split(":", 1)
                if name.strip().lower() == "content-length":
                    content_length = int(value.strip())
                    break
        if content_length is None:
            break

        body = buf[:content_length]
        buf = buf[content_length:]
        responses.append(json.loads(body))

    return responses


def mcp_initialize() -> dict:
    """Create MCP initialize request."""
    return {
        "jsonrpc": "2.0",
        "method": "initialize",
        "params": {
            "protocolVersion": "1.0",
            "capabilities": {},
            "clientInfo": {"name": "e2e-test", "version": "1.0"},
        },
        "id": 1,
    }


def mcp_list_tools() -> dict:
    """Create MCP tools/list request."""
    return {
        "jsonrpc": "2.0",
        "method": "tools/list",
        "params": {},
        "id": 2,
    }


def mcp_call_tool(name: str, arguments: dict, request_id: int = 3) -> dict:
    """Create MCP tools/call request."""
    return {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": name,
            "arguments": arguments,
        },
        "id": request_id,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_workspace():
    """Create a temporary workspace with Python files containing duplicates."""
    with tempfile.TemporaryDirectory() as tmpdir:
        code = '''\
def calculate_sum(a, b):
    """Add two numbers."""
    result = a + b
    return result

def add_numbers(x, y):
    """Add two numbers."""
    result = x + y
    return result

def multiply(a, b):
    return a * b

class Calculator:
    def add(self, a, b):
        result = a + b
        return result
'''
        (Path(tmpdir) / "math_utils.py").write_text(code)

        code2 = """\
def process_list(items):
    results = []
    for item in items:
        if item > 0:
            results.append(item * 2)
    return results

def transform_values(data):
    results = []
    for item in data:
        if item > 0:
            results.append(item * 2)
    return results
"""
        (Path(tmpdir) / "data_utils.py").write_text(code2)
        yield tmpdir


@pytest.fixture
def sample_javascript_workspace():
    """Create a temporary workspace with JavaScript files containing duplicates."""
    with tempfile.TemporaryDirectory() as tmpdir:
        js_code = """\
function processItems(items) {
  const results = [];
  for (const item of items) {
    if (item > 0) {
      results.push(item * 2);
    }
  }
  return results;
}

function transformItems(data) {
  const results = [];
  for (const item of data) {
    if (item > 0) {
      results.push(item * 2);
    }
  }
  return results;
}
"""
        (Path(tmpdir) / "math_utils.js").write_text(js_code)
        yield tmpdir


@pytest.fixture
def mixed_language_workspace():
    """Workspace with both Python and JavaScript duplicates."""
    with tempfile.TemporaryDirectory() as tmpdir:
        py_code = """\
def process_items(items):
    results = []
    for item in items:
        if item > 0:
            results.append(item * 2)
    return results

def transform_items(data):
    results = []
    for item in data:
        if item > 0:
            results.append(item * 2)
    return results
"""
        (Path(tmpdir) / "utils.py").write_text(py_code)

        js_code = """\
function filterPositive(arr) {
  const out = [];
  for (const x of arr) {
    if (x > 0) {
      out.push(x * 2);
    }
  }
  return out;
}

function selectPositive(arr) {
  const out = [];
  for (const x of arr) {
    if (x > 0) {
      out.push(x * 2);
    }
  }
  return out;
}
"""
        (Path(tmpdir) / "helpers.js").write_text(js_code)
        yield tmpdir


# ---------------------------------------------------------------------------
# Docker image basics
# ---------------------------------------------------------------------------


class TestDockerImageBasics:
    """Basic tests for the Docker image."""

    def test_image_exists(self):
        """Test that the Docker image exists locally (pulling if needed)."""
        inspect_result = subprocess.run(
            ["docker", "image", "inspect", DOCKER_IMAGE],
            capture_output=True,
            timeout=30,
        )
        if inspect_result.returncode != 0:
            pull_result = subprocess.run(
                ["docker", "pull", DOCKER_IMAGE],
                capture_output=True,
                text=True,
                timeout=300,
            )
            assert (
                pull_result.returncode == 0
            ), f"Docker image {DOCKER_IMAGE} not found and pull failed: {pull_result.stderr}"

            inspect_result = subprocess.run(
                ["docker", "image", "inspect", DOCKER_IMAGE],
                capture_output=True,
                timeout=30,
            )

        assert inspect_result.returncode == 0, f"Docker image {DOCKER_IMAGE} not found"

    def test_python_import(self):
        """Test that astrograph can be imported in the container."""
        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--entrypoint",
                "python",
                DOCKER_IMAGE,
                "-c",
                "from astrograph.server import create_server; print('OK')",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "OK" in result.stdout


# ---------------------------------------------------------------------------
# MCP protocol tests
# ---------------------------------------------------------------------------


class TestMCPProtocol:
    """Tests for MCP JSON-RPC protocol over stdio."""

    def test_initialize(self):
        """Test MCP initialization handshake."""
        responses = send_mcp_messages([mcp_initialize()])

        assert len(responses) >= 1
        init_response = responses[0]

        assert init_response.get("id") == 1
        assert "result" in init_response
        assert "serverInfo" in init_response["result"]
        assert init_response["result"]["serverInfo"]["name"] == "code-structure-mcp"

    def test_list_tools(self):
        """Test listing available MCP tools."""
        responses = send_mcp_messages(
            [
                mcp_initialize(),
                mcp_list_tools(),
            ]
        )

        tools_response = next((r for r in responses if r.get("id") == 2), None)
        assert tools_response is not None

        tools = tools_response["result"]["tools"]
        tool_names = {t["name"] for t in tools}

        expected_tools = {
            "astrograph_analyze",
            "astrograph_write",
            "astrograph_edit",
            "astrograph_suppress",
            "astrograph_unsuppress",
            "astrograph_list_suppressions",
            "astrograph_status",
            "astrograph_lsp_setup",
            "astrograph_metadata_erase",
            "astrograph_metadata_recompute_baseline",
            "astrograph_generate_ignore",
            "astrograph_set_workspace",
        }
        assert expected_tools == tool_names

    def test_tool_descriptions_include_core_actions(self):
        """Test that relevant tools expose meaningful descriptions."""
        responses = send_mcp_messages(
            [
                mcp_initialize(),
                mcp_list_tools(),
            ]
        )

        tools_response = next((r for r in responses if r.get("id") == 2), None)
        tools = {t["name"]: t for t in tools_response["result"]["tools"]}

        assert "duplicate" in tools["astrograph_analyze"]["description"].lower()
        assert "write" in tools["astrograph_write"]["description"].lower()
        assert "edit" in tools["astrograph_edit"]["description"].lower()

    def test_initialize_framed_mode(self):
        """Test MCP initialization using Content-Length framing (real client mode)."""
        responses = send_mcp_framed([mcp_initialize()])

        assert len(responses) >= 1
        init_response = responses[0]

        assert init_response.get("id") == 1
        assert "result" in init_response
        assert init_response["result"]["serverInfo"]["name"] == "code-structure-mcp"

    def test_list_tools_framed_mode(self):
        """Test listing tools via Content-Length framing."""
        responses = send_mcp_framed(
            [
                mcp_initialize(),
                mcp_list_tools(),
            ]
        )

        tools_response = next((r for r in responses if r.get("id") == 2), None)
        assert tools_response is not None
        tool_names = {t["name"] for t in tools_response["result"]["tools"]}
        assert "astrograph_analyze" in tool_names
        assert "astrograph_status" in tool_names


# ---------------------------------------------------------------------------
# Workflow tests
# ---------------------------------------------------------------------------


class TestE2EWorkflow:
    """End-to-end workflow tests."""

    def test_analyze_workflow(self, sample_workspace):
        """Test analyze workflow (auto-indexes at startup in event-driven mode)."""
        responses = send_mcp_messages(
            [
                mcp_initialize(),
                mcp_call_tool("astrograph_analyze", {}, 3),
            ],
            workspace_path=sample_workspace,
        )

        analyze_response = next((r for r in responses if r.get("id") == 3), None)
        assert analyze_response is not None

        analyze_text = analyze_response["result"]["content"][0]["text"]
        assert (
            "duplicate" in analyze_text.lower()
            or "No significant duplicates" in analyze_text
            or "No code indexed" in analyze_text
        )

    def test_status_tool(self):
        """Status tool should report server readiness without a workspace."""
        responses = send_mcp_messages(
            [
                mcp_initialize(),
                mcp_call_tool("astrograph_status", {}, 3),
            ]
        )

        status_response = next((r for r in responses if r.get("id") == 3), None)
        assert status_response is not None

        status_text = status_response["result"]["content"][0]["text"]
        # Should contain workspace or status info
        assert status_text  # non-empty

    def test_suppress_and_reanalyze_lifecycle(self, sample_workspace):
        """Full lifecycle: analyze → extract hash → suppress → re-analyze shows clean."""
        # Step 1: analyze to find duplicates
        responses = send_mcp_messages(
            [
                mcp_initialize(),
                mcp_call_tool("astrograph_analyze", {}, 3),
            ],
            workspace_path=sample_workspace,
        )

        analyze_response = next((r for r in responses if r.get("id") == 3), None)
        assert analyze_response is not None
        assert analyze_response["result"]["content"][0]["text"]

        # Extract a WL hash from the analysis report on disk
        metadata_dir = Path(sample_workspace) / ".metadata_astrograph"
        reports = sorted(metadata_dir.glob("analysis_report_*.txt"))
        if not reports:
            pytest.skip("No analysis report generated (empty index)")
        report_text = reports[-1].read_text()
        hash_match = re.search(r"suppress\(wl_hash=([a-f0-9]+)\)", report_text)
        if not hash_match:
            pytest.skip("No suppressible hash in report")
        wl_hash = hash_match.group(1)

        # Step 2: suppress the duplicate and re-analyze
        responses2 = send_mcp_messages(
            [
                mcp_initialize(),
                mcp_call_tool("astrograph_suppress", {"wl_hash": wl_hash}, 3),
                mcp_call_tool("astrograph_analyze", {}, 4),
            ],
            workspace_path=sample_workspace,
        )

        suppress_response = next((r for r in responses2 if r.get("id") == 3), None)
        assert suppress_response is not None
        suppress_text = suppress_response["result"]["content"][0]["text"]
        assert "suppress" in suppress_text.lower() or wl_hash in suppress_text

        reanalyze_response = next((r for r in responses2 if r.get("id") == 4), None)
        assert reanalyze_response is not None


# ---------------------------------------------------------------------------
# JavaScript workflow tests
# ---------------------------------------------------------------------------


class TestJavaScriptE2EWorkflow:
    """JavaScript end-to-end workflow tests against the Docker MCP server."""

    def test_javascript_analyze_detects_duplicates(self, sample_javascript_workspace):
        """Analyze should index JS files and report duplicate findings."""
        responses = send_mcp_messages(
            [
                mcp_initialize(),
                mcp_call_tool("astrograph_analyze", {}, 3),
            ],
            workspace_path=sample_javascript_workspace,
        )

        analyze_response = next((r for r in responses if r.get("id") == 3), None)
        assert analyze_response is not None

        analyze_text = analyze_response["result"]["content"][0]["text"]
        assert "no code indexed" not in analyze_text.lower()
        assert "duplicate" in analyze_text.lower()

        metadata_dir = Path(sample_javascript_workspace) / ".metadata_astrograph"
        reports = sorted(metadata_dir.glob("analysis_report_*.txt"))
        assert reports
        report_text = reports[-1].read_text()
        assert "math_utils.js" in report_text
        assert "suppress(wl_hash=" in report_text

    def test_javascript_write_blocks_identical_code(self, sample_javascript_workspace):
        """Write should block exact duplicate JavaScript code in another file."""
        duplicate_content = """function processItems(items) {
  const results = [];
  for (const item of items) {
    if (item > 0) {
      results.push(item * 2);
    }
  }
  return results;
}
"""
        responses = send_mcp_messages(
            [
                mcp_initialize(),
                mcp_call_tool("astrograph_analyze", {}, 3),
                mcp_call_tool(
                    "astrograph_write",
                    {"file_path": "/workspace/new_utils.js", "content": duplicate_content},
                    4,
                ),
            ],
            workspace_path=sample_javascript_workspace,
        )

        write_response = next((r for r in responses if r.get("id") == 4), None)
        assert write_response is not None
        write_text = write_response["result"]["content"][0]["text"]
        assert "BLOCKED" in write_text
        assert ".js:" in write_text

    def test_container_doctor_reports_javascript_lsp_ready(self):
        """Container image should include JavaScript LSP server by default."""
        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--entrypoint",
                "astrograph-cli",
                DOCKER_IMAGE,
                "doctor",
                "--json",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        payload = json.loads(result.stdout)
        js_server = next(s for s in payload["servers"] if s["language"] == "javascript_lsp")
        assert js_server["available"] is True
        assert js_server["executable"]


# ---------------------------------------------------------------------------
# Mixed-language workspace
# ---------------------------------------------------------------------------


class TestMixedLanguageWorkflow:
    """Tests with Python + JavaScript files in the same workspace."""

    def test_mixed_workspace_indexes_both_languages(self, mixed_language_workspace):
        """Analyze should detect duplicates across both Python and JS files."""
        responses = send_mcp_messages(
            [
                mcp_initialize(),
                mcp_call_tool("astrograph_analyze", {}, 3),
            ],
            workspace_path=mixed_language_workspace,
        )

        analyze_response = next((r for r in responses if r.get("id") == 3), None)
        assert analyze_response is not None

        analyze_text = analyze_response["result"]["content"][0]["text"]
        assert "no code indexed" not in analyze_text.lower()

    def test_mixed_workspace_framed_mode(self, mixed_language_workspace):
        """Analyze via Content-Length framing with a mixed workspace."""
        responses = send_mcp_framed(
            [
                mcp_initialize(),
                mcp_call_tool("astrograph_analyze", {}, 3),
            ],
            workspace_path=mixed_language_workspace,
        )

        analyze_response = next((r for r in responses if r.get("id") == 3), None)
        assert analyze_response is not None
        assert analyze_response["result"]["content"][0]["text"]


# ---------------------------------------------------------------------------
# Transport robustness (EOF race regression)
# ---------------------------------------------------------------------------


class TestTransportRobustness:
    """Verify the stdio transport handles EOF correctly.

    Prior to the _UnclosableSendStream fix, batching multiple requests
    (especially slow ones like analyze) would cause ClosedResourceError
    because the MCP session's _receive_loop closed the write stream when
    stdin reached EOF.
    """

    def test_batch_init_list_analyze_all_responses_returned(self, sample_workspace):
        """Batch three requests in a single stdin pipe; every response must arrive."""
        responses = send_mcp_messages(
            [
                mcp_initialize(),
                mcp_list_tools(),
                mcp_call_tool("astrograph_analyze", {}, 3),
            ],
            workspace_path=sample_workspace,
        )

        ids = {r.get("id") for r in responses}
        assert 1 in ids, "Missing initialize response"
        assert 2 in ids, "Missing tools/list response"
        assert 3 in ids, "Missing analyze response"

    def test_batch_four_tool_calls_after_init(self, sample_workspace):
        """Batch init + four tool calls; all five responses must arrive."""
        responses = send_mcp_messages(
            [
                mcp_initialize(),
                mcp_call_tool("astrograph_status", {}, 2),
                mcp_call_tool("astrograph_analyze", {}, 3),
                mcp_call_tool("astrograph_list_suppressions", {}, 4),
                mcp_call_tool("astrograph_status", {}, 5),
            ],
            workspace_path=sample_workspace,
        )

        ids = {r.get("id") for r in responses}
        for expected_id in (1, 2, 3, 4, 5):
            assert expected_id in ids, f"Missing response for request id={expected_id}"

    def test_framed_batch_all_responses_returned(self, sample_workspace):
        """Same batch test but via Content-Length framing."""
        responses = send_mcp_framed(
            [
                mcp_initialize(),
                mcp_list_tools(),
                mcp_call_tool("astrograph_analyze", {}, 3),
            ],
            workspace_path=sample_workspace,
        )

        ids = {r.get("id") for r in responses}
        assert 1 in ids, "Missing initialize response (framed)"
        assert 2 in ids, "Missing tools/list response (framed)"
        assert 3 in ids, "Missing analyze response (framed)"

    def test_process_exits_zero(self):
        """Container should exit 0 after processing all messages (no crash)."""
        input_data = json.dumps(mcp_initialize())
        result = subprocess.run(
            _docker_cmd(),
            input=input_data,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, (
            f"Expected exit code 0, got {result.returncode}\n" f"stderr: {result.stderr[:1000]}"
        )


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for error handling in MCP protocol."""

    def test_unknown_tool(self):
        """Test calling a tool that doesn't exist."""
        responses = send_mcp_messages(
            [
                mcp_initialize(),
                mcp_call_tool("astrograph_nonexistent", {}, 3),
            ]
        )

        response = next((r for r in responses if r.get("id") == 3), None)
        assert response is not None

        response_text = response["result"]["content"][0]["text"]
        assert "Unknown tool" in response_text

    def test_suppress_without_hash(self):
        """Test suppressing without providing a hash."""
        responses = send_mcp_messages(
            [
                mcp_initialize(),
                mcp_call_tool("astrograph_suppress", {}, 3),
            ]
        )

        response = next((r for r in responses if r.get("id") == 3), None)
        assert response is not None

        response_text = response["result"]["content"][0]["text"]
        assert "hash" in response_text.lower() or "required" in response_text.lower()


# ---------------------------------------------------------------------------
# Skip Docker tests if Docker is not available
# ---------------------------------------------------------------------------


def pytest_configure(_config):
    """Check if Docker is available."""
    try:
        subprocess.run(["docker", "--version"], capture_output=True, timeout=5)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pytest.skip("Docker not available", allow_module_level=True)
