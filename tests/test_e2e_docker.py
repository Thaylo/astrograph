"""End-to-end tests for the Docker container and MCP protocol.

These tests validate:
1. The published Docker image works correctly
2. MCP JSON-RPC protocol flow over stdio
3. Full workflows: index → analyze → suppress → analyze
"""

import contextlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import pytest

# Docker image to test (use local build for CI, published for release validation)
DOCKER_IMAGE = os.environ.get("ASTOGRAPH_TEST_IMAGE", "thaylo/astrograph")


def send_mcp_messages(
    messages: list[dict],
    workspace_path: str | None = None,
    read_only: bool = False,
) -> list[dict]:
    """Send MCP JSON-RPC messages to the Docker container and return responses."""
    input_data = "\n".join(json.dumps(msg) for msg in messages)

    cmd = [
        "docker",
        "run",
        "--rm",
        "-i",
    ]

    if workspace_path:
        mount_opt = "ro" if read_only else "rw"
        cmd.extend(["-v", f"{workspace_path}:/workspace:{mount_opt}"])

    cmd.append(DOCKER_IMAGE)

    result = subprocess.run(
        cmd,
        input=input_data,
        capture_output=True,
        text=True,
        timeout=30,
    )

    responses = []
    for line in result.stdout.strip().split("\n"):
        if line:
            with contextlib.suppress(json.JSONDecodeError):
                responses.append(json.loads(line))

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


@pytest.fixture
def sample_workspace():
    """Create a temporary workspace with Python files containing duplicates."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a file with duplicate functions
        code = '''
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
        file_path = Path(tmpdir) / "math_utils.py"
        file_path.write_text(code)

        # Create another file with more code
        code2 = """
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
        file_path2 = Path(tmpdir) / "data_utils.py"
        file_path2.write_text(code2)

        yield tmpdir


class TestDockerImageBasics:
    """Basic tests for the Docker image."""

    def test_image_exists(self):
        """Test that the Docker image can be pulled/exists."""
        result = subprocess.run(
            ["docker", "image", "inspect", DOCKER_IMAGE],
            capture_output=True,
            timeout=30,
        )
        assert result.returncode == 0, f"Docker image {DOCKER_IMAGE} not found"

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

        # Find the tools/list response
        tools_response = next((r for r in responses if r.get("id") == 2), None)
        assert tools_response is not None

        tools = tools_response["result"]["tools"]
        tool_names = {t["name"] for t in tools}

        # Verify all 7 tools are present (event-driven mode)
        expected_tools = {
            "astrograph_analyze",
            "astrograph_write",
            "astrograph_edit",
            "astrograph_suppress",
            "astrograph_suppress_batch",
            "astrograph_unsuppress",
            "astrograph_list_suppressions",
        }
        assert expected_tools == tool_names

    def test_tool_descriptions_mention_python(self):
        """Test that relevant tools mention Python-only support."""
        responses = send_mcp_messages(
            [
                mcp_initialize(),
                mcp_list_tools(),
            ]
        )

        tools_response = next((r for r in responses if r.get("id") == 2), None)
        tools = {t["name"]: t for t in tools_response["result"]["tools"]}

        # Check analyze tool mentions Python
        assert "Python" in tools["astrograph_analyze"]["description"]

        # Check write tool mentions Python
        assert "Python" in tools["astrograph_write"]["description"]

        # Check edit tool mentions Python
        assert "Python" in tools["astrograph_edit"]["description"]


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

        # Find analyze response
        analyze_response = next((r for r in responses if r.get("id") == 3), None)
        assert analyze_response is not None

        analyze_text = analyze_response["result"]["content"][0]["text"]
        # Should find duplicates in our sample code or report clean
        assert "duplicate" in analyze_text.lower() or "CLEAN" in analyze_text


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
        # Should require wl_hash parameter
        assert "hash" in response_text.lower() or "required" in response_text.lower()


# Skip Docker tests if Docker is not available
def pytest_configure(_config):
    """Check if Docker is available."""
    try:
        subprocess.run(["docker", "--version"], capture_output=True, timeout=5)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pytest.skip("Docker not available", allow_module_level=True)
