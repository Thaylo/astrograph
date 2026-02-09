# Minimal Docker image for ASTrograph MCP server
FROM python:3.12-slim

# OCI labels for Docker Desktop and registries
LABEL org.opencontainers.image.source="https://github.com/Thaylo/astrograph"
LABEL org.opencontainers.image.url="https://github.com/Thaylo/astrograph"
LABEL org.opencontainers.image.documentation="https://github.com/Thaylo/astrograph#readme"
LABEL org.opencontainers.image.description="MCP server for structural code duplication detection and language-aware semantic analysis"
LABEL org.opencontainers.image.licenses="MIT"

WORKDIR /app

# Bundle JS LSP runtime for frictionless plugin-based adoption.
RUN apt-get update && \
    apt-get install -y --no-install-recommends nodejs npm && \
    npm install -g typescript typescript-language-server && \
    rm -rf /var/lib/apt/lists/*

# Copy all required files
COPY pyproject.toml README.md ./
COPY src/ src/

# Upgrade pip to fix CVE-2025-8869 and CVE-2026-1703, then install
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# MCP server runs on stdio
ENTRYPOINT ["python", "-m", "astrograph.server"]
