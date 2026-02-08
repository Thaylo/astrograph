"""Shared configuration-driven base for LSP-backed language plugins."""

from __future__ import annotations

from ._lsp_base import LSPClient, LSPLanguagePluginBase
from .lsp_client import create_subprocess_client_from_env


class ConfiguredLSPLanguagePluginBase(LSPLanguagePluginBase):
    """LSPLanguagePluginBase with env/default command wiring."""

    DEFAULT_COMMAND: tuple[str, ...] = ()
    COMMAND_ENV_VAR: str = ""
    TIMEOUT_ENV_VAR: str = ""

    def __init__(self, lsp_client: LSPClient | None = None) -> None:
        client = lsp_client
        if client is None:
            client = create_subprocess_client_from_env(
                default_command=self.DEFAULT_COMMAND,
                command_env_var=self.COMMAND_ENV_VAR,
                timeout_env_var=self.TIMEOUT_ENV_VAR,
                language_id=self.LANGUAGE_ID,
            )
        super().__init__(lsp_client=client)
