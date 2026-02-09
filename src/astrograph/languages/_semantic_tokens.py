"""LSP semantic token decoding and query utilities.

Decodes the flat integer array returned by ``textDocument/semanticTokens/full``
into structured token objects and provides an indexed query API for fast lookups
by token type, modifier, and text.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SemanticTokenLegend:
    """Server-advertised token type and modifier names."""

    token_types: tuple[str, ...]
    token_modifiers: tuple[str, ...]


@dataclass(frozen=True)
class SemanticToken:
    """Single resolved semantic token with source text."""

    line: int  # 0-based
    start_char: int  # 0-based
    length: int
    token_type: str  # resolved name, e.g. "keyword"
    modifiers: frozenset[str]  # e.g. {"static", "readonly"}
    text: str  # actual source text


@dataclass(frozen=True)
class SemanticTokenResult:
    """Full token response from the LSP server."""

    tokens: tuple[SemanticToken, ...]
    legend: SemanticTokenLegend


def decode_semantic_tokens(
    data: list[int],
    legend: SemanticTokenLegend,
    source_lines: list[str],
) -> tuple[SemanticToken, ...]:
    """Decode the flat ``data`` array into resolved :class:`SemanticToken` objects.

    The LSP protocol encodes tokens as groups of five integers:
    ``(deltaLine, deltaStartChar, length, tokenTypeIndex, modifierBitmask)``.
    Running line/char state is maintained across groups; type and modifier
    indices are resolved from *legend*.
    """
    tokens: list[SemanticToken] = []
    num_values = len(data)
    if num_values % 5 != 0:
        return ()

    current_line = 0
    current_char = 0

    for i in range(0, num_values, 5):
        delta_line = data[i]
        delta_start = data[i + 1]
        length = data[i + 2]
        type_index = data[i + 3]
        modifier_bits = data[i + 4]

        current_line += delta_line
        if delta_line > 0:
            current_char = delta_start
        else:
            current_char += delta_start

        # Guard against corrupted data producing negative positions
        if current_char < 0:
            current_char = 0

        # Resolve token type
        if 0 <= type_index < len(legend.token_types):
            token_type = legend.token_types[type_index]
        else:
            token_type = f"unknown_{type_index}"

        # Resolve modifier bitmask
        modifiers: set[str] = set()
        if modifier_bits:
            for bit_pos, mod_name in enumerate(legend.token_modifiers):
                if modifier_bits & (1 << bit_pos):
                    modifiers.add(mod_name)

        # Extract text from source
        if 0 <= current_line < len(source_lines):
            line_text = source_lines[current_line]
            text = line_text[current_char : current_char + length]
        else:
            text = ""

        tokens.append(
            SemanticToken(
                line=current_line,
                start_char=current_char,
                length=length,
                token_type=token_type,
                modifiers=frozenset(modifiers),
                text=text,
            )
        )

    return tuple(tokens)


class TokenIndex:
    """Indexed query API over a collection of semantic tokens.

    Internally groups tokens by ``token_type`` for O(1) type lookups,
    and builds a text-to-types mapping for fast text queries.
    """

    def __init__(self, tokens: tuple[SemanticToken, ...]) -> None:
        self._tokens = tokens
        # token_type -> list of tokens
        self._by_type: dict[str, list[SemanticToken]] = {}
        # text -> set of token_types that contain it
        self._text_types: dict[str, set[str]] = {}
        # modifier -> list of tokens
        self._by_modifier: dict[str, list[SemanticToken]] = {}

        for token in tokens:
            self._by_type.setdefault(token.token_type, []).append(token)
            self._text_types.setdefault(token.text, set()).add(token.token_type)
            for mod in token.modifiers:
                self._by_modifier.setdefault(mod, []).append(token)

    def has_text(self, text: str, *, token_type: str | None = None) -> bool:
        """Check if *text* appears as a token, optionally restricted to *token_type*."""
        if token_type is not None:
            types = self._text_types.get(text)
            return types is not None and token_type in types
        return text in self._text_types

    def has_type(self, token_type: str) -> bool:
        """Check if any token of *token_type* exists."""
        return token_type in self._by_type

    def has_modifier(self, modifier: str) -> bool:
        """Check if any token carries *modifier*."""
        return modifier in self._by_modifier

    def texts_of_type(self, token_type: str) -> set[str]:
        """Return the set of distinct text values for tokens of *token_type*."""
        tokens = self._by_type.get(token_type)
        if tokens is None:
            return set()
        return {t.text for t in tokens}

    def count_type(self, token_type: str) -> int:
        """Return the number of tokens of *token_type*."""
        tokens = self._by_type.get(token_type)
        return len(tokens) if tokens is not None else 0
