"""ASTrograph - MCP server for structural code duplication detection and language-aware semantic analysis."""

import warnings

# esprima 4.0.1 (unmaintained) uses nested character sets in scanner.py:1040
# that trigger FutureWarning on Python >=3.12. The regex works correctly.
warnings.filterwarnings("ignore", category=FutureWarning, module=r"esprima\.scanner")

__version__ = "0.5.66"
