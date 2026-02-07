# Contributing to ASTrograph

## Development Setup

```bash
git clone https://github.com/Thaylo/astrograph.git
cd astrograph
pip install -e ".[dev]"

# If working on tree-sitter language plugins:
pip install -e ".[dev,treesitter]"
```

Run the test suite:

```bash
pytest tests/ -q
```

Verify local LSP prerequisites (useful for plugin-backed languages):

```bash
astrograph-cli doctor
astrograph-cli install-lsps --dry-run
```

Current first-party plugin-backed languages:
- `python` (`PythonLSPPlugin`)
- `javascript_lsp` (`JavaScriptLSPPlugin`)

## Adding a New Language Plugin

ASTrograph uses a plugin architecture for language support. Each language plugin tells the system how to parse source files into graphs that can be compared for structural equivalence.

### Choose a Base Class

There are three base classes in `src/astrograph/languages/`:

| Base Class | When to Use |
|------------|-------------|
| `BaseLanguagePlugin` | You have a custom parser (like Python's built-in `ast` module) |
| `LSPLanguagePluginBase` | You want code-unit extraction from Language Server Protocol document symbols |
| `TreeSitterLanguagePlugin` | You want to use a [tree-sitter](https://tree-sitter.github.io/) grammar (recommended for most languages) |

`TreeSitterLanguagePlugin` handles parsing, graph construction, and code unit extraction for you. You only need to implement ~5 small methods.
`LSPLanguagePluginBase` handles symbol-to-`CodeUnit` mapping and lightweight structural graphing; you provide an LSP client implementation.

### Step-by-Step (Tree-Sitter)

#### 1. Create the plugin file

Create `src/astrograph/languages/<language>_plugin.py`:

```python
"""<Language> language plugin using tree-sitter."""

from ._treesitter_base import TreeSitterLanguagePlugin


class <Language>Plugin(TreeSitterLanguagePlugin):
    """<Language> support via tree-sitter."""

    @property
    def language_id(self) -> str:
        return "<language>"

    @property
    def file_extensions(self) -> frozenset[str]:
        return frozenset({".ext1", ".ext2"})

    @property
    def skip_dirs(self) -> frozenset[str]:
        return frozenset({"<lang-specific-dirs>"})

    def _tree_sitter_language(self):
        import tree_sitter_<language>
        return tree_sitter_<language>.language()

    def _node_label(self, node, normalize_ops=False):
        # Return a structural label for this CST node.
        # See "Graph Labeling Guidelines" below.
        return node.type

    def _is_function_node(self, node):
        return node.type in ("<function_node_types>",)

    def _is_class_node(self, node):
        return node.type == "<class_node_type>"

    def _get_name(self, node):
        for child in node.children:
            if child.type == "identifier":
                return child.text.decode()
        return "<anonymous>"
```

#### 2. Implement the required methods

Each method has a specific role:

| Method | Purpose |
|--------|---------|
| `language_id` | Unique string identifier (e.g., `"javascript"`) |
| `file_extensions` | File extensions to match (e.g., `frozenset({".js", ".mjs"})`) |
| `skip_dirs` | Language-specific directories to skip during indexing |
| `_tree_sitter_language()` | Return the tree-sitter `Language` object |
| `_node_label(node, normalize_ops)` | Return a structural label for a CST node |
| `_is_function_node(node)` | Return `True` if the node is a function/method definition |
| `_is_class_node(node)` | Return `True` if the node is a class definition |
| `_get_name(node)` | Extract the name string from a function/class node |

Optional overrides:

| Method | Default | Purpose |
|--------|---------|---------|
| `_is_block_node(node)` | `False` | Enable block-level duplicate detection (for/while/if) |
| `_should_skip_node(node)` | Skips single-char punctuation | Filter noise nodes from the graph |
| `_get_block_type(node)` | `node.type` | Return block type name (e.g., `"for"`, `"if"`) |

#### 3. Register the plugin

Add your plugin class to the `astrograph.languages` entry point group in `pyproject.toml`:

```toml
[project.entry-points."astrograph.languages"]
python = "astrograph.languages.python_lsp_plugin:PythonLSPPlugin"
<language> = "astrograph.languages.<language>_plugin:<Language>Plugin"
```

For local experiments, you can also load plugin class paths without packaging:

```bash
export ASTROGRAPH_LANGUAGE_PLUGINS="astrograph.languages.<language>_plugin:<Language>Plugin"
```

#### 4. Verify plugin conformance tests

The shared `language_plugin` fixture auto-discovers registered plugins from `LanguageRegistry`, so your plugin is included automatically once it is discoverable.

Conformance checks include:
- `extract_code_units()` returns valid `CodeUnit` objects
- `source_to_graph()` returns a valid graph
- `code_unit_to_ast_graph()` computes metadata
- Structurally identical code produces equivalent graph hashes

#### 5. Add language-specific tests

Create `tests/languages/test_<language>_plugin.py` with tests for your language's specifics (e.g., language-specific node types, edge cases, syntax variations).

### Complete Example: JavaScript Plugin

```python
"""JavaScript language plugin using tree-sitter."""

from ._treesitter_base import TreeSitterLanguagePlugin


class JavaScriptPlugin(TreeSitterLanguagePlugin):
    """JavaScript support via tree-sitter."""

    @property
    def language_id(self) -> str:
        return "javascript"

    @property
    def file_extensions(self) -> frozenset[str]:
        return frozenset({".js", ".mjs", ".cjs"})

    @property
    def skip_dirs(self) -> frozenset[str]:
        return frozenset({"node_modules", "dist", "build"})

    def _tree_sitter_language(self):
        import tree_sitter_javascript
        return tree_sitter_javascript.language()

    def _node_label(self, node, normalize_ops=False):
        # Use node type as structural label.
        # Identifiers and literals get generic labels to ignore naming.
        if node.type == "identifier":
            return "identifier"
        if node.type in ("string", "number", "true", "false", "null"):
            return "literal"
        return node.type

    def _is_function_node(self, node):
        return node.type in (
            "function_declaration",
            "arrow_function",
            "method_definition",
        )

    def _is_class_node(self, node):
        return node.type == "class_declaration"

    def _get_name(self, node):
        for child in node.children:
            if child.type == "identifier":
                return child.text.decode()
        return "<anonymous>"

    def _is_block_node(self, node):
        return node.type in ("for_statement", "while_statement", "if_statement")

    def _get_block_type(self, node):
        return node.type.replace("_statement", "")
```

### Graph Labeling Guidelines

The `_node_label()` method is the most important method in your plugin. It determines what makes two pieces of code "structurally equivalent."

**Good labels are structural, not value-based:**

| Node | Good Label | Bad Label | Why |
|------|-----------|-----------|-----|
| Variable `x` | `"identifier"` | `"x"` | Variable names don't affect structure |
| String `"hello"` | `"literal"` | `"hello"` | Literal values don't affect structure |
| `+` operator | `"binary_expression"` or `"+"` | — | Either works; operators can be structural |
| `if` statement | `"if_statement"` | — | Node type is already structural |

The goal: two functions that do the same thing with different variable names should produce isomorphic graphs.

**When `normalize_ops` is `True`**, operators like `+`, `-`, `*` should all map to a generic label (e.g., `"binary_op"`). This catches cases like `a + b` vs `a - b` as structurally similar.

## Testing

### Conformance tests

The `language_plugin` parametrized fixture in `tests/languages/conftest.py` runs the same test suite against every registered plugin. Add your language ID to the fixture params and the conformance tests run automatically.

### Running tests

```bash
# All tests
pytest tests/ -q

# Language plugin tests only
pytest tests/languages/ -q

# Docker E2E protocol + integration checks
pytest tests/test_e2e_docker.py -q

# A specific language
pytest tests/languages/test_<language>_plugin.py -q
```

## Code Quality

The project uses these tools (enforced in CI):

| Tool | Purpose | Config |
|------|---------|--------|
| **ruff** | Linting and formatting | `pyproject.toml` `[tool.ruff]` |
| **mypy** | Type checking | `pyproject.toml` `[tool.mypy]` |
| **pytest** | Tests with coverage | `pyproject.toml` `[tool.pytest]` |
| **vulture** | Dead code detection | `pyproject.toml` `[tool.vulture]` |

Run before submitting:

```bash
ruff check src/ tests/
ruff format src/ tests/
mypy src/astrograph
pytest tests/ -q
```

Coverage threshold is 92%. The `_treesitter_base.py` module is excluded from coverage since it requires tree-sitter grammars.
