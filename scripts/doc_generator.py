"""
Documentation Generator for Defensive Transitions Trigger Codebase

This script uses AST parsing to extract metadata from Python files including:
- Function signatures
- Docstrings
- Line numbers
- Imports
- Class definitions
"""

import ast
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import json


class FunctionInfo:
    def __init__(self, name: str, lineno: int, args: List[str], returns: Optional[str],
                 docstring: Optional[str], decorators: List[str]):
        self.name = name
        self.lineno = lineno
        self.args = args
        self.returns = returns
        self.docstring = docstring
        self.decorators = decorators

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'lineno': self.lineno,
            'args': self.args,
            'returns': self.returns,
            'docstring': self.docstring,
            'decorators': self.decorators
        }


class ClassInfo:
    def __init__(self, name: str, lineno: int, docstring: Optional[str],
                 methods: List[FunctionInfo], bases: List[str]):
        self.name = name
        self.lineno = lineno
        self.docstring = docstring
        self.methods = methods
        self.bases = bases

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'lineno': self.lineno,
            'docstring': self.docstring,
            'methods': [m.to_dict() for m in self.methods],
            'bases': self.bases
        }


class DocumentationExtractor:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.file_name = Path(file_path).name
        self.module_name = Path(file_path).stem

    def extract(self) -> Dict[str, Any]:
        """Extract all documentation metadata from a Python file."""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            source = f.read()
            line_count = source.count('\n') + 1

        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return {'error': f"Syntax error: {e}"}

        module_docstring = ast.get_docstring(tree)
        imports = self._extract_imports(tree)
        functions = self._extract_functions(tree)
        classes = self._extract_classes(tree)

        return {
            'file_path': self.file_path,
            'file_name': self.file_name,
            'module_name': self.module_name,
            'line_count': line_count,
            'module_docstring': module_docstring,
            'imports': imports,
            'functions': [f.to_dict() for f in functions],
            'classes': [c.to_dict() for c in classes]
        }

    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract all import statements."""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}" if module else alias.name)
        return sorted(set(imports))

    def _extract_functions(self, tree: ast.AST, class_methods: bool = False) -> List[FunctionInfo]:
        """Extract all function definitions."""
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Skip class methods when extracting module-level functions
                if not class_methods and self._is_method(node, tree):
                    continue

                args = [arg.arg for arg in node.args.args]
                returns = ast.unparse(node.returns) if node.returns else None
                docstring = ast.get_docstring(node)
                decorators = [ast.unparse(d) for d in node.decorator_list]

                functions.append(FunctionInfo(
                    name=node.name,
                    lineno=node.lineno,
                    args=args,
                    returns=returns,
                    docstring=docstring,
                    decorators=decorators
                ))

        return sorted(functions, key=lambda f: f.lineno)

    def _extract_classes(self, tree: ast.AST) -> List[ClassInfo]:
        """Extract all class definitions with their methods."""
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                docstring = ast.get_docstring(node)
                bases = [ast.unparse(base) for base in node.bases]

                # Extract methods
                methods = []
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        args = [arg.arg for arg in item.args.args]
                        returns = ast.unparse(item.returns) if item.returns else None
                        method_docstring = ast.get_docstring(item)
                        decorators = [ast.unparse(d) for d in item.decorator_list]

                        methods.append(FunctionInfo(
                            name=item.name,
                            lineno=item.lineno,
                            args=args,
                            returns=returns,
                            docstring=method_docstring,
                            decorators=decorators
                        ))

                classes.append(ClassInfo(
                    name=node.name,
                    lineno=node.lineno,
                    docstring=docstring,
                    methods=methods,
                    bases=bases
                ))

        return sorted(classes, key=lambda c: c.lineno)

    def _is_method(self, func_node: ast.FunctionDef, tree: ast.AST) -> bool:
        """Check if a function is a method (inside a class)."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if func_node in ast.walk(node):
                    return True
        return False


def generate_function_markdown(func: Dict[str, Any]) -> str:
    """Generate markdown documentation for a function."""
    md = f"### Function: `{func['name']}`\n\n"

    if func['docstring']:
        md += f"**Purpose**: {func['docstring'].split(chr(10))[0]}\n\n"

    # Signature
    args_str = ', '.join(func['args'])
    returns_str = f" -> {func['returns']}" if func['returns'] else ""
    md += f"**Signature**:\n```python\ndef {func['name']}({args_str}){returns_str}:\n```\n\n"

    # Parameters table
    if func['args']:
        md += "**Parameters**:\n"
        md += "| Parameter | Type | Description |\n"
        md += "|-----------|------|-------------|\n"
        for arg in func['args']:
            md += f"| `{arg}` | - | - |\n"
        md += "\n"

    # Full docstring
    if func['docstring']:
        md += f"**Documentation**:\n```\n{func['docstring']}\n```\n\n"

    md += f"**Location**: Line {func['lineno']}\n\n"

    return md


def generate_file_markdown(metadata: Dict[str, Any]) -> str:
    """Generate complete markdown documentation for a file."""
    md = f"# Module: {metadata['file_name']}\n\n"

    md += "## Overview\n\n"
    if metadata.get('module_docstring'):
        md += f"{metadata['module_docstring']}\n\n"

    md += "## Module Metadata\n\n"
    md += "| Attribute | Value |\n"
    md += "|-----------|-------|\n"
    md += f"| **Path** | `{metadata['file_path']}` |\n"
    md += f"| **Lines of Code** | {metadata['line_count']} |\n"
    md += f"| **Functions** | {len(metadata['functions'])} |\n"
    md += f"| **Classes** | {len(metadata['classes'])} |\n\n"

    if metadata.get('imports'):
        md += "## Dependencies\n\n"
        md += "```python\n"
        for imp in metadata['imports'][:20]:  # Limit to first 20
            md += f"{imp}\n"
        if len(metadata['imports']) > 20:
            md += f"... and {len(metadata['imports']) - 20} more\n"
        md += "```\n\n"

    if metadata.get('classes'):
        md += "## Classes\n\n"
        for cls in metadata['classes']:
            md += f"### Class: `{cls['name']}`\n\n"
            if cls['docstring']:
                md += f"{cls['docstring']}\n\n"
            md += f"**Location**: Line {cls['lineno']}\n\n"
            if cls['methods']:
                md += f"**Methods**: {len(cls['methods'])}\n"
                for method in cls['methods']:
                    md += f"- `{method['name']}()` (Line {method['lineno']})\n"
                md += "\n"

    if metadata.get('functions'):
        md += "## Functions\n\n"
        for func in metadata['functions']:
            md += generate_function_markdown(func)
            md += "---\n\n"

    return md


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python doc_generator.py <python_file>")
        sys.exit(1)

    file_path = sys.argv[1]

    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    extractor = DocumentationExtractor(file_path)
    metadata = extractor.extract()

    if 'error' in metadata:
        print(f"Error: {metadata['error']}")
        sys.exit(1)

    # Output JSON metadata
    if len(sys.argv) > 2 and sys.argv[2] == '--json':
        print(json.dumps(metadata, indent=2))
    else:
        # Output markdown documentation
        markdown = generate_file_markdown(metadata)
        print(markdown)
