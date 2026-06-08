```markdown
# gnome-materials Development Patterns

> Auto-generated skill from repository analysis

## Overview
This skill covers the development patterns and conventions used in the `gnome-materials` Python repository. It documents file organization, code style, commit practices, and testing approaches to help contributors write consistent, maintainable code.

## Coding Conventions

### File Naming
- Use **snake_case** for all file names.
  - Example: `material_utils.py`, `color_palette.py`

### Import Style
- Use **relative imports** within the package.
  - Example:
    ```python
    from .color_palette import get_palette
    ```

### Export Style
- Use **named exports**; explicitly define what is exported from each module.
  - Example:
    ```python
    __all__ = ['get_palette', 'Material']
    ```

### Commit Messages
- Follow **conventional commit** format.
- Use the `feat` prefix for new features.
  - Example: `feat: add support for custom color themes`

## Workflows

### Adding a New Feature
**Trigger:** When implementing a new capability or function  
**Command:** `/add-feature`

1. Create a new Python file using snake_case if needed.
2. Implement the feature using relative imports for internal modules.
3. Add named exports to the module's `__all__` list.
4. Write or update corresponding test files (see Testing Patterns).
5. Commit your changes using the conventional commit format:
    ```
    feat: short description of the feature
    ```
6. Push your branch and open a pull request.

### Writing Tests
**Trigger:** When adding or updating code that requires verification  
**Command:** `/write-test`

1. Create a new test file named with the pattern `*.test.*` (e.g., `material_utils.test.py`).
2. Write test cases for the new or updated functionality.
3. Use standard Python `assert` statements or your preferred testing framework.
4. Run the tests to ensure correctness.

### Refactoring Code
**Trigger:** When improving code structure without changing its behavior  
**Command:** `/refactor`

1. Update code using snake_case for files and relative imports.
2. Ensure all exports remain named and explicit.
3. Update or add tests if necessary.
4. Commit using the conventional format:
    ```
    feat: refactor [module] for clarity
    ```
5. Push and open a pull request.

## Testing Patterns

- Test files follow the `*.test.*` naming pattern (e.g., `color_palette.test.py`).
- The testing framework is not specified; use Python's built-in `unittest`, `pytest`, or simple `assert` statements.
- Place test files alongside the modules they test or in a dedicated `tests/` directory if present.

**Example test file:**
```python
# color_palette.test.py

from .color_palette import get_palette

def test_get_palette():
    palette = get_palette('default')
    assert 'background' in palette
```

## Commands
| Command        | Purpose                                   |
|----------------|-------------------------------------------|
| /add-feature   | Start the workflow for adding a new feature|
| /write-test    | Begin writing or updating tests            |
| /refactor      | Refactor code following conventions        |
```
