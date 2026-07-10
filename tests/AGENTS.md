# tests/ — pytest Suite

## OVERVIEW
13 tests, all passing. Two test files: config loader unit tests (9) + pipeline integration tests (3). Uses pytest fixtures, `unittest.mock`, and `pytest.raises`.

## STRUCTURE
```
tests/
├── conftest.py              # 2 fixtures (project_root, temp_dir)
├── test_config_loader.py    # 9 tests: merge logic, validation, loading
└── test_pipeline.py         # 3 tests: config completeness, CLI override, mock YOLO
```

## CONVENTIONS

- **Naming**: `test_<function>_<scenario>` — Chinese docstrings allowed
- **Style**: Pure pytest; one class-based group (`TestTrainConfigChain`) for organization
- **Imports**: `sys.path.insert(0, ...)` hacks exist but are REDUNDANT — `pythonpath = ["scripts"]` in pyproject.toml already resolves imports
- **Gap**: No integration tests for actual `train.py`/`eval.py`/`export.py` entry-point execution; no C# tests
- **Fixtures**: `temp_dir` fixture defined but never used

## ANTI-PATTERNS

- **DO NOT** add new `sys.path.insert` calls — pyproject.toml `pythonpath` handles module resolution
- **DO NOT** skip return type annotations on new test helpers
- Clean up: remove `sys.path.insert` hacks from existing test files, remove or use `temp_dir` fixture

## COMMANDS
```bash
pytest tests/ -v          # All 13 tests
pytest tests/test_config_loader.py -v   # Config tests only
```
