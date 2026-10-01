"""Runtime metadata contains software versions and no physical parameters."""

from kynovar.utils.paths import find_repo_root
from kynovar.utils.reproducibility import runtime_record


def test_runtime_record_has_no_law_parameters() -> None:
    record = runtime_record(find_repo_root())
    assert record["project"] == "kynovar"
    assert record["version"] == "0.8.0"
    assert record["python"]
    assert record["numpy"]
    assert "seed" not in record
    assert "k" not in record
    assert "p" not in record
    commit = record["git_commit"]
    assert commit is None or (isinstance(commit, str) and len(commit) == 40)
