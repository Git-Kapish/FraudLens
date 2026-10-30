"""The serve image must install the versions the models were trained with (they are pickles)."""

import re

from fraud.config import ROOT

SAME_PACKAGE = {"xgboost-cpu": "xgboost"}  # CPU-only build of the same release


def _pins(name: str) -> dict[str, str]:
    pins = {}
    for line in (ROOT / name).read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)(\[[^\]]*\])?==(\S+)", line)
        assert match, f"{name}: '{line}' is not pinned with =="
        package = match.group(1).lower()
        pins[SAME_PACKAGE.get(package, package)] = match.group(3)
    return pins


def test_serve_pins_match_training_pins():
    train, serve = _pins("requirements.txt"), _pins("requirements-serve.txt")
    assert set(serve) <= set(train), f"serve-only packages: {set(serve) - set(train)}"
    assert {p: serve[p] for p in serve} == {p: train[p] for p in serve}
