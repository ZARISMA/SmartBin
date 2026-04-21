from smartwaste.warnings import (
    SEVERITY_ERROR,
    SEVERITY_INFO,
    SEVERITY_WARNING,
    WarningRegistry,
)


def test_add_dedups_by_code():
    reg = WarningRegistry()
    reg.add("X", "first", SEVERITY_WARNING)
    reg.add("X", "second", SEVERITY_ERROR)
    items = reg.list()
    assert len(items) == 1
    assert items[0]["message"] == "second"
    assert items[0]["severity"] == SEVERITY_ERROR


def test_clear_removes_single_code():
    reg = WarningRegistry()
    reg.add("A", "a", SEVERITY_INFO)
    reg.add("B", "b", SEVERITY_WARNING)
    reg.clear("A")
    codes = {w["code"] for w in reg.list()}
    assert codes == {"B"}


def test_clear_all():
    reg = WarningRegistry()
    reg.add("A", "a")
    reg.add("B", "b")
    reg.clear_all()
    assert reg.list() == []


def test_invalid_severity_falls_back_to_warning():
    reg = WarningRegistry()
    reg.add("Z", "bad sev", severity="bogus")
    assert reg.list()[0]["severity"] == SEVERITY_WARNING
