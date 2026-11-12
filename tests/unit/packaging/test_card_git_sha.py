from fastmri_knee.packaging import card


def test_git_sha_prefers_the_pinned_env_var(monkeypatch):
    monkeypatch.setenv("GIT_SHA", "c" * 40)
    assert card.git_sha() == "c" * 40


def test_git_sha_ignores_a_blank_env_var(monkeypatch):
    monkeypatch.setenv("GIT_SHA", "  ")
    monkeypatch.setattr(card.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError("no git")))
    assert card.git_sha() == "unknown"
