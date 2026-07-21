"""Tests for hexabin/users.py — hashing, auth, seeding, CRUD."""

import pytest


def _setup(tmp_path, monkeypatch):
    """Isolated SQLite DB + the users module bound to it."""
    import hexabin.database as db

    monkeypatch.setattr(db, "DB_FILE", str(tmp_path / "users.db"))
    monkeypatch.setattr(db, "_initialized", False)
    monkeypatch.setattr(db, "_pg_pool", None)
    monkeypatch.setattr(db, "DB_BACKEND", "sqlite")
    db.init_db()
    import hexabin.users as users

    return users


# ─────────────────────────────────────────────────────────────────────────────
# Hashing
# ─────────────────────────────────────────────────────────────────────────────


class TestHashing:
    def test_hash_is_self_describing(self):
        from hexabin.users import hash_password

        h = hash_password("secret")
        assert h.startswith("pbkdf2_sha256$")
        assert len(h.split("$")) == 4

    def test_verify_correct(self):
        from hexabin.users import hash_password, verify_password

        assert verify_password("secret", hash_password("secret")) is True

    def test_verify_wrong(self):
        from hexabin.users import hash_password, verify_password

        assert verify_password("nope", hash_password("secret")) is False

    def test_salt_makes_hashes_differ(self):
        from hexabin.users import hash_password

        assert hash_password("secret") != hash_password("secret")

    def test_verify_malformed_stored(self):
        from hexabin.users import verify_password

        assert verify_password("x", "not-a-valid-hash") is False
        assert verify_password("x", "") is False


# ─────────────────────────────────────────────────────────────────────────────
# Seeding
# ─────────────────────────────────────────────────────────────────────────────


class TestSeed:
    def test_seeds_admin_on_empty(self, tmp_path, monkeypatch):
        users = _setup(tmp_path, monkeypatch)
        assert users.count_users() == 0
        users.seed_admin_if_empty()
        assert users.count_users() == 1
        assert users.user_exists("admin")

    def test_seed_is_idempotent(self, tmp_path, monkeypatch):
        users = _setup(tmp_path, monkeypatch)
        users.seed_admin_if_empty()
        users.seed_admin_if_empty()
        assert users.count_users() == 1

    def test_seeded_admin_password_verifies(self, tmp_path, monkeypatch):
        users = _setup(tmp_path, monkeypatch)
        users.seed_admin_if_empty()
        # Default env creds from config (admin / password123)
        from hexabin.config import ADMIN_PASSWORD, ADMIN_USERNAME

        assert users.verify_user(ADMIN_USERNAME, ADMIN_PASSWORD) is True


# ─────────────────────────────────────────────────────────────────────────────
# Authentication
# ─────────────────────────────────────────────────────────────────────────────


class TestAuth:
    def test_verify_user_roundtrip(self, tmp_path, monkeypatch):
        users = _setup(tmp_path, monkeypatch)
        users.create_user("op", "hunter2x")
        assert users.verify_user("op", "hunter2x") is True
        assert users.verify_user("op", "wrong") is False
        assert users.verify_user("ghost", "hunter2x") is False

    def test_verify_bearer_matches_any_user(self, tmp_path, monkeypatch):
        users = _setup(tmp_path, monkeypatch)
        users.create_user("op", "tokenpass")
        assert users.verify_bearer("tokenpass") is True
        assert users.verify_bearer("nope") is False
        assert users.verify_bearer("") is False


# ─────────────────────────────────────────────────────────────────────────────
# CRUD + policy
# ─────────────────────────────────────────────────────────────────────────────


class TestCrud:
    def test_create_and_list(self, tmp_path, monkeypatch):
        users = _setup(tmp_path, monkeypatch)
        users.create_user("alice", "password1")
        users.create_user("bob", "password2")
        names = {u["username"] for u in users.list_users()}
        assert names == {"alice", "bob"}
        # list never leaks hashes
        assert all("password_hash" not in u for u in users.list_users())

    def test_duplicate_username_fails(self, tmp_path, monkeypatch):
        users = _setup(tmp_path, monkeypatch)
        assert users.create_user("alice", "password1") is not None
        assert users.create_user("alice", "password2") is None

    def test_change_password(self, tmp_path, monkeypatch):
        users = _setup(tmp_path, monkeypatch)
        users.create_user("alice", "oldpassword")
        assert users.change_password("alice", "newpassword") is True
        assert users.verify_user("alice", "oldpassword") is False
        assert users.verify_user("alice", "newpassword") is True

    def test_delete_user(self, tmp_path, monkeypatch):
        users = _setup(tmp_path, monkeypatch)
        users.create_user("alice", "password1")
        assert users.delete_user("alice") is True
        assert users.user_exists("alice") is False
        assert users.delete_user("alice") is False  # already gone

    def test_valid_username(self, tmp_path, monkeypatch):
        users = _setup(tmp_path, monkeypatch)
        assert users.valid_username("good_user-1")
        assert not users.valid_username("ab")  # too short
        assert not users.valid_username("has space")
        assert not users.valid_username("bad!char")

    def test_valid_password(self, tmp_path, monkeypatch):
        users = _setup(tmp_path, monkeypatch)
        assert users.valid_password("123456")
        assert not users.valid_password("12345")
        assert not users.valid_password("")
