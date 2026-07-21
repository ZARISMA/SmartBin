"""
hexabin/users.py — dashboard accounts and password hashing.

The single hardcoded ``ADMIN_USERNAME``/``ADMIN_PASSWORD`` login is replaced by
DB-backed accounts. The database is authoritative: the env credentials only
**seed** the first account when the ``users`` table is empty (a break-glass that
re-activates if the table is ever emptied). Every account is a full admin.

Hashing is stdlib only — PBKDF2-HMAC-SHA256 with a per-user salt, constant-time
compare — so there is no new dependency. Stored format::

    pbkdf2_sha256$<iterations>$<salt_hex>$<hash_hex>

This module owns hashing + policy and delegates all persistence to
``database.py``.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets

from . import database as db
from .config import ADMIN_PASSWORD, ADMIN_USERNAME
from .log_setup import get_logger

logger = get_logger()

_ALGO = "pbkdf2_sha256"
_ITERATIONS = 200_000

#: Usernames: 3–32 chars of letters/digits/dot/underscore/hyphen.
USERNAME_RE = re.compile(r"^[A-Za-z0-9._-]{3,32}$")
MIN_PASSWORD_LEN = 6


# ── Hashing ───────────────────────────────────────────────────────────────────


def hash_password(password: str) -> str:
    """Return a self-describing PBKDF2 hash string for *password*."""
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _ITERATIONS)
    return f"{_ALGO}${_ITERATIONS}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """Constant-time check of *password* against a stored hash string."""
    try:
        algo, iters, salt_hex, hash_hex = stored.split("$")
    except (ValueError, AttributeError):
        return False
    if algo != _ALGO:
        return False
    try:
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, int(iters))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(dk, expected)


# ── Policy helpers ────────────────────────────────────────────────────────────


def valid_username(username: str) -> bool:
    return bool(username and USERNAME_RE.match(username))


def valid_password(password: str) -> bool:
    return bool(password) and len(password) >= MIN_PASSWORD_LEN


# ── Authentication ────────────────────────────────────────────────────────────


def verify_user(username: str, password: str) -> bool:
    """True when *username* exists and *password* matches its stored hash."""
    if not username or not password:
        return False
    user = db.get_user(username)
    if not user:
        return False
    return verify_password(password, user["password_hash"])


def verify_bearer(token: str) -> bool:
    """True when *token* matches any account's password.

    Preserves the legacy "admin password doubles as an API bearer token"
    behaviour now that credentials live in the database.
    """
    if not token:
        return False
    for stored in db.list_password_hashes():
        if verify_password(token, stored):
            return True
    return False


# ── Seeding & CRUD (thin wrappers that own hashing) ───────────────────────────


def seed_admin_if_empty() -> None:
    """Create the initial admin from env credentials iff no accounts exist."""
    try:
        if db.count_users() == 0:
            uid = db.create_user(ADMIN_USERNAME, hash_password(ADMIN_PASSWORD))
            if uid is not None:
                logger.info("Seeded initial admin account %r from env.", ADMIN_USERNAME)
    except Exception as e:  # never let seeding crash startup
        logger.error("Admin seed failed: %s", e)


def create_user(username: str, password: str) -> int | None:
    """Hash *password* and insert the account. Returns id, or None on failure."""
    return db.create_user(username, hash_password(password))


def change_password(username: str, password: str) -> bool:
    """Hash *password* and update the account. Returns True if a row changed."""
    return db.set_password(username, hash_password(password))


def list_users() -> list[dict]:
    return db.list_users()


def delete_user(username: str) -> bool:
    return db.delete_user(username)


def count_users() -> int:
    return db.count_users()


def user_exists(username: str) -> bool:
    return db.get_user(username) is not None
