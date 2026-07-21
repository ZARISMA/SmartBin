# 1. Database-authoritative auth, seeded from env

Date: 2026-07-21

## Status

Accepted

## Context

Until now the dashboard had a single operator whose credentials lived in env
vars (`HEXABIN_ADMIN_USERNAME` / `HEXABIN_ADMIN_PASSWORD`). `login()` did a
plaintext equality check against those values, and the same admin password
doubled as an API bearer token. There was no user table, no password hashing,
and no way to change the password or add operators from the UI.

The admin panel now needs to **change passwords and add users at runtime**.
Runtime-mutable credentials cannot live in immutable env/`.env` config, so
accounts must move into a writable store (the database). That forces a decision
about what happens to the original env credentials.

## Decision

The **database is authoritative** for accounts. A new `users` table
(`username`, `password_hash`, `created_at`) holds every operator; passwords are
hashed with stdlib PBKDF2-HMAC-SHA256 (per-user salt, constant-time compare) —
no new dependency.

`HEXABIN_ADMIN_USERNAME` / `HEXABIN_ADMIN_PASSWORD` are demoted to a **seed**:
on startup, if the `users` table is empty, one account is created from them.
Once any account exists, the env values are ignored. Changing your password in
the UI therefore retires the old one immediately — including the default
`password123`. The env seed only re-activates if the table is emptied, which is
the intended break-glass against lockout.

Bearer-token auth (the "admin password as an API token" behaviour) is preserved
by matching the presented token against every stored hash.

## Consequences

- **Positive:** passwords are hashed at rest; operators are self-service;
  changing the default password actually retires it; no new dependency.
- **Positive:** the seed keeps first-run and disaster-recovery simple — an empty
  DB still comes up loginable with the documented default.
- **Negative / surprising:** after first boot, editing the env password has **no
  effect** on existing accounts. This surprises anyone who expects env to stay
  authoritative; it is the reason this ADR exists. Documented in CLAUDE.md.
- **Trade-off considered:** keeping env always-valid as a recovery login was
  rejected because it means the default `password123` stays usable until the env
  var is *also* changed and the process restarted — i.e. "change password"
  wouldn't fully retire the old one.
- The last remaining account cannot be deleted (enforced server-side) so the
  fleet can never be locked out through the UI.
