"""Local password authentication for chat CLI access."""

from __future__ import annotations

import os
import sys
from getpass import getpass

import bcrypt

MAX_LOGIN_ATTEMPTS = 3
PASSWORD_HASH_ENV = "CHAT_PASSWORD_HASH"


class AuthenticationError(Exception):
    """Raised when authentication fails."""


def hash_password(password: str) -> str:
    if not password:
        raise ValueError("password must not be empty")
    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12))
    return hashed.decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    if not password or not password_hash:
        return False
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


def load_password_hash() -> str:
    password_hash = os.environ.get(PASSWORD_HASH_ENV, "").strip()
    if not password_hash:
        raise AuthenticationError(
            f"{PASSWORD_HASH_ENV} is not configured; "
            "rebuild the container image with a password hash"
        )
    return password_hash


def authenticate(max_attempts: int = MAX_LOGIN_ATTEMPTS) -> None:
    password_hash = load_password_hash()
    for attempt in range(1, max_attempts + 1):
        password = getpass("Password: ")
        if verify_password(password, password_hash):
            return
        remaining = max_attempts - attempt
        if remaining:
            print(f"Authentication failed. {remaining} attempt(s) remaining.", file=sys.stderr)
        else:
            raise AuthenticationError("too many failed login attempts")
