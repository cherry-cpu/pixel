from __future__ import annotations

import base64
import os
from dataclasses import dataclass

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


def _derive_key(passphrase: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=390_000,
    )
    return base64.urlsafe_b64encode(kdf.derive(passphrase.encode("utf-8")))


@dataclass
class CryptoManager:
    salt: bytes
    _fernet: Fernet

    @classmethod
    def from_passphrase(cls, passphrase: str, salt: bytes) -> "CryptoManager":
        key = _derive_key(passphrase, salt)
        return cls(salt=salt, _fernet=Fernet(key))

    @staticmethod
    def new_salt() -> bytes:
        return os.urandom(16)

    def encrypt(self, data: bytes) -> bytes:
        return self._fernet.encrypt(data)

    def decrypt(self, token: bytes) -> bytes:
        return self._fernet.decrypt(token)


class UnlockError(Exception):
    pass


def validate_unlock(crypto: CryptoManager, challenge_ciphertext: bytes, expected_plain: bytes) -> None:
    try:
        plain = crypto.decrypt(challenge_ciphertext)
    except InvalidToken as e:
        raise UnlockError("Incorrect passphrase (failed to decrypt library).") from e
    if plain != expected_plain:
        raise UnlockError("Incorrect passphrase (library check failed).")

