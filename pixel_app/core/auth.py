from __future__ import annotations

import base64
import os
from dataclasses import dataclass

from pixel_app.core.db import DB
from pixel_app.core.encryption import CryptoManager, UnlockError, validate_unlock


LIB_SALT_KEY = "lib.salt_b64"
LIB_CHALLENGE_KEY = "lib.challenge_b64"
LIB_CHALLENGE_PLAIN = b"pixel-library-unlock-check-v1"


@dataclass
class Auth:
    db: DB
    crypto: CryptoManager | None = None

    def ensure_initialized(self) -> None:
        salt_b64 = self.db.get_meta(LIB_SALT_KEY)
        challenge_b64 = self.db.get_meta(LIB_CHALLENGE_KEY)

        if salt_b64 and challenge_b64:
            return

        salt = CryptoManager.new_salt()
        bootstrap = CryptoManager.from_passphrase("bootstrap", salt)
        challenge = bootstrap.encrypt(LIB_CHALLENGE_PLAIN)

        self.db.set_meta(LIB_SALT_KEY, base64.b64encode(salt).decode("ascii"))
        self.db.set_meta(LIB_CHALLENGE_KEY, base64.b64encode(challenge).decode("ascii"))
        self.db.set_meta("lib.created_by", "pixel")
        self.db.set_meta("lib.version", "1")

    def ensure_unlocked(self, passphrase: str) -> tuple[bool, str]:
        self.ensure_initialized()
        salt = base64.b64decode(self.db.get_meta(LIB_SALT_KEY) or "")
        challenge = base64.b64decode(self.db.get_meta(LIB_CHALLENGE_KEY) or "")

        # If it’s a new library, allow first unlock to set passphrase:
        # We detect “new” by checking whether challenge decrypts with bootstrap.
        bootstrap = CryptoManager.from_passphrase("bootstrap", salt)
        try:
            validate_unlock(bootstrap, challenge, LIB_CHALLENGE_PLAIN)
            is_new = True
        except UnlockError:
            is_new = False

        crypto = CryptoManager.from_passphrase(passphrase, salt)
        if is_new:
            # re-encrypt challenge under real passphrase
            new_challenge = crypto.encrypt(LIB_CHALLENGE_PLAIN)
            self.db.set_meta(LIB_CHALLENGE_KEY, base64.b64encode(new_challenge).decode("ascii"))
            self.crypto = crypto
            return True, "Library initialized."

        try:
            validate_unlock(crypto, challenge, LIB_CHALLENGE_PLAIN)
        except UnlockError as e:
            self.crypto = None
            return False, str(e)

        self.crypto = crypto
        return True, "Unlocked."

    def require_crypto(self) -> CryptoManager:
        if self.crypto is None:
            raise RuntimeError("Library locked. Provide passphrase in sidebar.")
        return self.crypto

