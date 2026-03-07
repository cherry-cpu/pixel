from __future__ import annotations

import base64
import hashlib
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

    def register_user(self, username: str, password: str) -> tuple[bool, str]:
        import datetime
        
        # Check if user exists
        existing = self.db.query("SELECT username FROM users WHERE username = ?", (username,))
        if existing:
            return False, "Username already exists."
        
        salt = os.urandom(32)
        key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100000)
        
        salt_hex = salt.hex()
        key_hex = key.hex()
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        self.db.execute(
            "INSERT INTO users (username, password_hash, salt, created_at) VALUES (?, ?, ?, ?)",
            (username, key_hex, salt_hex, now_iso)
        )
        return True, "User created successfully."

    def verify_user(self, username: str, password: str) -> bool:
        rows = self.db.query("SELECT password_hash, salt FROM users WHERE username = ?", (username,))
        if not rows:
            return False
            
        row = rows[0]
        stored_hash = row["password_hash"]
        stored_salt = bytes.fromhex(row["salt"])
        
        key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), stored_salt, 100000)
        return key.hex() == stored_hash

    def require_crypto(self) -> CryptoManager:
        if self.crypto is None:
            raise RuntimeError("Library locked. Provide passphrase in sidebar.")
        return self.crypto

