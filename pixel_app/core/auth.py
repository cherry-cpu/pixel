from __future__ import annotations

import base64
import hashlib
import os
from dataclasses import dataclass

from pixel_app.core.db import DB


@dataclass
class Auth:
    db: DB

    def ensure_initialized(self) -> None:
        self.db.set_meta("lib.created_by", "pixel")
        self.db.set_meta("lib.version", "1")



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



