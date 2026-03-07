from __future__ import annotations

import io
import json
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from cryptography.fernet import Fernet

from pixel_app.core.auth import Auth
from pixel_app.core.db import DB
from pixel_app.core.library import Library


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ShareService:
    db: DB
    auth: Auth
    library: Library
    shares_dir: Path

    def init_dirs(self) -> None:
        self.shares_dir.mkdir(parents=True, exist_ok=True)

    def create_share_package(self, photo_ids: list[str], note: str | None = None) -> dict:
        """
        Creates an encrypted payload on disk and returns:
        - token: share token (also encryption key)
        - payload_path: file path on disk
        - download_bytes: a small JSON file user can download (token + metadata)
        """
        self.init_dirs()
        # We no longer require crypto to read photos

        share_token = str(uuid.uuid4()).replace("-", "")
        key_b64 = share_token.encode("ascii")[:32].ljust(32, b"x")
        f = Fernet(key_b64)

        # Build a zip in-memory containing raw images + minimal metadata
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
            manifest = {"created_at": _utc_now_iso(), "photos": [], "note": note or ""}
            for pid in photo_ids:
                row = self.library.get_photo(pid)
                if not row:
                    continue
                raw = self.library.read_photo_bytes(row)
                name = row["original_name"]
                arcname = f"{pid}_{name}"
                z.writestr(arcname, raw)
                manifest["photos"].append(
                    {"id": pid, "original_name": name, "mime": row.get("mime", "image/jpeg")}
                )
            z.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))

        zip_bytes = buf.getvalue()
        enc_payload = f.encrypt(zip_bytes)

        payload_path = self.shares_dir / f"{uuid.uuid4()}.share"
        payload_path.write_bytes(enc_payload)

        self.db.execute(
            "INSERT INTO shares(token, created_at, note, payload_path) VALUES(?, ?, ?, ?)",
            (token, _utc_now_iso(), note or "", str(payload_path)),
        )

        download_bytes = json.dumps(
            {
                "token": token,
                "created_at": _utc_now_iso(),
                "note": note or "",
                "hint": "Open Pixel > Share > Import package, paste token, and upload the .share file (or point to it if on same machine).",
            },
            indent=2,
        ).encode("utf-8")

        return {
            "token": token,
            "payload_path": str(payload_path),
            "download_bytes": download_bytes,
        }

    def decrypt_share_payload(self, token: str, payload_bytes: bytes) -> bytes:
        f = Fernet(token.encode("ascii"))
        return f.decrypt(payload_bytes)

