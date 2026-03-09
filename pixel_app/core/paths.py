from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppPaths:
    root: Path

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "index.sqlite"

    @property
    def photos_dir(self) -> Path:
        return self.data_dir / "photos.enc"

    @property
    def thumbs_dir(self) -> Path:
        return self.data_dir / "thumbs"

    @property
    def shares_dir(self) -> Path:
        return self.data_dir / "shares"

