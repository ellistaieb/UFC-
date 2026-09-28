"""Acquisition of raw UFC Stats tables (mirror download or local CSV import).

Direct scraping of ufcstats.com was not possible from the development environment
(host blocked by the network proxy) and its terms of use could not be checked there.
We therefore rely on a public mirror of the same tables, produced by an open-source
ufcstats.com scraper, or on a local folder of CSV files with the same schema.
Raw files are kept unchanged in `data/raw/ufcstats/` with a provenance sidecar.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from ufc_quant.data.http import PoliteDownloader, _now

log = logging.getLogger(__name__)


def acquire_ufcstats(cfg: dict, raw_dir: Path, refresh: bool = False) -> dict:
    src = cfg["sources"]["ufcstats"]
    out_dir = Path(raw_dir) / "ufcstats"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"source": None, "files": {}, "acquired_at": _now()}

    if src.get("local_dir"):
        local = Path(src["local_dir"])
        manifest["source"] = f"local_dir:{local.resolve()}"
        for name in src["files"]:
            shutil.copy2(local / name, out_dir / name)
            manifest["files"][name] = {"copied_from": str(local / name)}
    else:
        dl = PoliteDownloader.from_config(cfg)
        manifest["source"] = src["mirror_base_url"]
        for name in src["files"]:
            meta = dl.fetch_to_file(f"{src['mirror_base_url']}/{name}", out_dir / name, refresh=refresh)
            manifest["files"][name] = meta
            log.info("ufcstats %s: %s bytes", name, meta.get("bytes"))

    manifest["notes"] = (
        "Tables originally scraped from ufcstats.com. ufcstats.com was not reachable from the "
        "development environment; its terms of use must be checked before any direct scraping."
    )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
