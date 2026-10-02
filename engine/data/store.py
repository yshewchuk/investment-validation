"""Tier-2 storage: partitioned, idempotent, content-hashed.

Tables live at ``data/curated/{table}/year=YYYY/part.parquet``. Partitioning by
year is what makes a rebuild affordable: a normalizer that only touched 2024
rewrites one partition instead of six million rows, and a consumer that only
needs 2018–2020 reads three files.

Writes are idempotent by construction — a partition is written to a temporary
name and moved into place, so a rebuild interrupted halfway leaves either the
old partition or the new one, never a half-written file. Re-running a rebuild
produces byte-identical partitions, which is what the determinism check asserts.

Parquet via pyarrow is the format; ``csv.gz`` is the fallback for an
environment where pyarrow cannot be installed. Both satisfy the same contracts,
and :func:`table_format` reports which is in use so a report's provenance block
can record it.
"""
from __future__ import annotations

import gzip
import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from engine import paths
from engine.data.schemas import SCHEMAS, assert_schema, coerce, empty_frame

__all__ = [
    "HAVE_PARQUET",
    "table_format",
    "write_table",
    "write_partition",
    "PartitionedWriter",
    "read_table",
    "iter_table",
    "table_years",
    "table_stats",
    "TableStats",
    "drop_table",
    "file_sha256",
]

try:  # pragma: no cover - environment probe
    import pyarrow  # noqa: F401

    HAVE_PARQUET = True
except ImportError:  # pragma: no cover
    HAVE_PARQUET = False

SUFFIX = ".parquet" if HAVE_PARQUET else ".csv.gz"

# PartitionedWriter.finalize()'s default bucket count: rows are
# range-partitioned into roughly this many groups of distinct primary-key
# values (not row counts -- one ticker's rows all land in one bucket), so
# on the pyarrow-backed path the largest frame finalize() ever builds, in
# either phase, is one bucket's rows -- never the whole year at once,
# deduplicated or not. (Only on that path -- see _dedupe_and_write.)
FINALIZE_BUCKET_COUNT = 16


def table_format() -> str:
    return "parquet" if HAVE_PARQUET else "csv.gz"


def _part_name(part: int) -> str:
    return f"part-{part:04d}{SUFFIX}"


def _partition_file(part_dir: Path, part: int = 0) -> Path:
    return part_dir / _part_name(part)


def _partition_files(part_dir: Path) -> list[Path]:
    """Every part file in a partition, in deterministic order."""
    if not part_dir.exists():
        return []
    return sorted(
        p for p in part_dir.iterdir() if p.name.startswith("part-") and p.name.endswith(SUFFIX)
    )


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------


def _write_frame(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    if HAVE_PARQUET:
        # Repeated writes of identical data must be byte-identical (the
        # determinism check relies on it), which pyarrow's defaults give us as
        # long as nothing here injects a timestamp or a row-order dependency.
        df.to_parquet(tmp, engine="pyarrow", index=False, compression="snappy")
    else:
        with gzip.open(tmp, "wt", newline="") as fh:
            df.to_csv(fh, index=False)
    os.replace(tmp, path)


def write_partition(df: pd.DataFrame, name: str, year: int, part: int = 0) -> Path:
    """Write one part file of one year partition, replacing any existing one."""
    part_dir = paths.assert_writable(paths.curated_partition(name, year))
    part_dir.mkdir(parents=True, exist_ok=True)
    path = _partition_file(part_dir, part)
    _write_frame(df.reset_index(drop=True), path)
    return path


def _dedupe_and_write(
    parts: list[Path], key_cols: list[str], bucket_count: int, name: str, year: int
) -> int:
    """Deduplicate ``parts`` by primary key and write the single compacted
    partition file directly -- without ever holding the whole raw
    (duplicate-inflated) year, OR the whole deduplicated year, as one frame.

    Rows are range-partitioned by ``key_cols[0]`` (the primary key's most
    significant column, e.g. ``ticker``) into ``bucket_count`` temp files:
    two rows sharing a primary key always share ``key_cols[0]`` too, so two
    copies of the same key always land in the same bucket -- deduping
    bucket-by-bucket is exactly equivalent to deduping the whole year at
    once. ``keep="first"`` and parts are processed in their existing sorted
    order, so which row survives is unchanged -- the earliest-fed batch
    wins, exactly as before.

    Buckets are then processed in ascending ``key_cols[0]`` order and
    streamed into the output one at a time through a single
    ``pyarrow.parquet.ParquetWriter``: each bucket is fully sorted by the
    complete primary key before it is written, and no bucket's key range
    overlaps another's, so the concatenation of row groups the writer
    produces is already in full, correct primary-key order -- no final
    cross-bucket concat or sort is needed. On this path, the largest frame
    this function ever builds, in either phase, is one bucket's rows: a
    fraction of the year, never the whole year, deduplicated or not.

    Falls back to a single in-memory concat + dedupe + sort + write when
    pyarrow is unavailable (``HAVE_PARQUET`` is False) -- the streaming
    writer needs pyarrow directly, not just the pandas/parquet round-trip
    the rest of this module tolerates running without.

    Returns the number of rows removed.
    """
    range_col = key_cols[0]
    out_path = _partition_file(paths.curated_partition(name, year), 0)

    if not HAVE_PARQUET:
        frame = pd.concat([_read_part(p, None) for p in parts], ignore_index=True)
        before = len(frame)
        frame = frame.drop_duplicates(subset=key_cols, keep="first")
        frame.sort_values(key_cols, kind="stable", inplace=True)
        write_partition(frame, name, year, 0)
        return before - len(frame)

    import pyarrow as pa
    import pyarrow.parquet as pq

    values: set = set()
    for part in parts:
        values.update(pd.unique(_read_part(part, [range_col])[range_col]))
    ordered = sorted(values)
    n_groups = max(1, min(bucket_count, len(ordered)))
    groups = [
        g.tolist()
        for g in np.array_split(np.array(ordered, dtype=object), n_groups)
        if len(g)
    ]

    # Upper bound of every group except the last; groups are contiguous
    # ranges of the sorted distinct values, so one searchsorted call per
    # part assigns every row to its bucket -- instead of filtering the
    # whole chunk once per bucket.
    boundaries = [group[-1] for group in groups[:-1]]

    before_total = 0
    after_total = 0
    tmp_out = out_path.with_name(out_path.name + ".tmp")
    writer = None
    schema = None
    with tempfile.TemporaryDirectory(prefix="finalize-bucket-", dir=paths.DATA) as work_str:
        work = Path(work_str)
        for part_index, part in enumerate(parts):
            chunk = _read_part(part, None)
            bucket_idx = np.searchsorted(boundaries, chunk[range_col].to_numpy(), side="left")
            for b in range(len(groups)):
                sub = chunk[bucket_idx == b]
                if len(sub):
                    _write_frame(
                        sub.reset_index(drop=True),
                        work / f"bucket-{b:04d}-{part_index:04d}{SUFFIX}",
                    )
            del chunk

        try:
            for b in range(len(groups)):
                sub_files = sorted(
                    work.glob(f"bucket-{b:04d}-*{SUFFIX}"),
                    key=lambda p: int(p.name.split("-")[-1].split(".")[0]),
                )
                if not sub_files:
                    continue
                bucket_frame = pd.concat(
                    [_read_part(f, None) for f in sub_files], ignore_index=True
                )
                before_total += len(bucket_frame)
                bucket_frame = bucket_frame.drop_duplicates(subset=key_cols, keep="first")
                bucket_frame.sort_values(key_cols, kind="stable", inplace=True)
                after_total += len(bucket_frame)
                table = pa.Table.from_pandas(
                    bucket_frame.reset_index(drop=True),
                    schema=schema,
                    preserve_index=False,
                )
                del bucket_frame
                if writer is None:
                    schema = table.schema
                    writer = pq.ParquetWriter(tmp_out, schema)
                writer.write_table(table)
                del table
        finally:
            if writer is not None:
                writer.close()

    # `parts` is never empty (the caller already checks) and every part
    # file holds at least one row (the writer never flushes an empty
    # frame), so `groups` always has at least one non-empty bucket and
    # `writer` is always set by this point -- no FileNotFoundError here.
    os.replace(tmp_out, out_path)
    return before_total - after_total


class PartitionedWriter:
    """Streaming writer for tables too large to hold in memory at once.

    ``daily_market`` is ~9.4M rows and ``option_chains`` ~6.5M, against ~6 GB of
    usable RAM: materializing either as one frame is not an option. Batches are
    accumulated per year and flushed as numbered part files, so the writer's
    peak memory is one batch rather than one table.

    Determinism holds as long as the caller feeds batches in a fixed order —
    the rebuild iterates sorted source files, so it does. Each batch is sorted
    on the table's primary key before it is written.

    Use as a context manager; ``__exit__`` flushes whatever is buffered.
    """

    #: Total buffered rows across *all* years before a flush. The cap has to be
    #: global: a per-year threshold never trips when input arrives ticker by
    #: ticker, because each ticker spreads ~160 rows across 20 years, so the
    #: whole 9.4M-row table would be resident before any single year hit its
    #: limit.
    MAX_BUFFERED_ROWS = 500_000

    def __init__(
        self,
        name: str,
        *,
        validate: bool = True,
        replace: bool = True,
        max_buffered_rows: int | None = None,
    ):
        self.name = name
        self.schema = SCHEMAS[name]
        self.validate = validate
        self.max_buffered_rows = max_buffered_rows or self.MAX_BUFFERED_ROWS
        self.rows_written = 0
        self.flushes = 0
        self._buffers: dict[int, list[pd.DataFrame]] = {}
        self._buffered_rows: dict[int, int] = {}
        self._parts: dict[int, int] = {}
        self._touched: set[int] = set()
        if replace:
            drop_table(name)

    @property
    def buffered_rows(self) -> int:
        return sum(self._buffered_rows.values())

    def add(self, df: pd.DataFrame) -> None:
        """Buffer a frame, flushing every year once the global cap is reached."""
        self.add_prepared(self.prepare(df))

    def prepare(self, df: pd.DataFrame | None) -> pd.DataFrame | None:
        """The coerced frame :meth:`add` would buffer, or ``None`` for no rows."""
        if df is None or len(df) == 0:
            return None
        return coerce(df, self.name)

    def add_prepared(self, out: pd.DataFrame | None) -> None:
        """Buffer a frame already passed through :meth:`prepare`."""
        if out is None:
            return
        part_col = self.schema.partition_by
        for year, chunk in out.groupby(part_col, sort=True):
            year = int(year)
            self._buffers.setdefault(year, []).append(chunk)
            self._buffered_rows[year] = self._buffered_rows.get(year, 0) + len(chunk)
        if self.buffered_rows >= self.max_buffered_rows:
            self.flush()

    def flush(self) -> None:
        """Write every buffered year out as the next part file."""
        for year in sorted(list(self._buffers)):
            self._flush_year(year)
        self.flushes += 1

    def _flush_year(self, year: int) -> None:
        chunks = self._buffers.pop(year, None)
        self._buffered_rows.pop(year, None)
        if not chunks:
            return
        frame = pd.concat(chunks, ignore_index=True)
        frame = frame.sort_values(list(self.schema.primary_key), kind="stable")
        if self.validate:
            # Key uniqueness cannot be asserted per part file: the same key can
            # legitimately arrive in two batches when two source payloads
            # overlap. `finalize(dedupe=True)` resolves it across the whole
            # table once every batch has been written.
            assert_schema(frame, self.name, check_keys=False)
        part = self._parts.get(year, 0)
        write_partition(frame, self.name, year, part)
        self._parts[year] = part + 1
        self._touched.add(year)
        self.rows_written += len(frame)

    def close(self) -> None:
        self.flush()

    def finalize(
        self, *, dedupe: bool = True, bucket_count: int = FINALIZE_BUCKET_COUNT
    ) -> int:
        """Compact each year into one part file, optionally deduplicating.

        A streamed build cannot enforce primary-key uniqueness as it goes: two
        source payloads can legitimately carry the same contract (entry-date and
        calendar pulls overlap on a trade date), and they may land in different
        batches. Uniqueness is therefore a whole-table property, resolved here
        in a second pass — one year at a time.

        The first occurrence wins, and batches are fed in sorted source order,
        so which row survives is a function of the source set rather than of
        scheduling. Returns the number of rows removed.

        Dedup runs through :func:`_dedupe_and_write`, which range-partitions
        rows by primary key into ``bucket_count`` temp files, deduplicates
        each bucket on its own, and streams the result straight into the
        output file one bucket at a time -- when pyarrow is available. On
        that path, the largest frame this ever builds, in either phase, is
        one bucket's rows — never the whole raw year, and never the whole
        deduplicated year either. Without pyarrow, :func:`_dedupe_and_write`
        falls back to one plain concat of the whole year, same as before.

        This also compacts the numbered part files a streamed write leaves
        behind, which makes later reads cheaper and the content hash stable
        against changes in flush timing.

        Failure semantics: the new partition is written and atomically
        replaces part 0 (`write_partition`'s existing tmp-file + `os.replace`)
        BEFORE any stale numbered part file is removed. A crash at any point
        up to and including that replace leaves every pre-finalize part file
        exactly as it was. A crash after the replace but before every stale
        part is removed leaves the new, fully-deduplicated part-0 plus
        whichever stale numbered parts have not yet been deleted; a read in
        that window sees part-0's rows PLUS the stale parts' not-yet-removed
        (pre-dedupe) rows, which can repeat primary keys part-0 already
        holds — the same cross-part duplication finalize exists to resolve,
        not a new failure mode. No crash loses data, and running finalize
        again resolves any such leftover duplication the same way it always
        does.
        """
        if not isinstance(bucket_count, int) or isinstance(bucket_count, bool) or bucket_count < 1:
            raise ValueError(f"bucket_count must be a positive int, got {bucket_count!r}")
        self.flush()
        removed = 0
        for year in sorted(self._touched):
            part_dir = paths.curated_partition(self.name, year)
            parts = _partition_files(part_dir)
            if not parts:
                continue
            key_cols = list(self.schema.primary_key)
            if dedupe and key_cols:
                year_removed = _dedupe_and_write(parts, key_cols, bucket_count, self.name, year)
            else:
                frame = pd.concat([_read_part(p, None) for p in parts], ignore_index=True)
                frame.sort_values(key_cols, kind="stable", inplace=True)
                write_partition(frame, self.name, year, 0)
                year_removed = 0
            removed += year_removed
            for stale in parts:
                if stale.name != _part_name(0):
                    stale.unlink(missing_ok=True)
            self._parts[year] = 1
        self.rows_written -= removed
        return removed

    def __enter__(self) -> "PartitionedWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def years(self) -> list[int]:
        return sorted(self._touched)


def write_table(
    df: pd.DataFrame,
    name: str,
    *,
    validate: bool = True,
    replace: bool = True,
) -> dict[int, Path]:
    """Coerce, validate, and write ``df`` as year partitions.

    ``replace=True`` drops partitions that the new frame does not cover, so a
    rebuild cannot leave a stale year behind. That is the difference between a
    table that is *rebuilt* and one that is merely *appended to*.
    """
    schema = SCHEMAS[name]
    out = coerce(df, name)
    if validate:
        assert_schema(out, name)

    if schema.partition_by is None:
        raise ValueError(f"{name} is not partitioned")
    part_col = schema.partition_by
    if part_col not in out.columns:
        raise ValueError(f"{name}: partition column {part_col!r} absent")

    if replace:
        # Drop first rather than diffing: a previous streamed write may have
        # left several part files in a year this frame now covers with one, and
        # a stale part-0001 would silently double-count rows on the next read.
        drop_table(name)

    written: dict[int, Path] = {}
    if len(out):
        # Sorting by primary key makes partition bytes a function of content
        # only — not of the order rows happened to arrive in.
        out = out.sort_values(list(schema.primary_key), kind="stable")
        for year, chunk in out.groupby(part_col, sort=True):
            written[int(year)] = write_partition(chunk, name, int(year))
    return written


def drop_table(name: str) -> None:
    shutil.rmtree(paths.assert_writable(paths.curated_table(name)), ignore_errors=True)


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------


def table_years(name: str) -> list[int]:
    root = paths.curated_table(name)
    if not root.exists():
        return []
    years = []
    for child in root.iterdir():
        if child.is_dir() and child.name.startswith("year="):
            try:
                years.append(int(child.name.split("=", 1)[1]))
            except ValueError:
                continue
    return sorted(years)


def _read_part(path: Path, columns: Sequence[str] | None) -> pd.DataFrame:
    """Read one partition, tolerating a partition older than the schema.

    A column added to a Tier-2 schema does not exist in partitions written
    before it — the option_chains liquidity fields (volume, open interest, bid
    and ask size) are the first case, present only from the 2026-09 pull. A
    reader asking for one used to get ``ArrowInvalid`` from every earlier
    partition, which would have forced a full rebuild of a 15M-row table to
    read a column that is NaN there anyway. Missing columns are filled with
    NaN instead, so "this partition predates the field" reads the same as
    "this row has no value" — which is exactly what it means.
    """
    wanted = list(columns) if columns else None
    if HAVE_PARQUET:
        if wanted is None:
            return pd.read_parquet(path)
        import pyarrow.parquet as pq

        available = set(pq.ParquetFile(path).schema.names)
        present = [c for c in wanted if c in available]
        frame = pd.read_parquet(path, columns=present)
    else:
        with gzip.open(path, "rt") as fh:
            frame = pd.read_csv(fh)
        if wanted is None:
            return frame
        present = [c for c in wanted if c in frame.columns]
        frame = frame[present]
    for missing in [c for c in wanted if c not in frame.columns]:
        frame[missing] = np.nan
    return frame[wanted]


def iter_table(
    name: str,
    *,
    years: Iterable[int] | None = None,
    columns: Sequence[str] | None = None,
):
    """Yield ``(year, frame)`` one partition at a time.

    The way to touch a table too big to hold in memory. Consumers that only
    need a rolling window or a per-year aggregate should use this rather than
    :func:`read_table`.
    """
    if name not in SCHEMAS:
        raise KeyError(f"unknown table {name!r}; known: {sorted(SCHEMAS)}")
    wanted = sorted(set(years)) if years is not None else table_years(name)
    for year in wanted:
        parts = _partition_files(paths.curated_partition(name, year))
        if not parts:
            continue
        frame = pd.concat([_read_part(p, columns) for p in parts], ignore_index=True)
        yield year, frame


def read_table(
    name: str,
    *,
    years: Iterable[int] | None = None,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Read a Tier-2 table, optionally restricted to years and columns."""
    frames = [frame for _, frame in iter_table(name, years=years, columns=columns)]
    if not frames:
        base = empty_frame(name)
        return base[list(columns)] if columns else base
    out = pd.concat(frames, ignore_index=True)
    # Drop the per-year partitions before coerce() copies the result: holding
    # them too put three copies of daily_market live at once, past the
    # nightly's 8 GB cap.
    frames.clear()
    return coerce(out, name, only=list(columns) if columns else None)


# --------------------------------------------------------------------------
# stats / manifest input
# --------------------------------------------------------------------------


@dataclass
class TableStats:
    name: str
    rows: int
    years: list[int]
    files: int
    bytes: int
    content_hash: str
    fmt: str

    def as_dict(self) -> dict:
        return {
            "table": self.name,
            "rows": self.rows,
            "years": f"{min(self.years)}–{max(self.years)}" if self.years else "—",
            "partitions": len(self.years),
            "files": self.files,
            "bytes": self.bytes,
            "content_hash": self.content_hash,
            "format": self.fmt,
        }


def table_stats(name: str) -> TableStats:
    """Row counts, coverage, and a content hash for the manifest.

    The content hash is over the *partition file digests*, not over a
    concatenated frame: it is cheap on a six-million-row table and it changes
    if and only if some partition's bytes changed.
    """
    years = table_years(name)
    digests: list[str] = []
    total_bytes = 0
    rows = 0
    files = 0
    for year in years:
        for path in _partition_files(paths.curated_partition(name, year)):
            files += 1
            total_bytes += path.stat().st_size
            digests.append(f"{year}/{path.name}:{file_sha256(path)}")
            if HAVE_PARQUET:
                import pyarrow.parquet as pq

                rows += pq.ParquetFile(path).metadata.num_rows
            else:
                with gzip.open(path, "rt") as fh:
                    rows += max(0, sum(1 for _ in fh) - 1)
    content_hash = hashlib.sha256("|".join(digests).encode()).hexdigest()
    return TableStats(
        name=name,
        rows=rows,
        years=years,
        files=files,
        bytes=total_bytes,
        content_hash=content_hash,
        fmt=table_format(),
    )
