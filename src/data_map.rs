// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/*!
Filesystem hole detection for sparse uploads.

Builds a map of the byte ranges in a file that actually contain data, using the
`SEEK_DATA`/`SEEK_HOLE` extensions to `lseek(2)`. Callers can then ask whether a
given block-sized range lies entirely within a hole and skip it without ever
reading it from disk.

Unlike a content scan (read every block, test for all-zeroes), this reflects the
*filesystem's* view of which regions were never written. It therefore does not
omit blocks that were explicitly written with zeroes -- those are real data
extents to `SEEK_DATA`. A hole always reads back as zeroes, so the set of holes
is a strict subset of the blocks a content scan would treat as zero; skipping a
hole never changes the reconstructed image on an unencrypted snapshot.
*/

use log::debug;
use nix::errno::Errno;
use nix::fcntl::{posix_fadvise, PosixFadviseAdvice};
use nix::unistd::{lseek, Whence};
use serde::Deserialize;
use snafu::{ensure, ResultExt, Snafu};
use std::convert::TryFrom;
use std::fs::File;
use std::path::{Path, PathBuf};

/// A sorted, non-overlapping set of `[start, end)` byte ranges that contain
/// data. Any offset not covered by an extent is a hole (reads back as zeroes).
#[derive(Debug, Clone)]
pub(crate) struct DataMap {
    extents: Vec<(u64, u64)>,
}

impl DataMap {
    /// Treat the entire file as data. Used when hole detection is disabled or
    /// unsupported by the underlying filesystem (or for block devices), so that
    /// no block is ever skipped as a hole.
    pub(crate) fn full(size: u64) -> Self {
        DataMap {
            extents: if size == 0 {
                Vec::new()
            } else {
                vec![(0, size)]
            },
        }
    }

    /// Build a data map from a `qemu-img map --output=json` side-car.
    ///
    /// Each array element is one contiguous `[start, start+length)` run;
    /// elements with `data: true` are data extents (uploaded), everything else
    /// is a hole (omitted). The SEEK-based generator records written-zero
    /// regions as `data: true`, so they are uploaded -- identical block
    /// selection to a live `--detect-holes` scan. The side-car must cover
    /// exactly `expected_size` bytes (the end of its last extent); a mismatch
    /// means it describes a different image and is a hard error rather than a
    /// silently corrupt upload.
    pub(crate) fn from_sidecar(path: &Path, expected_size: u64) -> Result<Self, SidecarError> {
        let bytes = std::fs::read(path).context(ReadSnafu { path })?;
        let raw: Vec<SidecarExtent> =
            serde_json::from_slice(&bytes).context(ParseSnafu { path })?;

        let mut covered: u64 = 0;
        let mut extents: Vec<(u64, u64)> = Vec::new();
        for e in &raw {
            let end = e.start.saturating_add(e.length);
            covered = covered.max(end);
            if e.data {
                extents.push((e.start, end));
            }
        }
        ensure!(
            covered == expected_size,
            SizeMismatchSnafu {
                path,
                covered,
                expected: expected_size,
            }
        );

        // Sort and coalesce touching/overlapping data extents so the map stays
        // sorted and non-overlapping for `range_has_data`'s binary search. A
        // well-formed qemu-img map is already disjoint and sorted; this is
        // defensive and never changes which bytes count as data.
        extents.sort_unstable();
        let mut merged: Vec<(u64, u64)> = Vec::with_capacity(extents.len());
        for (start, end) in extents {
            match merged.last_mut() {
                Some(last) if start <= last.1 => {
                    if end > last.1 {
                        last.1 = end;
                    }
                }
                _ => merged.push((start, end)),
            }
        }
        Ok(DataMap { extents: merged })
    }

    /// Build a data map for `path` by walking its `SEEK_DATA`/`SEEK_HOLE`
    /// geometry.
    ///
    /// Returns `Ok(None)` when the filesystem does not support hole detection,
    /// signalling the caller to fall back to [`DataMap::full`]. A fully sparse
    /// file yields an empty extent list.
    pub(crate) fn from_path(path: &Path, size: u64) -> std::io::Result<Option<Self>> {
        let file = File::open(path)?;
        Ok(Self::from_file(&file, size))
    }

    /// Build a data map from an already-open file. Returns `None` if the
    /// filesystem does not implement `SEEK_DATA`/`SEEK_HOLE`, or on any
    /// unexpected `lseek` error, so the caller can safely fall back to
    /// uploading every block.
    pub(crate) fn from_file(file: &File, size: u64) -> Option<Self> {
        let size_i = i64::try_from(size).ok()?;

        // Make the SEEK_DATA/SEEK_HOLE walk reflect the on-disk map, not
        // page-cache state. On iomap filesystems (XFS, ext4) a zero
        // folio cached over an *unwritten* (preallocated-but-never-written)
        // extent is reported as DATA when the cache is warm and as a HOLE
        // when cold, so a warm cache both inflates the map and makes it
        // non-deterministic across runs. Flush dirty pages first (since
        // POSIX_FADV_DONTNEED only drops clean pages), then evict, so the
        // walk sees the stable cold map. Both steps are best-effort:
        // filesystems may ignore them, and eviction is only needed here at
        // the scan -- later block reads may re-warm the cache without
        // affecting this already-built map. If flushing does not work
        // perfectly, we would upload slightly more holes as full zero runs,
        // but would never miss real data, so correctness is preserved.
        if let Err(e) = file.sync_all() {
            debug!("fsync before hole scan failed ({e}); proceeding without flush");
        }
        if let Err(e) = posix_fadvise(file, 0, 0, PosixFadviseAdvice::POSIX_FADV_DONTNEED) {
            debug!("posix_fadvise(DONTNEED) before hole scan failed ({e}); hole map may reflect warm page cache");
        }

        let mut extents: Vec<(u64, u64)> = Vec::new();
        let mut pos: i64 = 0;

        while pos < size_i {
            // Find the start of the next data region at or after `pos`.
            let data_start = match lseek(file, pos, Whence::SeekData) {
                Ok(off) => off,
                // No data between `pos` and EOF: the remainder is a hole.
                Err(Errno::ENXIO) => break,
                // ENOTSUP/EINVAL => no SEEK_HOLE support; anything else is
                // unexpected. In all cases fall back to a full read.
                Err(e) => {
                    debug!("lseek(SEEK_DATA) at {pos} failed ({e}); disabling hole detection");
                    return None;
                }
            };
            if data_start >= size_i {
                break;
            }
            // Find the end of that data region: the next hole, or EOF.
            let data_end = match lseek(file, data_start, Whence::SeekHole) {
                Ok(off) => off.min(size_i),
                Err(e) => {
                    debug!(
                        "lseek(SEEK_HOLE) at {data_start} failed ({e}); disabling hole detection"
                    );
                    return None;
                }
            };
            // Defensive: SEEK_HOLE must advance past SEEK_DATA. If it does not,
            // stop rather than loop forever.
            if data_end <= data_start {
                debug!("lseek(SEEK_HOLE) did not advance ({data_start} -> {data_end}); disabling hole detection");
                return None;
            }
            extents.push((data_start as u64, data_end as u64));
            pos = data_end;
        }

        Some(DataMap { extents })
    }

    /// Returns `true` if `[offset, offset + len)` overlaps any data extent. A
    /// range with no overlap lies entirely within a hole and can be omitted
    /// without reading it. A zero-length range never has data.
    pub(crate) fn range_has_data(&self, offset: u64, len: u64) -> bool {
        if len == 0 {
            return false;
        }
        let end = offset.saturating_add(len);
        // First extent whose end is strictly greater than `offset`; extents are
        // sorted and non-overlapping, so this is the only candidate that could
        // overlap `[offset, end)`.
        let idx = self.extents.partition_point(|&(_, e)| e <= offset);
        match self.extents.get(idx) {
            Some(&(start, _)) => start < end,
            None => false,
        }
    }

    /// Number of data extents in the map (for logging/diagnostics).
    pub(crate) fn extent_count(&self) -> usize {
        self.extents.len()
    }

    /// Total number of bytes that contain data across all extents.
    #[cfg(test)]
    pub(crate) fn data_bytes(&self) -> u64 {
        self.extents.iter().map(|&(s, e)| e - s).sum()
    }
}

/// One extent object from a `qemu-img map --output=json` side-car. Only the
/// fields that drive block selection are read; `depth`, `present`, `zero`,
/// `compressed`, and `offset` are accepted and ignored. `data` defaults to
/// `false` when absent.
#[derive(Debug, Deserialize)]
struct SidecarExtent {
    start: u64,
    length: u64,
    #[serde(default)]
    data: bool,
}

/// Errors from loading a hole map out of a side-car file.
#[derive(Debug, Snafu)]
#[snafu(visibility(pub(crate)))]
pub(crate) enum SidecarError {
    #[snafu(display("failed to read side-car '{}': {}", path.display(), source))]
    Read {
        path: PathBuf,
        source: std::io::Error,
    },

    #[snafu(display("failed to parse side-car '{}' as qemu-img-map JSON: {}", path.display(), source))]
    Parse {
        path: PathBuf,
        source: serde_json::Error,
    },

    #[snafu(display(
        "side-car '{}' covers {covered} bytes but the image is {expected} bytes; \
         refusing to upload with a mismatched map",
        path.display()
    ))]
    SizeMismatch {
        path: PathBuf,
        covered: u64,
        expected: u64,
    },
}

#[cfg(test)]
mod test {
    use super::*;
    use std::io::{Seek, SeekFrom, Write};

    fn write_sidecar(json: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().expect("tempfile");
        f.write_all(json.as_bytes()).expect("write");
        f.flush().expect("flush");
        f
    }

    const SIDECAR: &str = r##"[{ "start": 0, "length": 512, "depth": 0, "present": true, "zero": false, "data": true, "compressed": false, "offset": 0},
{ "start": 512, "length": 512, "depth": 0, "present": true, "zero": true, "data": false, "compressed": false, "offset": 512}]
"##;

    #[test]
    fn sidecar_selects_data_extents() {
        let sc = write_sidecar(SIDECAR);
        let map = DataMap::from_sidecar(sc.path(), 1024).expect("load");
        assert_eq!(map.extent_count(), 1);
        assert_eq!(map.data_bytes(), 512);
        assert!(map.range_has_data(0, 512)); // data run
        assert!(!map.range_has_data(512, 512)); // hole
    }

    #[test]
    fn sidecar_written_zeroes_are_data() {
        // A fully-written all-zero image is a single data extent
        // (data:true, zero:false), so nothing is omitted.
        let sc = write_sidecar(
            r##"[{ "start": 0, "length": 1024, "depth": 0, "present": true, "zero": false, "data": true, "compressed": false, "offset": 0}]"##,
        );
        let map = DataMap::from_sidecar(sc.path(), 1024).expect("load");
        assert!(map.range_has_data(0, 1024));
        assert_eq!(map.data_bytes(), 1024);
    }

    #[test]
    fn sidecar_fully_sparse_is_all_holes() {
        let sc = write_sidecar(
            r##"[{ "start": 0, "length": 4096, "depth": 0, "present": true, "zero": true, "data": false, "compressed": false, "offset": 0}]"##,
        );
        let map = DataMap::from_sidecar(sc.path(), 4096).expect("load");
        assert_eq!(map.extent_count(), 0);
        assert!(!map.range_has_data(0, 4096));
    }

    #[test]
    fn sidecar_size_mismatch_is_error() {
        let sc = write_sidecar(SIDECAR); // covers 1024
        assert!(matches!(
            DataMap::from_sidecar(sc.path(), 999).unwrap_err(),
            SidecarError::SizeMismatch {
                covered: 1024,
                expected: 999,
                ..
            }
        ));
    }

    #[test]
    fn sidecar_empty_map_covers_zero() {
        let sc = write_sidecar("[]\n");
        assert!(DataMap::from_sidecar(sc.path(), 0).is_ok());
        assert!(matches!(
            DataMap::from_sidecar(sc.path(), 4096).unwrap_err(),
            SidecarError::SizeMismatch {
                covered: 0,
                expected: 4096,
                ..
            }
        ));
    }

    #[test]
    fn full_map_covers_everything() {
        let m = DataMap::full(1024);
        assert!(m.range_has_data(0, 512));
        assert!(m.range_has_data(512, 512));
        // A range starting at EOF has no data.
        assert!(!m.range_has_data(1024, 512));
        assert_eq!(m.data_bytes(), 1024);
        assert_eq!(m.extent_count(), 1);
    }

    #[test]
    fn empty_file_full_map_is_empty() {
        let m = DataMap::full(0);
        assert!(!m.range_has_data(0, 512));
        assert_eq!(m.data_bytes(), 0);
        assert_eq!(m.extent_count(), 0);
    }

    #[test]
    fn range_has_data_bisect() {
        // Data at [0,100) and [1000,1100); holes everywhere else.
        let m = DataMap {
            extents: vec![(0, 100), (1000, 1100)],
        };
        assert!(m.range_has_data(0, 10)); // inside first extent
        assert!(m.range_has_data(50, 100)); // straddles end of first extent
        assert!(!m.range_has_data(100, 900)); // entirely in the middle hole
        assert!(!m.range_has_data(200, 700)); // entirely in the middle hole
        assert!(m.range_has_data(900, 200)); // straddles start of second extent
        assert!(m.range_has_data(1050, 10)); // inside second extent
        assert!(!m.range_has_data(1100, 500)); // trailing hole
        assert!(!m.range_has_data(50, 0)); // zero-length range
    }

    // Best-effort end-to-end check against the real filesystem. If the backing
    // filesystem does not support SEEK_HOLE (from_file returns None) or does not
    // actually punch a hole for the sparse region, we only assert the invariants
    // that must hold regardless.
    #[test]
    fn sparse_file_geometry() {
        let block = 512 * 1024usize; // mirror the EBS block size
        let mut tf = tempfile::tempfile().expect("create tempfile");

        // Layout: [data block 0][hole block 1][data block 2], total 3 blocks.
        let payload = vec![0xABu8; block];
        tf.write_all(&payload).expect("write block 0");
        tf.seek(SeekFrom::Start((2 * block) as u64))
            .expect("seek to block 2");
        tf.write_all(&payload).expect("write block 2");
        tf.flush().expect("flush");

        let size = (3 * block) as u64;
        // Ensure the reported length matches even if the fs kept it dense.
        tf.set_len(size).expect("set_len");

        let map = match DataMap::from_file(&tf, size) {
            Some(m) => m,
            None => return, // filesystem lacks SEEK_HOLE support; nothing to assert
        };

        // Block 0 and block 2 must always be data.
        assert!(
            map.range_has_data(0, block as u64),
            "block 0 should be data"
        );
        assert!(
            map.range_has_data((2 * block) as u64, block as u64),
            "block 2 should be data"
        );

        // If the fs punched the hole, block 1 must read as a hole and the total
        // data must be <= 2 blocks. If it did not (dense allocation), the map is
        // a single full extent -- still internally consistent.
        if map.extent_count() > 1 {
            assert!(
                !map.range_has_data(block as u64, block as u64),
                "block 1 should be a hole"
            );
            assert!(map.data_bytes() <= (2 * block) as u64);
        }
    }
}
