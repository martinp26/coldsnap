// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/*!
Generate a `qemu-img map --output=json`-compatible JSON side-car for a raw image.

This is the producer counterpart to [`crate::data_map::DataMap::from_sidecar`]
(the `--holes-from-sidecar` consumer). Given a raw, uncompressed image it walks
the file's real filesystem holes with `SEEK_DATA`/`SEEK_HOLE` (reusing
[`DataMap::from_path`]) and emits the allocation map as JSON, byte-for-byte
identical to `qemu-img map --output=json` for a single-layer raw image.

Each array element is one contiguous `[start, start + length)` run of the image;
the runs are contiguous and sorted and together cover `[0, size)`:

```text
hole  ->  present: true, zero: true,  data: false
data  ->  present: true, zero: false, data: true
```

`present` is always true for a raw file: there is no backing chain, so the one
layer determines every range. `offset` equals `start` and `compressed` is always
false, matching qemu-img's raw output.

An allocated-zero region -- real zero bytes written to disk -- is reported as
`data: true, zero: false`, NOT `data: true, zero: true`. This is a metadata-only
map built from `SEEK_DATA`/`SEEK_HOLE`, which never reads block contents, so
allocated-but-zero is indistinguishable from allocated data. `qemu-img map`
behaves the same. A side-car-driven upload therefore selects exactly the blocks a
live `--detect-holes` scan would.
*/

use crate::data_map::DataMap;
use snafu::{OptionExt, ResultExt, Snafu};
use std::fs;
use std::path::Path;

/// One extent of the emitted map.
struct MapEntry {
    start: u64,
    length: u64,
    data: bool,
}

/// Render a bool the way qemu-img's JSON writer does: bare `true`/`false`.
fn b(value: bool) -> &'static str {
    if value {
        "true"
    } else {
        "false"
    }
}

/// Turn the sorted data extents and total size into the full alternating map.
///
/// `data_extents` is the `[(start, end), ...]` list of data-bearing runs (holes
/// excluded), exactly as produced by [`DataMap`]. The gaps between them -- and
/// any trailing gap up to `size` -- become hole entries so the result covers
/// `[0, size)` with no gaps, like `qemu-img map`.
fn build_map_entries(data_extents: &[(u64, u64)], size: u64) -> Vec<MapEntry> {
    let mut entries = Vec::new();
    let mut pos: u64 = 0;
    for &(start, end) in data_extents {
        if start > pos {
            entries.push(MapEntry {
                start: pos,
                length: start - pos,
                data: false,
            });
        }
        entries.push(MapEntry {
            start,
            length: end - start,
            data: true,
        });
        pos = end;
    }
    if pos < size {
        entries.push(MapEntry {
            start: pos,
            length: size - pos,
            data: false,
        });
    }
    entries
}

/// Serialize entries byte-for-byte like `qemu-img map --output=json`.
///
/// Reproduces qemu-img.c `dump_map_entry(OFORMAT_JSON)`: the array opens with
/// `[` glued to the first object, objects are joined by `,\n`, each object is
/// `{ ` + space-separated `"key": value` pairs + `}`, `offset` is always
/// emitted (equal to `start` for a raw file), and the stream is terminated by
/// `]\n`. An empty map is `[]\n`.
fn qemu_format(entries: &[MapEntry]) -> String {
    if entries.is_empty() {
        return "[]\n".to_string();
    }
    let mut out = String::new();
    for (i, e) in entries.iter().enumerate() {
        out.push_str(if i == 0 { "[" } else { ",\n" });
        out.push_str(&format!(
            "{{ \"start\": {}, \"length\": {}, \"depth\": 0, \"present\": true, \
             \"zero\": {}, \"data\": {}, \"compressed\": false, \"offset\": {}}}",
            e.start,
            e.length,
            b(!e.data),
            b(e.data),
            e.start,
        ));
    }
    out.push_str("]\n");
    out
}

/// Build the qemu-img-map JSON text for the raw image at `path`.
///
/// Walks the file's `SEEK_DATA`/`SEEK_HOLE` geometry (via [`DataMap::from_path`],
/// which also flushes and drops the page cache first so the map reflects the
/// on-disk layout rather than warm-cache state). Errors out when the filesystem
/// does not support hole detection: a whole-file "everything is data" map is
/// worse than none, since it would defeat the point of the side-car.
pub fn generate_sidecar(path: &Path) -> Result<String, Error> {
    let size = fs::metadata(path)
        .context(MetadataSnafu { path })?
        .len();

    let map = DataMap::from_path(path, size)
        .context(ScanSnafu { path })?
        .context(UnsupportedSnafu { path })?;

    let entries = build_map_entries(map.extents(), size);
    Ok(qemu_format(&entries))
}

/// Errors from generating a side-car.
#[derive(Debug, Snafu)]
#[snafu(visibility(pub(crate)))]
pub enum Error {
    #[snafu(display("failed to read metadata for '{}': {}", path.display(), source))]
    Metadata {
        path: std::path::PathBuf,
        source: std::io::Error,
    },

    #[snafu(display("failed to scan '{}' for holes: {}", path.display(), source))]
    Scan {
        path: std::path::PathBuf,
        source: std::io::Error,
    },

    #[snafu(display(
        "filesystem holding '{}' does not support SEEK_HOLE hole detection; \
         cannot generate a side-car",
        path.display()
    ))]
    Unsupported { path: std::path::PathBuf },
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn qemu_format_is_byte_exact() {
        // data [0,512), hole [512,1024) -- the canonical PyAutoSnap reference.
        let entries = build_map_entries(&[(0, 512)], 1024);
        let expected = "[{ \"start\": 0, \"length\": 512, \"depth\": 0, \"present\": true, \
             \"zero\": false, \"data\": true, \"compressed\": false, \"offset\": 0},\n\
             { \"start\": 512, \"length\": 512, \"depth\": 0, \"present\": true, \
             \"zero\": true, \"data\": false, \"compressed\": false, \"offset\": 512}]\n";
        assert_eq!(qemu_format(&entries), expected);
    }

    #[test]
    fn build_map_entries_alternates_and_covers_whole_size() {
        // data [0,512), hole [512,1024), data [1024,1536), trailing hole [1536,2048)
        let entries = build_map_entries(&[(0, 512), (1024, 1536)], 2048);
        let shape: Vec<(u64, u64, bool)> = entries
            .iter()
            .map(|e| (e.start, e.length, e.data))
            .collect();
        assert_eq!(
            shape,
            vec![
                (0, 512, true),
                (512, 512, false),
                (1024, 512, true),
                (1536, 512, false),
            ]
        );
        // Contiguous coverage of [0, size) with no gaps.
        assert_eq!(entries.first().unwrap().start, 0);
        let last = entries.last().unwrap();
        assert_eq!(last.start + last.length, 2048);
        for w in entries.windows(2) {
            assert_eq!(w[0].start + w[0].length, w[1].start);
        }
    }

    #[test]
    fn qemu_format_empty_is_bracket_pair() {
        assert_eq!(qemu_format(&build_map_entries(&[], 0)), "[]\n");
    }
}
