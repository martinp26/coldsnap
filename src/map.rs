// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/*!
Generate a `qemu-img map --output=json`-compatible hole map ("side-car") for a raw
image, for use with `upload --holes-from-sidecar`.

The map is built with `SEEK_DATA`/`SEEK_HOLE` and covers `[0, size)` with
alternating runs: holes are `data: false, zero: true`, everything else is
`data: true, zero: false`. As with `qemu-img map`, written zeroes are data.
*/

use crate::data_map::DataMap;
use serde::Serialize;
use serde_json::ser::Formatter;
use snafu::{OptionExt, ResultExt, Snafu};
use std::fs;
use std::io;
use std::path::Path;

/// One extent, with `qemu-img map`'s fields in its key order.
#[derive(Serialize)]
struct MapEntry {
    start: u64,
    length: u64,
    depth: u8,
    present: bool,
    zero: bool,
    data: bool,
    compressed: bool,
    offset: u64,
}

impl MapEntry {
    fn new(start: u64, length: u64, data: bool) -> Self {
        MapEntry {
            start,
            length,
            depth: 0,
            present: true,
            zero: !data,
            data,
            compressed: false,
            offset: start,
        }
    }
}

/// Expand sorted `[start, end)` data extents into alternating data/hole entries
/// covering `[0, size)`.
fn build_map_entries(data_extents: &[(u64, u64)], size: u64) -> Vec<MapEntry> {
    let mut entries = Vec::new();
    let mut pos: u64 = 0;
    for &(start, end) in data_extents {
        if start > pos {
            entries.push(MapEntry::new(pos, start - pos, false));
        }
        entries.push(MapEntry::new(start, end - start, true));
        pos = end;
    }
    if pos < size {
        entries.push(MapEntry::new(pos, size - pos, false));
    }
    entries
}

/// Formats the array with one indented extent per line.
struct SidecarFormatter;

impl Formatter for SidecarFormatter {
    fn begin_array<W: ?Sized + io::Write>(&mut self, w: &mut W) -> io::Result<()> {
        w.write_all(b"[")
    }
    fn begin_array_value<W: ?Sized + io::Write>(
        &mut self,
        w: &mut W,
        first: bool,
    ) -> io::Result<()> {
        if !first {
            w.write_all(b",")?;
        }
        w.write_all(b"\n  ")
    }
    fn end_array<W: ?Sized + io::Write>(&mut self, w: &mut W) -> io::Result<()> {
        w.write_all(b"\n]")
    }

    fn begin_object<W: ?Sized + io::Write>(&mut self, w: &mut W) -> io::Result<()> {
        w.write_all(b"{ ")
    }
    fn begin_object_key<W: ?Sized + io::Write>(
        &mut self,
        w: &mut W,
        first: bool,
    ) -> io::Result<()> {
        if first {
            Ok(())
        } else {
            w.write_all(b", ")
        }
    }
    fn begin_object_value<W: ?Sized + io::Write>(&mut self, w: &mut W) -> io::Result<()> {
        w.write_all(b": ")
    }
    fn end_object<W: ?Sized + io::Write>(&mut self, w: &mut W) -> io::Result<()> {
        w.write_all(b" }")
    }
}

/// Serialize the entries as a JSON array; an empty map is `[]\n`.
fn qemu_format(entries: &[MapEntry]) -> String {
    if entries.is_empty() {
        return "[]\n".to_string();
    }
    let mut buf = Vec::new();
    let mut ser = serde_json::Serializer::with_formatter(&mut buf, SidecarFormatter);
    entries
        .serialize(&mut ser)
        .expect("serializing map entries to a Vec cannot fail");
    buf.push(b'\n');
    String::from_utf8(buf).expect("serde_json emits valid UTF-8")
}

/// Build the side-car JSON for the raw image at `path`. Fails if the filesystem
/// does not support hole detection.
pub fn generate_sidecar(path: &Path) -> Result<String, Error> {
    let size = fs::metadata(path).context(MetadataSnafu { path })?.len();

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
    fn qemu_format_json_structure_round_trips() {
        // data [0,512), hole [512,1024) -- the canonical PyAutoSnap reference.
        let out = qemu_format(&build_map_entries(&[(0, 512)], 1024));

        // Parse back and confirm the qemu field set round-trips.
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&out).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0]["start"], 0);
        assert_eq!(parsed[0]["length"], 512);
        assert_eq!(parsed[0]["data"], true);
        assert_eq!(parsed[0]["zero"], false);
        assert_eq!(parsed[0]["offset"], 0);
        assert_eq!(parsed[0]["depth"], 0);
        assert_eq!(parsed[0]["present"], true);
        assert_eq!(parsed[0]["compressed"], false);
        assert_eq!(parsed[1]["start"], 512);
        assert_eq!(parsed[1]["data"], false);
        assert_eq!(parsed[1]["zero"], true);
        assert_eq!(parsed[1]["offset"], 512);
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
