//! Untrusted counts and size limits must fail cleanly instead of
//! pre-allocating from the input or reading past the declared size.
#![cfg(all(feature = "std", target_pointer_width = "64"))]
// Most structs here exist only to drive the reader into an error, so their
// fields are never read back.
#![allow(dead_code)]

use std::collections::{HashMap, HashSet};

use deku::prelude::*;

/// More elements than could ever fit in memory: `count * size_of::<T>()`
/// overflows, so pre-allocating from it panics with "capacity overflow".
const HUGE: u64 = u64::MAX;

fn huge_count_input() -> Vec<u8> {
    let mut input = HUGE.to_be_bytes().to_vec();
    input.extend_from_slice(&[0xAA, 0xBB, 0xCC, 0xDD]);
    input
}

fn assert_incomplete<T: core::fmt::Debug>(r: Result<T, DekuError>) {
    match r {
        Err(DekuError::Incomplete(_)) => {}
        other => panic!("expected Incomplete, got {other:?}"),
    }
}

#[test]
fn vec_count_does_not_preallocate() {
    #[derive(Debug, DekuRead)]
    #[deku(endian = "big")]
    struct S {
        n: u64,
        #[deku(count = "*n")]
        items: Vec<u32>,
    }
    assert_incomplete(S::from_bytes((&huge_count_input(), 0)));
}

#[test]
fn boxed_slice_count_does_not_preallocate() {
    #[derive(Debug, DekuRead)]
    #[deku(endian = "big")]
    struct S {
        n: u64,
        #[deku(count = "*n")]
        items: Box<[u32]>,
    }
    assert_incomplete(S::from_bytes((&huge_count_input(), 0)));
}

#[test]
fn hashmap_count_does_not_preallocate() {
    #[derive(Debug, DekuRead)]
    #[deku(endian = "big")]
    struct S {
        n: u64,
        #[deku(count = "*n")]
        items: HashMap<u16, u16>,
    }
    assert_incomplete(S::from_bytes((&huge_count_input(), 0)));
}

#[test]
fn hashset_count_does_not_preallocate() {
    #[derive(Debug, DekuRead)]
    #[deku(endian = "big")]
    struct S {
        n: u64,
        #[deku(count = "*n")]
        items: HashSet<u32>,
    }
    assert_incomplete(S::from_bytes((&huge_count_input(), 0)));
}

/// 1 byte when `kind == 0`, 3 bytes when `kind == 1`.
#[derive(Debug, PartialEq, Eq, Hash, DekuRead)]
#[deku(endian = "endian", ctx = "endian: deku::ctx::Endian")]
struct Elem {
    kind: u8,
    #[deku(cond = "*kind == 1")]
    extra: Option<u16>,
}

/// A 2-byte budget whose first element is 3 bytes, followed by more bytes
/// that must not be consumed as elements.
const OVERRUN: [u8; 7] = [0x02, 0x01, 0xAA, 0xBB, 0x00, 0x00, 0x00];

fn assert_overrun<T: core::fmt::Debug>(r: Result<T, DekuError>) {
    match r {
        Err(DekuError::Parse(msg)) if msg.contains("element overran size limit") => {}
        other => panic!("expected overrun error, got {other:?}"),
    }
}

#[test]
fn vec_bytes_read_rejects_overrun() {
    #[derive(Debug, DekuRead)]
    #[deku(endian = "big")]
    struct S {
        len: u8,
        #[deku(bytes_read = "*len")]
        items: Vec<Elem>,
    }
    assert_overrun(S::from_bytes((&OVERRUN, 0)));
}

#[test]
fn hashset_bytes_read_rejects_overrun() {
    #[derive(Debug, DekuRead)]
    #[deku(endian = "big")]
    struct S {
        len: u8,
        #[deku(bytes_read = "*len")]
        items: HashSet<Elem>,
    }
    assert_overrun(S::from_bytes((&OVERRUN, 0)));
}

#[test]
fn hashmap_bytes_read_rejects_overrun() {
    #[derive(Debug, DekuRead)]
    #[deku(endian = "big")]
    struct S {
        len: u8,
        #[deku(bytes_read = "*len")]
        items: HashMap<u8, Elem>,
    }
    // Entry = 1-byte key + Elem. Budget 3; first entry is 1 + 3 = 4 bytes.
    let input = [0x03, 0x07, 0x01, 0xAA, 0xBB, 0x00, 0x00, 0x00];
    assert_overrun(S::from_bytes((&input, 0)));
}

#[test]
fn exact_budget_still_parses() {
    #[derive(Debug, DekuRead)]
    #[deku(endian = "big")]
    struct S {
        len: u8,
        #[deku(bytes_read = "*len")]
        items: Vec<Elem>,
        tail: u8,
    }
    let mut input = OVERRUN;
    input[0] = 0x03;
    let ((rest, _), s) = S::from_bytes((&input, 0)).unwrap();
    assert_eq!(s.items.len(), 1);
    assert_eq!(s.tail, 0x00);
    assert_eq!(rest.len(), 2);
}
