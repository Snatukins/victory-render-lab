//! Read-through for the wave-first lowering change. The two affected shader
//! containers bypass this cache on wave32-only devices. All other outputs
//! retain their previous translation and encoding. Any later translator or
//! codec change disables this compatibility entry until independently audited.

const CURRENT_TRANSLATOR: &str = "540581f1b478e1b8";
const CURRENT_ENCODING: &str = "ba6a8e9337af76fff49434005f26f3598781d1085a9a07d4b98c360c0043d589";
const PREVIOUS_TRANSLATOR: &str = "1c77b3918465ce3d";
const PREVIOUS_ENCODING: &str = "52e4c3763d93770a75dd9ad4035706fa4560406f0a6ea1dd022513b8f64eb16a";

pub(super) fn previous_key(
    translator: &str,
    encoding: blake3::Hash,
    limit: u32,
    wave32_only: bool,
) -> Option<[u8; 32]> {
    if translator != CURRENT_TRANSLATOR || encoding.to_hex().as_str() != CURRENT_ENCODING {
        return None;
    }
    let previous_encoding = blake3::Hash::from_hex(PREVIOUS_ENCODING).expect("valid fingerprint");
    Some(super::key_with_encoding_hash(PREVIOUS_TRANSLATOR, previous_encoding, limit, wave32_only))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compatible_cache_requires_the_audited_revision() {
        let encoding = blake3::Hash::from_hex(CURRENT_ENCODING).unwrap();
        let expected = blake3::Hash::from_hex(
            "5a66cb05239f00ce9a1f6bfd9d62051a957209b3414cbfeebdcc0a99e8f67987",
        ).unwrap();
        assert_eq!(previous_key(CURRENT_TRANSLATOR, encoding, 4096, true), Some(*expected.as_bytes()));
        assert_eq!(previous_key("other translator", encoding, 4096, true), None);
        assert_eq!(previous_key(CURRENT_TRANSLATOR, blake3::hash(b"other encoding"), 4096, true), None);
        assert_ne!(previous_key(CURRENT_TRANSLATOR, encoding, 4096, false), Some(*expected.as_bytes()));
    }
}
