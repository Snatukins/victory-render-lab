//! Read-through for the previous Wave32 cache. Only the sampled shaders
//! selected for split-wave translation bypass it.

const CURRENT_TRANSLATOR: &str = "377427d6d602a7d6";
const CURRENT_ENCODING: &str = "aacca372293bb5642a9243d788a95fc99f2a9f713cc184f4c12b5375e1877813";
const PREVIOUS_TRANSLATOR: &str = "75d05d3c8ef363ad";
const PREVIOUS_ENCODING: &str = "ba6a8e9337af76fff49434005f26f3598781d1085a9a07d4b98c360c0043d589";

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
            "434902ef3151c62208861585681f2b1b425874a3d3e0d7b9403432190cd081e4",
        ).unwrap();
        assert_eq!(previous_key(CURRENT_TRANSLATOR, encoding, 4096, true), Some(*expected.as_bytes()));
        assert_eq!(previous_key("other translator", encoding, 4096, true), None);
        assert_eq!(previous_key(CURRENT_TRANSLATOR, blake3::hash(b"other encoding"), 4096, true), None);
        assert_ne!(previous_key(CURRENT_TRANSLATOR, encoding, 4096, false), Some(*expected.as_bytes()));
    }
}
