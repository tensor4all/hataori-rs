//! Structured-payload codecs for the framed MPI wire protocol.
//!
//! Framing and the frame header in [`crate::wire`] are independent of the
//! codec; a codec only turns a structured payload into bytes and back. Senders
//! and receivers must agree on one codec per exchange, so the codec is chosen at
//! compile time. `bincode` is the default and keeps the wire bytes unchanged.

use crate::map::truncate_message;
use crate::wire::WireError;
use serde::{de::DeserializeOwned, Serialize};

/// Encoding of a structured wire payload.
pub(crate) trait PayloadCodec {
    /// Encodes `value`, mapping any failure to [`WireError::Encode`].
    fn encode<T: Serialize>(value: &T) -> Result<Vec<u8>, WireError>;

    /// Decodes exactly one value from `bytes`; trailing bytes are an error.
    fn decode<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, WireError>;
}

/// The default codec: bincode 2 serde with fixed-size little-endian integers.
pub(crate) struct Bincode;

impl PayloadCodec for Bincode {
    fn encode<T: Serialize>(value: &T) -> Result<Vec<u8>, WireError> {
        bincode::serde::encode_to_vec(value, config()).map_err(|error| WireError::Encode {
            detail: truncate_message(error.to_string()),
        })
    }

    fn decode<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, WireError> {
        let (value, used) =
            bincode::serde::decode_from_slice(bytes, config()).map_err(|error| {
                WireError::Decode {
                    detail: truncate_message(error.to_string()),
                }
            })?;
        if used != bytes.len() {
            return Err(WireError::TrailingBytes {
                used,
                total: bytes.len(),
            });
        }
        Ok(value)
    }
}

fn config() -> bincode::config::Configuration<bincode::config::LittleEndian, bincode::config::Fixint>
{
    bincode::config::standard()
        .with_fixed_int_encoding()
        .with_little_endian()
}

/// Opt-in evaluation codec backed by `bitcode` (feature `bitcode`).
///
/// `bitcode`'s serde format is not self-describing and does not support every
/// serde type (for example `deserialize_any`), so it is not the wire default.
#[cfg(feature = "bitcode")]
#[allow(
    dead_code,
    reason = "the MPI path keeps bincode; tests and the measurement harness select this backend"
)]
pub(crate) struct Bitcode;

#[cfg(feature = "bitcode")]
impl PayloadCodec for Bitcode {
    fn encode<T: Serialize>(value: &T) -> Result<Vec<u8>, WireError> {
        bitcode::serialize(value).map_err(|error| WireError::Encode {
            detail: truncate_message(error.to_string()),
        })
    }

    fn decode<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, WireError> {
        // `bitcode::deserialize` decodes one value and rejects trailing bytes.
        bitcode::deserialize(bytes).map_err(|error| WireError::Decode {
            detail: truncate_message(error.to_string()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wire::{decode_payload_with, encode_payload, encode_payload_with};
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Serialize, Deserialize, PartialEq)]
    struct Pair {
        left: u16,
        right: String,
    }

    #[derive(Debug, Serialize, Deserialize, PartialEq)]
    struct Nested {
        pairs: Vec<Pair>,
        tag: Option<u32>,
        weights: Vec<f64>,
        flags: Vec<bool>,
    }

    fn pair() -> Pair {
        Pair {
            left: 7,
            right: "wire".into(),
        }
    }

    fn nested() -> Nested {
        Nested {
            pairs: vec![
                Pair {
                    left: 1,
                    right: "a".into(),
                },
                Pair {
                    left: 1 << 15,
                    right: String::new(),
                },
            ],
            tag: Some(9),
            weights: vec![0.5, -1.0, f64::MIN_POSITIVE],
            flags: vec![true, false],
        }
    }

    /// Round-trips a small struct, a nested value, empty values, and a large `Vec`
    /// through one codec and checks every value and the byte budget.
    fn assert_roundtrips<C: PayloadCodec>() {
        let bytes = encode_payload_with::<_, C>(&pair()).unwrap();
        assert_eq!(
            decode_payload_with::<Pair, C>(&bytes).unwrap(),
            pair(),
            "small struct did not round-trip"
        );

        let nested_value = nested();
        let bytes = encode_payload_with::<_, C>(&nested_value).unwrap();
        assert_eq!(
            decode_payload_with::<Nested, C>(&bytes).unwrap(),
            nested_value,
            "nested value did not round-trip"
        );

        let empty_nested = Nested {
            pairs: Vec::new(),
            tag: None,
            weights: Vec::new(),
            flags: Vec::new(),
        };
        let bytes = encode_payload_with::<_, C>(&empty_nested).unwrap();
        assert_eq!(
            decode_payload_with::<Nested, C>(&bytes).unwrap(),
            empty_nested,
            "empty nested value did not round-trip"
        );

        // A unit payload has no fields: it encodes to an empty buffer, and
        // decoding that empty buffer back is the whole round trip.
        let bytes = encode_payload_with::<_, C>(&()).unwrap();
        assert!(bytes.is_empty(), "the unit payload encoded to bytes");
        decode_payload_with::<(), C>(&bytes).unwrap();

        let empty_text = String::new();
        let bytes = encode_payload_with::<_, C>(&empty_text).unwrap();
        assert_eq!(
            decode_payload_with::<String, C>(&bytes).unwrap(),
            empty_text,
            "empty string did not round-trip"
        );

        let large: Vec<u64> = (0..10_000).map(|value| value * 3).collect();
        let bytes = encode_payload_with::<_, C>(&large).unwrap();
        assert_eq!(
            decode_payload_with::<Vec<u64>, C>(&bytes).unwrap(),
            large,
            "large vector did not round-trip"
        );
    }

    /// Every truncation, one trailing byte, and garbage must be rejected.
    fn assert_rejects_malformed<C: PayloadCodec>() {
        let bytes = encode_payload_with::<_, C>(&pair()).unwrap();
        for length in 0..bytes.len() {
            assert!(
                decode_payload_with::<Pair, C>(&bytes[..length]).is_err(),
                "a payload truncated to {length} bytes decoded"
            );
        }

        let mut padded = bytes.clone();
        padded.push(0);
        assert!(
            decode_payload_with::<Pair, C>(&padded).is_err(),
            "a payload with a trailing byte decoded"
        );

        assert!(
            decode_payload_with::<Pair, C>(&[0xff; 8]).is_err(),
            "garbage decoded"
        );
    }

    #[test]
    fn wire_default_is_the_bincode_codec_with_unchanged_bytes() {
        assert_eq!(
            encode_payload(&pair()).unwrap(),
            encode_payload_with::<_, Bincode>(&pair()).unwrap()
        );
        assert_eq!(encode_payload(&0x1234_u16).unwrap(), vec![0x34, 0x12]);
    }

    #[test]
    fn bincode_roundtrips_small_nested_empty_and_large_payloads() {
        assert_roundtrips::<Bincode>();
    }

    #[test]
    fn bincode_rejects_truncated_trailing_and_garbage_bytes() {
        assert_rejects_malformed::<Bincode>();
    }

    #[cfg(feature = "bitcode")]
    #[test]
    fn bitcode_roundtrips_small_nested_empty_and_large_payloads() {
        assert_roundtrips::<Bitcode>();
    }

    #[cfg(feature = "bitcode")]
    #[test]
    fn bitcode_rejects_truncated_trailing_and_garbage_bytes() {
        assert_rejects_malformed::<Bitcode>();
    }

    /// Measurement harness, not a regression test. Run one shape at a time on an
    /// idle machine:
    ///
    /// ```sh
    /// cargo test --release --no-default-features --features mpi,bitcode \
    ///     --lib payload_codec_measurement -- --ignored --nocapture
    /// ```
    #[cfg(feature = "bitcode")]
    #[test]
    #[ignore = "measurement: run explicitly with --release --ignored --nocapture"]
    fn payload_codec_measurement() {
        use std::hint::black_box;
        use std::time::{Duration, Instant};

        fn measure<C: PayloadCodec, T: Serialize + DeserializeOwned>(
            value: &T,
            rounds: u32,
        ) -> (usize, Duration, Duration) {
            let mut encode = Duration::MAX;
            let mut decode = Duration::MAX;
            let mut length = 0;
            for _ in 0..rounds {
                let start = Instant::now();
                let bytes = encode_payload_with::<T, C>(value).unwrap();
                encode = encode.min(start.elapsed());
                length = bytes.len();
                let start = Instant::now();
                let decoded = decode_payload_with::<T, C>(&bytes).unwrap();
                let elapsed = start.elapsed();
                black_box(&decoded);
                decode = decode.min(elapsed);
            }
            (length, encode, decode)
        }

        fn row<C: PayloadCodec, T: Serialize + DeserializeOwned>(
            value: &T,
            rounds: u32,
        ) -> (usize, f64, f64) {
            let (length, encode, decode) = measure::<C, T>(value, rounds);
            (
                length,
                encode.as_secs_f64() * 1e9,
                decode.as_secs_f64() * 1e9,
            )
        }

        println!("shape\tcodec\tbytes\tencode_ns\tdecode_ns");
        macro_rules! both {
            ($shape:literal, $value:expr, $rounds:expr) => {{
                let value = $value;
                let bincode = row::<Bincode, _>(&value, $rounds);
                let bitcode = row::<Bitcode, _>(&value, $rounds);
                println!(
                    "{}\tbincode\t{}\t{:.0}\t{:.0}",
                    $shape, bincode.0, bincode.1, bincode.2
                );
                println!(
                    "{}\tbitcode\t{}\t{:.0}\t{:.0}",
                    $shape, bitcode.0, bitcode.1, bitcode.2
                );
            }};
        }

        both!("struct_pair", pair(), 2_000);
        both!(
            "vec_pair_1000",
            (0..1_000)
                .map(|index| Pair {
                    left: index as u16,
                    right: format!("item-{index}"),
                })
                .collect::<Vec<_>>(),
            300
        );
        both!(
            "vec_u64_100k_small",
            (0..100_000u64).map(|value| value * 7).collect::<Vec<_>>(),
            50
        );
        both!(
            "vec_u64_100k_wide",
            (0..100_000u64)
                .map(|value| value.wrapping_mul(0x9E37_79B9_7F4A_7C15))
                .collect::<Vec<_>>(),
            50
        );
        both!(
            "vec_f64_1m",
            (0..1_000_000)
                .map(|value| value as f64 * 1.5)
                .collect::<Vec<_>>(),
            10
        );
    }
}
