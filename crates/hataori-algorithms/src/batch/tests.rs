use super::*;
struct Number(u64);
impl WireValue for Number {
    const SCHEMA_ID: u64 = 91;
    fn encode(self) -> Result<Segments, ActionError> {
        self.0.encode()
    }
    fn decode(s: Segments) -> Result<Self, ActionError> {
        Ok(Self(u64::decode(s)?))
    }
}
impl Action for Number {
    const ID: u128 = 91;
    type Output = u64;
}
#[test]
fn codec_rejects_trailing_header_and_wrong_lengths() {
    let value = BatchAction::<Number> {
        start: 0,
        mode: LocalMode::Sequential,
        items: vec![Number(3)],
    };
    let encoded = value.encode().unwrap();
    let mut trailing = encoded.clone();
    trailing[0].push(0);
    assert!(BatchAction::<Number>::decode(trailing).is_err());
    let mut wrong = encoded;
    *wrong.last_mut().unwrap() = vec![0];
    assert!(BatchAction::<Number>::decode(wrong).is_err());
}
