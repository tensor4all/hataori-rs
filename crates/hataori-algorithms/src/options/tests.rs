use super::*;
#[test]
fn validates() {
    assert!(PmapOptions::default().validate().is_ok());
    let o = PmapOptions {
        batch_size: 0,
        ..PmapOptions::default()
    };
    assert!(o.validate().is_err());
}
