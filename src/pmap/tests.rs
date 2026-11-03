#[test]
fn prefetch_is_opt_in() {
    assert!(!super::PmapOptions::default().prefetch);
}

#[test]
fn root_event_chooser_alternates_when_both_sources_are_ready() {
    let mut prefer_local = true;
    let mut events = Vec::new();
    for _ in 0..4 {
        let (event, next) = super::choose_root_event(true, true, prefer_local);
        events.push(event);
        prefer_local = next;
    }
    assert_eq!(
        events,
        [
            super::RootEvent::Local,
            super::RootEvent::Remote,
            super::RootEvent::Local,
            super::RootEvent::Remote
        ]
    );
}

#[test]
fn root_event_chooser_keeps_one_sided_progress() {
    let (local, turn) = super::choose_root_event(true, false, true);
    assert_eq!(local, super::RootEvent::Local);
    let (remote, _) = super::choose_root_event(false, true, turn);
    assert_eq!(remote, super::RootEvent::Remote);
}
