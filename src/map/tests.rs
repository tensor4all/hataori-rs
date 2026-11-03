use super::{map, MapError};

#[test]
fn empty_input_succeeds() {
    let result = map(Vec::<i32>::new(), |_| Ok::<_, &str>(0));
    assert_eq!(result.unwrap(), Vec::<i32>::new());
}

#[test]
fn successful_inputs_are_ordered_and_called_once() {
    let calls = std::cell::Cell::new(0);
    let result = map(vec![3, 1, 2], |item| {
        calls.set(calls.get() + 1);
        Ok::<_, &str>(item * 2)
    });

    assert_eq!(result.unwrap(), vec![6, 2, 4]);
    assert_eq!(calls.get(), 3);
}

#[test]
fn first_error_stops_later_callbacks() {
    let calls = std::cell::Cell::new(0);
    let result = map(vec![10, 20, 30], |item| {
        calls.set(calls.get() + 1);
        if item == 20 {
            Err::<i32, _>("bad item")
        } else {
            Ok(item)
        }
    });

    let error = result.unwrap_err();
    assert_eq!(error.index(), 1);
    assert_eq!(error.message(), "bad item");
    assert_eq!(calls.get(), 2);
}

#[test]
fn ascii_error_is_capped() {
    let result = map(vec![()], |_| Err::<(), _>("x".repeat(5000)));
    assert_eq!(result.unwrap_err().message().len(), 4096);
}

#[test]
fn multibyte_error_is_truncated_at_utf8_boundary() {
    let message = format!("{}é", "a".repeat(4095));
    let error = map(vec![()], |_| Err::<(), _>(message.clone())).unwrap_err();

    assert_eq!(error.message().len(), 4095);
    assert!(error.message().is_char_boundary(error.message().len()));
}

#[test]
fn serial_map_accepts_borrowed_non_static_values_and_state() {
    use std::{cell::RefCell, rc::Rc};

    let input_text = String::from("borrowed input");
    let error_text = String::from("borrowed error");
    let calls = Rc::new(RefCell::new(0));
    let captured = Rc::clone(&calls);
    let input = Rc::new(input_text.as_str());
    let output: Result<Vec<Rc<&str>>, MapError> = map(vec![input], |item| {
        *captured.borrow_mut() += 1;
        Ok::<_, Rc<&str>>(item)
    });

    assert_eq!(output.unwrap().as_slice(), &[Rc::new("borrowed input")]);
    assert_eq!(*calls.borrow(), 1);

    let error = map(vec![Rc::new(input_text.as_str())], |_| {
        Err::<(), _>(Rc::new(error_text.as_str()))
    })
    .unwrap_err();
    assert_eq!(error.message(), "borrowed error");
}
