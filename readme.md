# python-rustlearn

[![Circle CI](https://circleci.com/gh/maciejkula/python-rustlearn.svg?style=svg)](https://circleci.com/gh/maciejkula/python-rustlearn)

A simple example of using a Rust library (in this case, [rustlearn](https://github.com/maciejkula/rustlearn)) from Python.

We'll we rustlearn to estimate a simple logistic regression model on the MNIST digits dataset --- but we'll do the data processing and model evaluation in Python.

## Setting up a C API in Rust

First, we need to create a new crate using `cargo new` and set it to compile to a shared object via `Cargo.toml`:

```
[dependencies]
rustlearn = "0.3.0"

[lib]
name = "rustlearn"
crate-type = ["dylib"]
```

Because we're only writing a simple wrapper, we can start adding code directly in [`lib.rs](/pyrustlearn/rustlearn-bindings/src/lib.rs).

The first thing we construct is a helper function which can create a rustlearn `Array` from a raw C pointer:

```rust
fn construct_array(input: *mut f32, rows: usize, cols: usize) -> Array {

    let len = rows * cols;
    
    let mut data = unsafe {
        Array::from(Vec::from_raw_parts(input, len, len))
    };

    data.reshape(rows, cols);

    data
}
```

This allows us to share numpy arrays with Rust without making any copies. The small `unsafe` section turns a raw pointer into a Rust vector.

With the use of this helper we can write a public function to fit the model. We use the `#[no_mangle]` directive to enable us to call it by name.

```rust
#[no_mangle]
pub extern fn fit_sgdclassifier(X_ptr: *mut f32, X_rows: usize, X_cols: usize,
                                y_ptr: *mut f32, y_rows: usize, y_cols: usize)
                                -> *const multiclass::OneVsRestWrapper<sgdclassifier::SGDClassifier>
{

    let X = construct_array(X_ptr, X_rows, X_cols);
    let y = construct_array(y_ptr, y_rows, y_cols);

    let mut model = sgdclassifier::Hyperparameters::new(X.cols())
        .learning_rate(0.05)
        .l2_penalty(0.000001)
        .l1_penalty(0.000001)
        .one_vs_rest();

    for _ in 0..5 {
        model.fit(&X, &y).unwrap();
    }

    let boxed_model = Box::new(model);

    // We don't want to free the numpy arrays
    mem::forget(X);
    mem::forget(y);

    Box::into_raw(boxed_model)
}
```

The function takes pointers for the data and label arrays and returns a pointer for a model object. Because our Python code will only interact with the model object via API calls, we don't really need to worry about what this is on the Python side.

Having obtained the input data, we set up a logistic regression model and run 5 epochs of training, in a way that is very similar to what we would do in Python.

Once this is done we need to take care of a couple of tricky things.

1. We wrap the model object into a `Box`. That's because Rust allocates things on the stack by default: `model` is on the stack, and will be destroyed once it goes out of scope. But we want to keep it around and use it later --- to make that possible, we move it to the heap by boxing it.
2. We tell Rust we don't want it to call destructors on the input arrays. If it did, the data would get freed on the Python side as well and break our Python code. We do this by calling `mem::forget`.
3. Finally, we turn the box containing the model into a raw pointer. This does two things: allows us to return a straightforward pointer _and_ tells Rust not to call the destructor on the boxed value when it goes out of scope.

The drill is similar for getting predictions. We get the input data and the model pointer, transform it to a boxede model (which is unsafe because the pointer could be invalid), and compute the predictions. We then tell Rust to forget about freeing the data we want to keep around, and return a pointer to the predictions data.

```rust
#[no_mangle]
pub extern fn predict_sgdclassifier(X_ptr: *mut f32, X_rows: usize, X_cols: usize,
                                    model_ptr: * mut multiclass::OneVsRestWrapper<sgdclassifier::SGDClassifier>)
                                    -> *const f32
{
    let model = unsafe {
        Box::from_raw(model_ptr)
    };

    let X = construct_array(X_ptr, X_rows, X_cols);

    let predictions = model.predict(&X).unwrap();
    let predictions_ptr = predictions.data()[..].as_ptr();

    // We don't want to free the numpy arrays...
    mem::forget(X);
    // or the memory we allocated for predictions...
    mem::forget(predictions);
    // or the model, we might need it later
    mem::forget(model);

    predictions_ptr
}
```

Finally, we need a way of freeing the model object. This is very simple: we turn the model pointer into a box and let Rust free it in the normal way:

```rust
#[no_mangle]
pub extern fn free_sgdclassifier(model_ptr: * mut multiclass::OneVsRestWrapper<sgdclassifier::SGDClassifier>) {
    let _ = unsafe {
        Box::from_raw(model_ptr)
    };
}
```