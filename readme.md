# python-rustlearn

[![Circle CI](https://circleci.com/gh/maciejkula/python-rustlearn.svg?style=svg)](https://circleci.com/gh/maciejkula/python-rustlearn)

A simple example of using a Rust library (in this case, [rustlearn](https://github.com/maciejkula/rustlearn)) from Python.

We'll we rustlearn to estimate a simple logistic regression model on the MNIST digits dataset --- but we'll do the data processing and model evaluation in Python.

## Installation
This example is set up as a Python package. You should be able to run it by:

1. Installing the Rust compiler: https://www.rust-lang.org/downloads.html
2. Installing `libcffi-dev` (on Ubuntu, `sudo apt-get install libcffi-dev`)
3. Cloning this repository, and
4. Running `pip install .` in the resulting directory.

You can verify that everything works by running `python setup.py test`.

## Setting up a C API in Rust

First, we need to create a new crate using `cargo new` and set it to compile to a shared object via `Cargo.toml`:

```
[dependencies]
rustlearn = "0.3.0"

[lib]
name = "rustlearn"
crate-type = ["dylib"]
```

Because we're only writing a simple wrapper, we can start adding code directly in [`lib.rs`](/pyrustlearn/rustlearn-bindings/src/lib.rs).

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

Once this is done, we can simply call `cargo build` to build the shared object we need.

## Using the API from Python
To write the Python bindings I'm going to use [`cffi`](https://cffi.readthedocs.org/en/latest/).

The first thing we need to do is to describe the C interface and open the `.so` file:

```python
def _build_bindings():

    ffi = cffi.FFI()
    ffi.cdef("""
    struct SGDModel;
    struct SGDModel *fit_sgdclassifier(float *X_ptr, unsigned long X_rows, unsigned long X_cols,
                                float *y_ptr, unsigned long y_rows, unsigned long y_cols);
    float* predict_sgdclassifier(float *X_ptr, unsigned long X_rows, unsigned long X_cols,
                                 struct SGDModel *model);
    void free_sgdclassifier(struct SGDModel *model);
    """)

    path  = os.path.join(os.path.dirname(__file__),
                         'rustlearn-bindings/target/release/librustlearn.so')
    lib = ffi.dlopen(path)

    return ffi, lib
```

The function definitions are the key part. We specify an opaque struct `SGDModel`, and the three functions we've exposed from the Rust code via the `#[no_mangle]` attribute. `cffi` parses and checks their syntax.

We can then open the library by calling `ffi.dlopen(...)`.

To make using the interface easier, we can define a couple of helper functions:

```python
def _as_float(array):
    """
    Cast a np.float32 array to a float*.
    """

    return ffi.cast('float*', array.ctypes.data)


def _as_usize(num):
    """
    Cast num to something like a rust usize.
    """

    return ffi.cast('unsigned long', num)


def _as_float_ndarray(ptr, size):
    """
    Turn a float* to a numpy array.
    """

    return np.core.multiarray.int_asbuffer(ptr, size * np.float32.itemsize)
```

For numpy arrays, `array.ctypes.data` is the memory address of the data buffer: to pass it as `float*`, we simply cast it. (Note that this only works for C-contiguous arrays.)

We can then define an sklearn-like class for our model:

```python
class SGDClassifier(object):

    def __init__(self):
        self.model = None

    def fit(self, X, y):

        self.model = lib.fit_sgdclassifier(_as_float(X),
                                           _as_usize(X.shape[0]),
                                           _as_usize(X.shape[1]),
                                           _as_float(y),
                                           _as_usize(len(y)),
                                           _as_usize(1))

    def predict(self, X):

        if self.model is None:
            raise Exception('Call fit before calling predict')

        predictions_ptr = lib.predict_sgdclassifier(
            _as_float(X), _as_usize(X.shape[0]), _as_usize(X.shape[1]),
            ffi.cast('struct SGDModel*', self.model)
        )

        predictions = np.frombuffer(ffi.buffer(predictions_ptr, len(X)
                                               * ffi.sizeof('float')),
                                    dtype=np.float32)

        return predictions

    def __del__(self):

        if self.model is not None:
            lib.free_sgdclassifier(self.model)
            self.model = None
```

In the `fit` function, we convert the inputs into pointers, call the Rust API, and get a model pointer back. To predict, we use the predict API function with the new input data and the model pointer. We get back a data pointer which we turn into a numpy array via a cffi buffer. Finally, we add a custom destructor to make sure we clean up the model data when we are finished.

You can see the whole file [here](/pyrustlearn/__init__.py).

## Putting it all together
To make sure that everything works, we'll [test it](/examples/mnist.py) on the MNIST digits dataset.

Let's define a couple of helper functions:

```python
import numpy as np

from sklearn.cross_validation import KFold
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score

from pyrustlearn import SGDClassifier


def _get_data():

    data = load_digits()

    X = data.data.astype(np.float32)
    y = data.target.astype(np.float32)

    return (X, y)
```

and the main model evaluation loop:

```python
def run_example():

    data, target = _get_data()

    n_folds = 5
    accuracy = 0.0

    for (train_idx, test_idx) in KFold(n=len(data), n_folds=n_folds, shuffle=True):

        train_X = data[train_idx]
        train_y = target[train_idx]

        test_X = data[test_idx]
        test_y = target[test_idx]

        model = SGDClassifier()
        model.fit(train_X, train_y)

        predictions = model.predict(test_X)

        accuracy += accuracy_score(predictions, test_y)

    return accuracy / n_folds
```

Running this should succeed, resulting in an accuracy of about 0.92.