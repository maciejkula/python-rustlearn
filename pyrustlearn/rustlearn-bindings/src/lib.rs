#![allow(non_snake_case)]
extern crate rustlearn;

use std::mem;

use rustlearn::prelude::*;
use rustlearn::multiclass;
use rustlearn::linear_models::sgdclassifier;


fn construct_array(input: *mut f32, rows: usize, cols: usize) -> Array {

    let len = rows * cols;
    
    let mut data = unsafe {
        Array::from(Vec::from_raw_parts(input, len, len))
    };

    data.reshape(rows, cols);

    data
}


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


#[no_mangle]
pub extern fn free_sgdclassifier(model_ptr: * mut multiclass::OneVsRestWrapper<sgdclassifier::SGDClassifier>) {
    let _ = unsafe {
        Box::from_raw(model_ptr)
    };
}

#[test]
fn it_works() {
}
