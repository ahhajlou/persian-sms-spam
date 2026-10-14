//! JNI binding for spam-core, for the NoSpam app.
//!
//! Hand-written on purpose: four functions do not justify UniFFI and the JNA it would add to
//! the APK. The Kotlin side is
//!
//! ```kotlin
//! package com.nospam.nospam.core.ml
//!
//! object NativeCore {
//!     init { System.loadLibrary("spam_core_jni") }
//!     @JvmStatic external fun pipelineVersion(): Int
//!     @JvmStatic external fun normalizerMode(): String
//!     @JvmStatic external fun ngrams(text: String): Array<String>
//!     @JvmStatic external fun stages(text: String): Array<String>
//! }
//! ```
//!
//! The Rust function names encode that package and class: renaming `NativeCore` means
//! renaming every `Java_com_nospam_nospam_core_ml_NativeCore_*` function here. (Without
//! `@JvmStatic` the methods are instance methods; the functions below work for both.)
//!
//! The app always uses the `rules` normalizer: the `full` one needs hazm's 14 MB word lists,
//! and no model on the phone is trained with it. A model whose `pipeline.normalizer` differs
//! from `normalizerMode()` must be rejected by the caller.
//!
//! Errors never unwind into the JVM: a null or invalid string throws
//! `IllegalArgumentException`, a panic throws `IllegalStateException`, and the function
//! returns null (or 0).

use std::panic::{catch_unwind, AssertUnwindSafe};

use jni::objects::{JObject, JObjectArray, JString};
use jni::sys::{jint, jobjectArray, jstring};
use jni::JNIEnv;
use spam_core::Normalizer;

static NORMALIZER: Normalizer = Normalizer::Rules;

/// Run `f`, turning an error or a panic into a Java exception and `default`.
fn guarded<'l, T>(
    env: &mut JNIEnv<'l>,
    default: T,
    f: impl FnOnce(&mut JNIEnv<'l>) -> Result<T, String>,
) -> T {
    let (class, message) = match catch_unwind(AssertUnwindSafe(|| f(&mut *env))) {
        Ok(Ok(value)) => return value,
        Ok(Err(message)) => ("java/lang/IllegalArgumentException", message),
        Err(_) => (
            "java/lang/IllegalStateException",
            "spam-core panicked".to_string(),
        ),
    };
    // Keep an exception that is already pending (it is the more precise one).
    if !env.exception_check().unwrap_or(true) {
        let _ = env.throw_new(class, message);
    }
    default
}

fn read_string(env: &mut JNIEnv, text: &JString) -> Result<String, String> {
    if text.as_raw().is_null() {
        return Err("text is null".to_string());
    }
    env.get_string(text)
        .map(String::from)
        .map_err(|e| format!("cannot read text: {e}"))
}

fn string_array<'l>(env: &mut JNIEnv<'l>, items: &[String]) -> Result<JObjectArray<'l>, String> {
    let fail = |e: jni::errors::Error| format!("cannot build the result: {e}");
    let class = env.find_class("java/lang/String").map_err(fail)?;
    let array = env
        .new_object_array(items.len() as jint, class, JObject::null())
        .map_err(fail)?;
    for (i, item) in items.iter().enumerate() {
        let s = env.new_string(item).map_err(fail)?;
        env.set_object_array_element(&array, i as jint, &s)
            .map_err(fail)?;
        env.delete_local_ref(s).map_err(fail)?; // thousands of n-grams: don't fill the local frame
    }
    Ok(array)
}

/// `PIPELINE_VERSION` of the compiled core.
#[no_mangle]
pub extern "system" fn Java_com_nospam_nospam_core_ml_NativeCore_pipelineVersion(
    _env: JNIEnv,
    _this: JObject,
) -> jint {
    spam_core::PIPELINE_VERSION as jint
}

/// "rules" (the only mode built into the app).
#[no_mangle]
pub extern "system" fn Java_com_nospam_nospam_core_ml_NativeCore_normalizerMode(
    mut env: JNIEnv,
    _this: JObject,
) -> jstring {
    guarded(&mut env, std::ptr::null_mut(), |env| {
        env.new_string(NORMALIZER.mode())
            .map(|s| s.into_raw())
            .map_err(|e| e.to_string())
    })
}

/// Raw SMS -> the TF-IDF features: preprocess, then lowercase and char_wb 2-4-grams.
#[no_mangle]
pub extern "system" fn Java_com_nospam_nospam_core_ml_NativeCore_ngrams(
    mut env: JNIEnv,
    _this: JObject,
    text: JString,
) -> jobjectArray {
    guarded(&mut env, std::ptr::null_mut(), |env| {
        let text = read_string(env, &text)?;
        Ok(string_array(env, &spam_core::features(&text, &NORMALIZER))?.into_raw())
    })
}

/// Raw SMS -> the output of each step: [anonymized, urls masked, numbers masked, normalized].
#[no_mangle]
pub extern "system" fn Java_com_nospam_nospam_core_ml_NativeCore_stages(
    mut env: JNIEnv,
    _this: JObject,
    text: JString,
) -> jobjectArray {
    guarded(&mut env, std::ptr::null_mut(), |env| {
        let text = read_string(env, &text)?;
        let s = spam_core::stages(&text, &NORMALIZER);
        Ok(string_array(env, &[s.anonymized, s.urls, s.numbers, s.normalized])?.into_raw())
    })
}
