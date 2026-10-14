//! Calls the exported functions through a real JVM (started on the host), the way the app
//! would. `-Xcheck:jni` makes the JVM verify every JNI call. Needs a JDK: JAVA_HOME or `java`
//! on PATH; without one the tests print SKIPPED unless SPAM_CORE_REQUIRE_JVM=1.
//!
//! Cross-compiling for Android is not tested here (Phase 3b, in NoSpam).

use std::sync::OnceLock;

use jni::objects::{JObject, JObjectArray, JString};
use jni::{AttachGuard, InitArgsBuilder, JNIVersion, JavaVM};
use spam_core::Normalizer;
use spam_core_jni::*;

fn jvm() -> Option<&'static JavaVM> {
    static JVM: OnceLock<Option<JavaVM>> = OnceLock::new();
    JVM.get_or_init(|| {
        let args = InitArgsBuilder::new()
            .version(JNIVersion::V8)
            .option("-Xcheck:jni")
            .build()
            .unwrap();
        match JavaVM::new(args) {
            Ok(vm) => Some(vm),
            Err(e) => {
                assert!(
                    std::env::var_os("SPAM_CORE_REQUIRE_JVM").is_none(),
                    "no JVM for the JNI tests ({e}); install a JDK or set JAVA_HOME"
                );
                eprintln!("SKIPPED the JNI tests: no JVM ({e})");
                None
            }
        }
    })
    .as_ref()
}

/// Run `f` with an attached JNIEnv, or skip when there is no JVM.
fn with_env(f: impl FnOnce(&mut AttachGuard)) {
    if let Some(vm) = jvm() {
        f(&mut vm.attach_current_thread().unwrap());
    }
}

fn clone_env<'a>(env: &mut AttachGuard<'a>) -> jni::JNIEnv<'a> {
    // SAFETY: the clone is used only while `env` is alive, on this thread.
    unsafe { env.unsafe_clone() }
}

fn strings(env: &mut AttachGuard, array: jni::sys::jobjectArray) -> Vec<String> {
    assert!(!array.is_null(), "got null");
    let array = unsafe { JObjectArray::from_raw(array) };
    let len = env.get_array_length(&array).unwrap();
    (0..len)
        .map(|i| {
            let s = env.get_object_array_element(&array, i).unwrap();
            String::from(env.get_string(&JString::from(s)).unwrap())
        })
        .collect()
}

fn ngrams_of(env: &mut AttachGuard, text: &str) -> Vec<String> {
    let jtext = env.new_string(text).unwrap();
    let result =
        Java_com_nospam_nospam_core_ml_NativeCore_ngrams(clone_env(env), JObject::null(), jtext);
    strings(env, result)
}

fn stages_of(env: &mut AttachGuard, text: &str) -> Vec<String> {
    let jtext = env.new_string(text).unwrap();
    let result =
        Java_com_nospam_nospam_core_ml_NativeCore_stages(clone_env(env), JObject::null(), jtext);
    strings(env, result)
}

const MESSAGES: &[&str] = &[
    "",
    "سلام دنیا",
    "کد تایید شما: ۱۲۳۴۵۶ برای ورود به https://bank.ir/x",
    "کارت 6037-9975-1234-5678 مانده:50,051,391",
    "emoji 😀👍🏽 and nul \u{0} and thin\u{2009}space",
    "می‌خواهم کتاب‌ها را ببینم... تخفیف 50%!!!",
];

#[test]
fn version_and_mode() {
    with_env(|env| {
        let v = Java_com_nospam_nospam_core_ml_NativeCore_pipelineVersion(
            clone_env(env),
            JObject::null(),
        );
        assert_eq!(v, spam_core::PIPELINE_VERSION as i32);
        let mode = Java_com_nospam_nospam_core_ml_NativeCore_normalizerMode(
            clone_env(env),
            JObject::null(),
        );
        assert!(!mode.is_null());
        let mode = String::from(env.get_string(&unsafe { JString::from_raw(mode) }).unwrap());
        assert_eq!(mode, "rules");
    });
}

#[test]
fn ngrams_equal_the_core() {
    with_env(|env| {
        for text in MESSAGES {
            assert_eq!(
                ngrams_of(env, text),
                spam_core::features(text, &Normalizer::Rules),
                "{text:?}"
            );
        }
    });
}

#[test]
fn stages_equal_the_core() {
    with_env(|env| {
        for text in MESSAGES {
            let s = spam_core::stages(text, &Normalizer::Rules);
            assert_eq!(
                stages_of(env, text),
                [s.anonymized, s.urls, s.numbers, s.normalized],
                "{text:?}"
            );
        }
    });
}

#[test]
fn many_ngrams_do_not_overflow_the_local_frame() {
    with_env(|env| {
        // 2,000 distinct-ish characters: thousands of n-grams in one array.
        let text: String = (0..400).map(|i| format!("w{i}x ")).collect();
        let got = ngrams_of(env, &text);
        assert!(got.len() > 3000);
        assert_eq!(got, spam_core::features(&text, &Normalizer::Rules));
    });
}

fn assert_null_text_throws(
    env: &mut AttachGuard,
    call: impl Fn(jni::JNIEnv, JString) -> jni::sys::jobjectArray,
) {
    let result = call(clone_env(env), unsafe {
        JString::from_raw(std::ptr::null_mut())
    });
    assert!(result.is_null());
    assert!(
        env.exception_check().unwrap(),
        "expected a pending exception"
    );
    let exception = env.exception_occurred().unwrap();
    env.exception_clear().unwrap();
    assert!(env
        .is_instance_of(&exception, "java/lang/IllegalArgumentException")
        .unwrap());
}

#[test]
fn null_text_throws_illegal_argument() {
    with_env(|env| {
        assert_null_text_throws(env, |e, t| {
            Java_com_nospam_nospam_core_ml_NativeCore_ngrams(e, JObject::null(), t)
        });
        assert_null_text_throws(env, |e, t| {
            Java_com_nospam_nospam_core_ml_NativeCore_stages(e, JObject::null(), t)
        });
    });
}
