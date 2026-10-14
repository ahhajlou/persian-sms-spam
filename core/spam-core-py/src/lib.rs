//! Python binding for spam-core: module `spam_core`.
//!
//! A thin layer; every rule and step lives in the spam-core crate.

use std::collections::BTreeMap;
use std::path::PathBuf;

use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// `Normalizer("rules")`, or `Normalizer("full", lexicons="path/to/dir")` with hazm's word lists.
#[pyclass(module = "spam_core")]
struct Normalizer {
    inner: engine::Normalizer,
}

#[pymethods]
impl Normalizer {
    #[new]
    #[pyo3(signature = (mode="rules", lexicons=None))]
    fn new(mode: &str, lexicons: Option<PathBuf>) -> PyResult<Self> {
        let inner = match (mode, lexicons) {
            ("rules", _) => engine::Normalizer::Rules,
            ("full", Some(dir)) => {
                engine::Normalizer::Full(engine::Lexicons::from_dir(&dir).map_err(|e| {
                    PyIOError::new_err(format!(
                        "word lists in {}: {e} (run `make lexicons`)",
                        dir.display()
                    ))
                })?)
            }
            ("full", None) => {
                return Err(PyValueError::new_err(
                    "the full normalizer needs lexicons=<dir>",
                ))
            }
            (other, _) => {
                return Err(PyValueError::new_err(format!(
                    "normalizer must be 'rules' or 'full', not {other:?}"
                )))
            }
        };
        Ok(Normalizer { inner })
    }

    #[getter]
    fn mode(&self) -> &'static str {
        self.inner.mode()
    }

    fn normalize(&self, text: &str) -> String {
        self.inner.normalize(text)
    }
}

/// -> (text, {tag: count}, rules skipped at the backtracking limit)
#[pyfunction]
fn anonymize(text: &str) -> (String, BTreeMap<String, usize>, usize) {
    let a = engine::anonymize(text);
    (a.text, a.counts, a.limit_hits)
}

#[pyfunction]
fn mask_urls(text: &str) -> String {
    engine::mask_urls(text)
}

#[pyfunction]
fn mask_numbers(text: &str) -> String {
    engine::mask_numbers(text)
}

/// Output after each step: anonymized, urls, numbers, normalized, counts, limit_hits.
#[pyfunction]
fn stages<'py>(
    py: Python<'py>,
    text: &str,
    normalizer: &Normalizer,
) -> PyResult<Bound<'py, PyDict>> {
    let s = engine::stages(text, &normalizer.inner);
    let d = PyDict::new(py);
    d.set_item("anonymized", s.anonymized)?;
    d.set_item("urls", s.urls)?;
    d.set_item("numbers", s.numbers)?;
    d.set_item("normalized", s.normalized)?;
    d.set_item("counts", s.counts)?;
    d.set_item("limit_hits", s.limit_hits)?;
    Ok(d)
}

#[pyfunction]
fn preprocess(text: &str, normalizer: &Normalizer) -> String {
    engine::preprocess(text, &normalizer.inner)
}

/// Lowercase + char_wb n-grams (2-4) of already preprocessed text.
#[pyfunction]
fn ngrams(text: &str) -> Vec<String> {
    engine::ngrams(text)
}

#[pyfunction]
fn features(text: &str, normalizer: &Normalizer) -> Vec<String> {
    engine::features(text, &normalizer.inner)
}

#[pyfunction]
fn card_ok(s: &str) -> bool {
    engine::card_ok(s)
}

#[pyfunction]
fn national_id_ok(s: &str) -> bool {
    engine::national_id_ok(s)
}

#[pymodule]
fn spam_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("PIPELINE_VERSION", engine::PIPELINE_VERSION)?;
    m.add("MAX_CHARS", engine::MAX_CHARS)?;
    m.add_class::<Normalizer>()?;
    m.add_function(wrap_pyfunction!(anonymize, m)?)?;
    m.add_function(wrap_pyfunction!(mask_urls, m)?)?;
    m.add_function(wrap_pyfunction!(mask_numbers, m)?)?;
    m.add_function(wrap_pyfunction!(stages, m)?)?;
    m.add_function(wrap_pyfunction!(preprocess, m)?)?;
    m.add_function(wrap_pyfunction!(ngrams, m)?)?;
    m.add_function(wrap_pyfunction!(features, m)?)?;
    m.add_function(wrap_pyfunction!(card_ok, m)?)?;
    m.add_function(wrap_pyfunction!(national_id_ok, m)?)?;
    Ok(())
}
