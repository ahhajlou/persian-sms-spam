use crate::regex_util::replace_all;
use crate::rules::url_rules;

/// Replace links (any scheme, `www.` and bare domains) with ` URLTOKEN `.
pub fn mask_urls(text: &str) -> String {
    mask_urls_counted(text, &mut 0)
}

/// Replace every run of decimal digits with ` NUM_TOKEN `.
pub fn mask_numbers(text: &str) -> String {
    mask_numbers_counted(text, &mut 0)
}

pub(crate) fn mask_urls_counted(text: &str, limit_hits: &mut usize) -> String {
    let m = &url_rules().url;
    replace_all(&m.re, text, limit_hits, |_| Some(m.replacement.clone()))
}

pub(crate) fn mask_numbers_counted(text: &str, limit_hits: &mut usize) -> String {
    let m = &url_rules().number;
    replace_all(&m.re, text, limit_hits, |_| Some(m.replacement.clone()))
}
