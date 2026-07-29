//! XML general-reference resolution shared by the document and transcript
//! parsers.
//!
//! # Why this exists
//!
//! quick-xml 0.38 stopped folding entity references into [`Event::Text`] and
//! started emitting them as a separate [`Event::GeneralRef`]. A reader that
//! only matches `Text` still compiles and still produces plausible output, it
//! just silently drops every `&amp;`, `&lt;` and `&#39;` in the document. That
//! is the worst shape a regression can take in an ingestion pipeline: the text
//! reaches the index looking fine, and the loss only shows up as an answer that
//! quotes a mangled sentence.
//!
//! Three parsers read XML here (DOCX body, EPUB OPF metadata, YouTube
//! transcripts) and all three need the same resolution, so it lives once.
//!
//! [`Event::Text`]: quick_xml::events::Event::Text
//! [`Event::GeneralRef`]: quick_xml::events::Event::GeneralRef

use quick_xml::events::BytesRef;

/// Resolve one general reference to the text it stands for.
///
/// Handles numeric character references (`&#38;`, `&#x26;`) and the five
/// entities XML predefines. Anything else is a DTD-declared entity, which
/// neither DOCX, EPUB OPF nor YouTube transcripts use; `None` says so rather
/// than guessing, and the caller decides whether to warn.
#[must_use]
pub fn resolve_general_ref(reference: &BytesRef<'_>) -> Option<String> {
    // Numeric first: `resolve_char_ref` only answers for `&#…;` forms, so a
    // named entity falls through to the table below.
    if let Ok(Some(resolved)) = reference.resolve_char_ref() {
        return Some(resolved.to_string());
    }

    match reference.decode().ok()?.as_ref() {
        "amp" => Some("&".to_owned()),
        "lt" => Some("<".to_owned()),
        "gt" => Some(">".to_owned()),
        "quot" => Some("\"".to_owned()),
        "apos" => Some("'".to_owned()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use quick_xml::Reader;
    use quick_xml::events::Event;

    use super::*;

    /// Drive a real reader, because the point of this module is what the
    /// reader emits, not what a hand-built `BytesRef` contains.
    fn text_of(xml: &str) -> String {
        let mut reader = Reader::from_str(xml);
        let mut out = String::new();
        loop {
            match reader.read_event() {
                Ok(Event::Text(e)) => out.push_str(&e.decode().unwrap_or_default()),
                Ok(Event::GeneralRef(e)) => {
                    if let Some(resolved) = resolve_general_ref(&e) {
                        out.push_str(&resolved);
                    }
                }
                Ok(Event::Eof) => break,
                Err(_) => break,
                _ => {}
            }
        }
        out
    }

    #[test]
    fn the_five_predefined_entities_round_trip() {
        assert_eq!(
            text_of("<t>a &amp; b &lt; c &gt; d &quot;e&quot; &apos;f&apos;</t>"),
            "a & b < c > d \"e\" 'f'"
        );
    }

    #[test]
    fn numeric_character_references_resolve() {
        // `&#39;` is what YouTube transcripts use for apostrophes.
        assert_eq!(text_of("<t>it&#39;s &#x26; more</t>"), "it's & more");
    }

    #[test]
    fn text_without_entities_is_unchanged() {
        assert_eq!(text_of("<t>plain sentence</t>"), "plain sentence");
    }

    /// The regression this module exists to prevent: dropping the reference
    /// entirely would yield "a  b" and read as correct.
    #[test]
    fn an_entity_is_not_silently_dropped() {
        let out = text_of("<t>a &amp; b</t>");
        assert!(out.contains('&'), "entity vanished: {out:?}");
    }

    #[test]
    fn an_unknown_dtd_entity_resolves_to_nothing_rather_than_garbage() {
        let mut reader = Reader::from_str("<t>&custom;</t>");
        loop {
            match reader.read_event() {
                Ok(Event::GeneralRef(e)) => {
                    assert_eq!(resolve_general_ref(&e), None);
                    return;
                }
                Ok(Event::Eof) | Err(_) => panic!("no general reference was emitted"),
                _ => {}
            }
        }
    }
}
