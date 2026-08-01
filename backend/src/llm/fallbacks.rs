//! What the assistant says when there is nothing to ground an answer on
//! (US-020, PRD edge cases 1 and 2).
//!
//! Not prompts: sentences. No model is asked to phrase them, which is the whole
//! point. "There is nothing to search" and "what came back does not support an
//! answer" are facts about the turn, and asking a model to say them is what used
//! to produce a fluent, confident, ungrounded reply instead.
//!
//! # One table, two readers
//!
//! [`FALLBACK_TEXTS`] holds every sentence every locale can produce, and it is
//! what the grounded evaluator matches to classify a turn as an abstention
//! ([`reads_as_abstention`](crate::services::rag::eval::grounding::reads_as_abstention)).
//!
//! The coupling used to run the other way. The evaluator matched a French
//! substring, so an English answer had to open with the French sentence for the
//! turn to be scored as an abstention: the metric was dictating the product
//! copy, and every user read a language that was not theirs. Adding a locale now
//! means adding a row here, and a test refuses a row the selector can produce
//! but the table does not list.

/// The two sentences one locale returns.
struct Fallbacks {
    no_sources: &'static str,
    insufficient_evidence: &'static str,
}

const FALLBACKS_FR: Fallbacks = Fallbacks {
    no_sources: "Aucune source n'est disponible dans ce notebook.",
    insufficient_evidence: "Les sources disponibles ne permettent pas de répondre avec suffisamment de confiance.",
};

const FALLBACKS_EN: Fallbacks = Fallbacks {
    no_sources: "No source is available in this notebook.",
    insufficient_evidence: "The available sources do not support a confident answer.",
};

const FALLBACKS_DE: Fallbacks = Fallbacks {
    no_sources: "In diesem Notizbuch ist keine Quelle verfügbar.",
    insufficient_evidence: "Die verfügbaren Quellen erlauben keine hinreichend sichere Antwort.",
};

const FALLBACKS_ES: Fallbacks = Fallbacks {
    no_sources: "No hay ninguna fuente disponible en este cuaderno.",
    insufficient_evidence: "Las fuentes disponibles no permiten responder con suficiente confianza.",
};

/// Every documented fallback sentence, all locales, both kinds.
pub const FALLBACK_TEXTS: &[&str] = &[
    FALLBACKS_FR.no_sources,
    FALLBACKS_FR.insufficient_evidence,
    FALLBACKS_EN.no_sources,
    FALLBACKS_EN.insufficient_evidence,
    FALLBACKS_DE.no_sources,
    FALLBACKS_DE.insufficient_evidence,
    FALLBACKS_ES.no_sources,
    FALLBACKS_ES.insufficient_evidence,
];

fn fallbacks_for(locale: &str) -> &'static Fallbacks {
    if locale.starts_with("fr") {
        &FALLBACKS_FR
    } else if locale.starts_with("de") {
        &FALLBACKS_DE
    } else if locale.starts_with("es") {
        &FALLBACKS_ES
    } else {
        &FALLBACKS_EN
    }
}

/// The text the assistant returns when the notebook has no source at all.
#[must_use]
pub fn no_sources_text(locale: &str) -> &'static str {
    fallbacks_for(locale).no_sources
}

/// The text the assistant returns when sources exist but none was retrieved.
#[must_use]
pub fn insufficient_evidence_text(locale: &str) -> &'static str {
    fallbacks_for(locale).insufficient_evidence
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The PRD's edge-case table specifies the French sentences word for word.
    #[test]
    fn the_documented_french_sentences_are_verbatim() {
        assert_eq!(
            no_sources_text("fr"),
            "Aucune source n'est disponible dans ce notebook."
        );
        assert_eq!(
            insufficient_evidence_text("fr"),
            "Les sources disponibles ne permettent pas de répondre avec suffisamment de confiance."
        );
    }

    /// A user reads their own language. The English fallback used to open with
    /// the French sentence so the evaluator's substring matcher would recognize
    /// it; the table below is what replaced that coupling.
    #[test]
    fn every_locale_answers_in_its_own_language() {
        for (locale, needle) in [
            ("en", "No source is available"),
            ("de", "keine Quelle"),
            ("es", "ninguna fuente"),
        ] {
            let text = no_sources_text(locale);
            assert!(text.contains(needle), "{locale}: {text}");
            assert!(
                !text.contains("Aucune source"),
                "{locale} must not carry the French sentence: {text}"
            );
        }
        assert!(!insufficient_evidence_text("en").contains("Les sources disponibles"));
    }

    /// Every sentence a locale can produce is in the table the evaluator reads.
    /// Adding a locale without extending the table would silently score its
    /// abstentions as unsupported answers (US-003).
    #[test]
    fn the_fallback_table_covers_every_locale_the_selector_serves() {
        for locale in ["fr", "en", "de", "es", "ja", "", "pt-BR"] {
            for text in [no_sources_text(locale), insufficient_evidence_text(locale)] {
                assert!(
                    FALLBACK_TEXTS.contains(&text),
                    "'{text}' ({locale}) is not in FALLBACK_TEXTS"
                );
            }
        }
        assert_eq!(
            FALLBACK_TEXTS.len(),
            8,
            "four locales, two sentences each; update this when a locale is added"
        );
    }

    #[test]
    fn the_evaluator_reads_every_locale_as_an_abstention() {
        use crate::services::rag::eval::grounding::reads_as_abstention;

        for locale in ["fr", "en", "de", "es"] {
            assert!(reads_as_abstention(no_sources_text(locale)), "{locale}");
            assert!(
                reads_as_abstention(insufficient_evidence_text(locale)),
                "{locale}"
            );
        }
        assert!(!reads_as_abstention("The retry budget is four attempts."));
    }
}
