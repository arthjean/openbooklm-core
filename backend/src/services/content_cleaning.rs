//! Scraped content cleaning for web sources.
//!
//! Cleans up raw markdown from Firecrawl to remove noise: images, nav elements,
//! YouTube embeds, newsletter CTAs, duplicate lines, and invisible characters.
//! The actual HTTP scraping is handled by [`FirecrawlClient`](crate::clients::FirecrawlClient).

use std::sync::LazyLock;

use regex::Regex;

/// Pre-compiled regex patterns for content cleaning (compiled once at startup)
static CONTENT_PATTERNS: LazyLock<ContentPatterns> = LazyLock::new(ContentPatterns::new);

/// Clean up scraped markdown content to remove noise.
///
/// Applied after [`FirecrawlClient::scrape_url`](crate::clients::FirecrawlClient::scrape_url)
/// returns raw markdown. Removes images, navigation elements, YouTube embed noise,
/// newsletter CTAs, invisible characters, and deduplicates consecutive lines.
pub fn clean_scraped_content(content: &str) -> String {
    let patterns = &*CONTENT_PATTERNS;

    // Remove invisible/formatting characters
    let mut result = content.to_string();
    for &(ch, replacement) in INVISIBLE_CHARS {
        result = result.replace(ch, replacement);
    }

    // Apply regex-based cleaning
    result = patterns.image.replace_all(&result, "").into_owned();
    result = patterns.image_url.replace_all(&result, "").into_owned();
    result = patterns.empty_link.replace_all(&result, "").into_owned();
    result = patterns.breadcrumb.replace_all(&result, "").into_owned();
    result = patterns.episode_nav.replace_all(&result, "").into_owned();
    result = patterns.episode_card.replace_all(&result, "").into_owned();

    // Remove string-based noise patterns
    for noise in YOUTUBE_NOISE {
        result = result.replace(noise, "");
    }

    result = patterns.timestamp.replace_all(&result, "").into_owned();
    result = result
        .replace("•Live•", "")
        .replace("•Live", "")
        .replace("Live•", "");

    for noise in NEWSLETTER_NOISE {
        result = result.replace(noise, "");
    }

    for noise in SECTION_NOISE {
        result = result.replace(noise, "");
    }

    result = patterns.nav_arrow.replace_all(&result, "").into_owned();
    result = patterns.pi_tags.replace_all(&result, "").into_owned();
    result = patterns
        .episodes_count
        .replace_all(&result, "")
        .into_owned();
    result = patterns.read_time.replace_all(&result, "").into_owned();

    // Remove backslashes from escaped characters
    result = result.replace("\\\\", "").replace('\\', "");

    result = patterns
        .standalone_link
        .replace_all(&result, "")
        .into_owned();
    result = patterns
        .link_to_text
        .replace_all(&result, "$1")
        .into_owned();
    result = patterns
        .empty_link_whitespace
        .replace_all(&result, "")
        .into_owned();
    result = patterns.empty_heading.replace_all(&result, "").into_owned();
    result = patterns.lone_punct.replace_all(&result, "").into_owned();

    // Remove related article titles (bold-wrapped and standalone lines)
    for title in RELATED_TITLES {
        result = result.replace(&format!("**{title}**"), "");
    }
    for re in &patterns.related_titles {
        result = re.replace_all(&result, "").into_owned();
    }

    // Deduplicate consecutive lines
    result = deduplicate_lines(&result);

    // Final cleanup
    result = patterns
        .multi_newline
        .replace_all(&result, "\n\n")
        .into_owned();
    result = patterns
        .whitespace_line
        .replace_all(&result, "")
        .into_owned();

    result.trim().to_string()
}

/// Remove duplicate consecutive lines
fn deduplicate_lines(content: &str) -> String {
    let mut deduped: Vec<&str> = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || deduped.last().map(|l| l.trim()) != Some(trimmed) {
            deduped.push(line);
        }
    }
    deduped.join("\n")
}

// ============================================================================
// Pattern data
// ============================================================================

/// Pre-compiled regex patterns for content cleaning
struct ContentPatterns {
    image: Regex,
    image_url: Regex,
    empty_link: Regex,
    breadcrumb: Regex,
    episode_nav: Regex,
    episode_card: Regex,
    timestamp: Regex,
    nav_arrow: Regex,
    pi_tags: Regex,
    episodes_count: Regex,
    read_time: Regex,
    standalone_link: Regex,
    link_to_text: Regex,
    empty_link_whitespace: Regex,
    empty_heading: Regex,
    lone_punct: Regex,
    multi_newline: Regex,
    whitespace_line: Regex,
    related_titles: Vec<Regex>,
}

impl ContentPatterns {
    // Every pattern is a static literal or a regex-escaped static title. A
    // failure is a source-code defect discovered by unit tests, not runtime
    // input that can be recovered from.
    #[allow(clippy::expect_used)]
    fn new() -> Self {
        Self {
            image: Regex::new(r"!\[[^\]]*\]\([^)]*\)").expect("valid regex"),
            image_url: Regex::new(
                r"(?m)^\s*https?://[^\s]+\.(jpg|jpeg|png|gif|svg|webp)[^\s]*\s*$",
            )
            .expect("valid regex"),
            empty_link: Regex::new(r"\[\]\([^)]*\)").expect("valid regex"),
            breadcrumb: Regex::new(r"(?m)^\[(Accueil|Home)\][^\n]*\n").expect("valid regex"),
            episode_nav: Regex::new(r"(?m)^\s*(\[\d+\]\([^)]+\)\s*)+\s*$").expect("valid regex"),
            episode_card: Regex::new(r"(?s)\[Episode \d+ / \d+.*?de lecture\]\([^)]+\)")
                .expect("valid regex"),
            timestamp: Regex::new(r"\d+:\d+\s*/\s*\d+:\d+|\b\d+:\d{2}\b").expect("valid regex"),
            nav_arrow: Regex::new(r"(?m)\[←[^\]]*\]\([^)]+\)|\[→[^\]]*\]\([^)]+\)")
                .expect("valid regex"),
            pi_tags: Regex::new(r"π\s*[^π\n]+").expect("valid regex"),
            episodes_count: Regex::new(r"\d+\s*(épisodes?|episodes?)").expect("valid regex"),
            read_time: Regex::new(r"\d+\s*min\.\s*(de\s*lecture|read)").expect("valid regex"),
            standalone_link: Regex::new(r"(?m)^\s*\[[^\]]+\]\([^)]+\)\s*$").expect("valid regex"),
            link_to_text: Regex::new(r"\[([^\]]+)\]\([^)]+\)").expect("valid regex"),
            empty_link_whitespace: Regex::new(r"\[\s*\]\([^)]*\)").expect("valid regex"),
            empty_heading: Regex::new(r"(?m)^#{1,6}\s*\**\s*\**\s*$").expect("valid regex"),
            lone_punct: Regex::new(r"(?m)^\s*[.,;:!?]\s*$").expect("valid regex"),
            multi_newline: Regex::new(r"\n{3,}").expect("valid regex"),
            whitespace_line: Regex::new(r"(?m)^\s+$").expect("valid regex"),
            related_titles: RELATED_TITLES
                .iter()
                .map(|title| {
                    let escaped = regex::escape(title);
                    Regex::new(&format!(r"(?m)^\s*{escaped}\s*$")).expect("valid regex")
                })
                .collect(),
        }
    }
}

/// Invisible/formatting characters to remove
const INVISIBLE_CHARS: &[(char, &str)] = &[
    ('\u{00AD}', ""),  // Soft hyphen
    ('\u{200B}', ""),  // Zero-width space
    ('\u{200C}', ""),  // Zero-width non-joiner
    ('\u{200D}', ""),  // Zero-width joiner
    ('\u{FEFF}', ""),  // BOM / Zero-width no-break space
    ('\u{00A0}', " "), // Non-breaking space -> regular space
];

/// YouTube embed noise patterns to remove
const YOUTUBE_NOISE: &[&str] = &[
    "Watch later",
    "Share",
    "Copy link",
    "Watch on",
    "If playback doesn't begin shortly",
    "try restarting your device",
    "You're signed out",
    "Videos you watch may be added",
    "TV's watch history",
    "influence TV recommendations",
    "To avoid this, cancel and sign in",
    "CancelConfirm",
    "Include playlist",
    "An error occurred while retrieving sharing information",
    "Please try again later",
    "Tap to unmute",
    "Info\n\nShopping",
    "Search\n\nInfo",
];

/// Newsletter/subscription noise patterns
const NEWSLETTER_NOISE: &[&str] = &[
    // English
    "Subscribe to our newsletter",
    "Our weekly newsletter",
    "weekly newsletter",
    "SUBSCRIBE",
    "Sign up and receive",
    "We could not confirm your subscription",
    "Your subscription is confirmed",
    "Please enter your email address",
    "I agree to receive your emails",
    "privacy policy",
    "legal notices",
    // French
    "Abonnez-vous à notre newsletter",
    "Notre newsletter hebdomadaire",
    "newsletter hebdomadaire",
    "hebdomadaire hebdomadaire",
    "JE M'ABONNE",
    "Inscrivez-vous et recevez",
    "Nous n'avons pas pu confirmer votre inscription",
    "Votre inscription est confirmée",
    "Veuillez renseigner votre adresse email",
    "J'accepte de recevoir vos e-mails",
    "politique de confidentialité",
    "mentions légales",
];

/// Website section noise patterns to remove
const SECTION_NOISE: &[&str] = &[
    // English
    "Other episodes:",
    "In the same topic:",
    "Discover the other episodes in this series",
    "Our selection of series",
    "Support reliable information",
    "based on the scientific method",
    "Donate",
    // French
    "Autres épisodes :",
    "Dans la même thématique :",
    "Découvrez les autres épisodes de ce dossier",
    "Notre sélection de dossiers",
    "Soutenez une information fiable",
    "basée sur la méthode scientifique",
    "Faire un don",
];

/// Related article titles to remove
const RELATED_TITLES: &[&str] = &[
    "Comment le quantique change la face du monde",
    "Batteries : les enjeux autour du stockage d'énergie se multiplient",
    "Prix Nobel : quelles applications pour les travaux des derniers lauréats",
];
