// motion.js — whether the page may animate.
//
// Local explanation: `still` is true when the URL has ?still=1 (screenshots, report figures, recorded clips and
// tests want deterministic frames); `reducedMotion()` is true for `still` or when the reader's system asks for
// reduced motion. The `still` flag also sets class "still" on <html>, which main.css treats like the media query
// (no transitions or smooth scrolling).
// Global context: the tour, the step cursor and (later) the player read these flags so the site never forces motion
// on readers who opted out.
// Sources: prefers-reduced-motion (https://developer.mozilla.org/docs/Web/CSS/@media/prefers-reduced-motion);
// matchMedia (https://developer.mozilla.org/docs/Web/API/Window/matchMedia).

export const still = new URLSearchParams(location.search).has("still"); // ?still=1
const query = matchMedia("(prefers-reduced-motion: reduce)"); // live: follows changes of the system setting

/** True when animations should be replaced by their end state. */
export const reducedMotion = () => still || query.matches;

if (still) document.documentElement.classList.add("still");
