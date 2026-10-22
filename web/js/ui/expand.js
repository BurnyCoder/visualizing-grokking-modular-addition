// expand.js — enlarge any figure in an overlay (the ⤢ button of templates/figure.html).
//
// Local explanation: clicking ⤢ moves the figure element itself into a modal dialog (so its live state, data and
// subscriptions stay as they are) and puts it back on close: ✕, Esc or a click on the backdrop. Focus moves into
// the dialog and Tab cycles inside it; on close focus returns to the ⤢ button. A window "resize" event lets Plotly's
// responsive charts refit both ways. On the dashboard the overlay starts below the sticky control bar, so playback
// stays usable while a figure is enlarged; on phones it is a full-screen sheet (main.css).
// Global context: every figure on every page, dashboard panels included.
// Sources: WAI-ARIA dialog (modal) pattern (https://www.w3.org/WAI/ARIA/apg/patterns/dialog-modal/); Plotly responsive
// charts refit on window resize (https://plotly.com/javascript/responsive-fluid-layout/).

const FOCUSABLE = "button, [href], input, select, textarea, [tabindex]:not([tabindex='-1'])";
const refit = () => window.dispatchEvent(new Event("resize"));

/** Open `fig` in the overlay. */
function open(fig, button) {
  const home = document.createComment("figure home"); // marks where the figure goes back
  fig.before(home);
  const overlay = document.createElement("div");
  overlay.className = "fig-modal";
  overlay.setAttribute("role", "dialog");
  overlay.setAttribute("aria-modal", "true");
  overlay.setAttribute("aria-label", fig.querySelector(".fig-title")?.textContent || "Figure");
  const bar = document.querySelector(".controlbar"); // dashboard: keep the player bar visible above the overlay
  if (bar) overlay.style.top = `${Math.max(0, bar.getBoundingClientRect().bottom)}px`;
  overlay.innerHTML = '<div class="fig-modal-box"><button type="button" class="btn fig-modal-close" aria-label="Close the enlarged figure">✕</button></div>';
  const box = overlay.firstChild;
  box.append(fig);
  document.body.append(overlay);
  document.body.classList.add("modal-open");
  const close = () => {
    home.replaceWith(fig);
    overlay.remove();
    document.body.classList.remove("modal-open");
    document.removeEventListener("keydown", onKey);
    refit();
    button.focus();
  };
  function onKey(e) {
    if (e.key === "Escape") { e.preventDefault(); close(); return; }
    if (e.key !== "Tab") return;
    const items = [...overlay.querySelectorAll(FOCUSABLE)].filter((x) => !x.disabled && x.offsetParent !== null);
    const first = items[0], last = items.at(-1);
    if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
    else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
  }
  document.addEventListener("keydown", onKey);
  overlay.addEventListener("click", (e) => { if (e.target === overlay) close(); });
  box.querySelector(".fig-modal-close").addEventListener("click", close);
  box.querySelector(".fig-modal-close").focus();
  refit();
}

/** Handle every ⤢ button on the page (event delegation, so lazily created figures need nothing extra). */
export function initExpand() {
  document.addEventListener("click", (e) => {
    const b = e.target.closest(".fig-expand");
    if (b && !b.closest(".fig-modal")) open(b.closest(".fig"), b);
  });
}
