// app.js — bootstrap loaded by every page (ES module).
//
// Local explanation: (1) theme toggle, (2) render every math element with KaTeX, (3) glossary/symbol tooltips,
// (4) lazily create each figure ([data-fig]) and widget ([data-widget]) when it scrolls into view, (5) start the
// dashboard controller on the dashboard page.
// Global context: pages are static HTML produced by site_build.py; all interactivity is attached here.
// Sources: IntersectionObserver (https://developer.mozilla.org/docs/Web/API/Intersection_Observer_API);
// katex.render (https://katex.org/docs/api).
import { initTooltips } from "./ui/tooltips.js";
import { FIGURES } from "./figures/index.js";
import { WIDGETS } from "./widgets/index.js";
import "./core/motion.js"; // sets html.still for ?still=1 captures before anything draws
import "./core/hooks.js"; // window.grokviz.anim test/capture hooks on every page

const cfg = window.GROKVIZ || {}; // per-page settings injected by the build (root path, page kind)

/** Light/dark toggle; remembers the choice per viewer (falls back silently if storage is blocked). */
function initTheme() {
  try { const t = localStorage.getItem("grokviz-theme"); if (t) document.documentElement.dataset.theme = t; } catch { /* storage blocked */ }
  document.querySelector(".theme-toggle")?.addEventListener("click", () => {
    const dark = getComputedStyle(document.documentElement).colorScheme.includes("dark");
    const next = dark ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("grokviz-theme", next); } catch { /* ignore */ }
    window.dispatchEvent(new Event("grokviz-theme")); // figures redraw with the new colours
  });
}

/** Wait for the deferred KaTeX script, then typeset every .math element (and later-added ones on demand). */
export async function renderMath(root = document) {
  while (!window.katex) await new Promise((r) => setTimeout(r, 30));
  for (const el of root.querySelectorAll(".math:not([data-done])")) {
    const display = el.classList.contains("block") || el.tagName === "DIV";
    try {
      window.katex.render(el.textContent, el, { displayMode: display, throwOnError: false, macros: cfg.macros || {}, trust: false, strict: "ignore" });
    } catch (e) { console.warn("KaTeX", e); }
    el.dataset.done = "1";
  }
}
window.grokvizRenderMath = renderMath; // figures call this after inserting math

/** Create figures/widgets when they approach the viewport (keeps first load light). */
function initLazy(selector, registry, attr) {
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      io.unobserve(e.target);
      const id = e.target.dataset[attr];
      const loader = registry[id];
      const body = e.target.querySelector(".fig-body") || e.target;
      if (!loader) { body.innerHTML = `<p class="loading">Unknown ${attr} “${id}”.</p>`; continue; }
      loader().then((mod) => mod.render(body, e.target.dataset)).then(() => renderMath(e.target))
        .catch((err) => { console.error(err); body.innerHTML = `<p class="draw-error">Could not draw this (${err.message}).</p>`; });
    }
  }, { rootMargin: "400px 0px" });
  document.querySelectorAll(selector).forEach((el) => io.observe(el));
}

initTheme();
renderMath();
initTooltips(cfg.root || "./");
initLazy("[data-fig]", FIGURES, "fig");
initLazy("[data-widget]", WIDGETS, "widget");
// The chapter list is a <details>: open beside the text on wide screens, collapsed above it on phones (same breakpoint as main.css).
document.querySelectorAll("details.toc").forEach((d) => { d.open = matchMedia("(min-width: 52.01rem)").matches; });
if (document.body.classList.contains("dashboard-page")) import("./dashboard.js").then((m) => m.initDashboard());
