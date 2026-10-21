// tooltips.js — hover/focus cards for defined terms (glossary) and mathematical symbols.
//
// Local explanation: loads assets/glossary.json ({terms: {id: {name, def, src}}, symbols: {id: {tex, def}}})
// written by the build, and shows a small card when the reader hovers or focuses an element with
// data-term / data-sym. Each card holds the one-sentence definition and its source.
// Global context: implements "every variable and composite has a one-sentence explanation" everywhere on the
// site, including the dashboard, where terms appear out of reading order.
// Source: Pointer/Focus events (https://developer.mozilla.org/docs/Web/API/Element/pointerenter_event).

let data = null; // glossary + symbols, loaded once

/** Attach delegated listeners for [data-term] and [data-sym] anywhere in the page. */
export async function initTooltips(root) {
  data = await fetch(`${root}assets/glossary.json`).then((r) => r.json()).catch(() => ({ terms: {}, symbols: {} }));
  const tt = document.getElementById("tooltip");
  const show = (el, html, x, y) => {
    tt.innerHTML = html;
    tt.hidden = false;
    const r = el.getBoundingClientRect();
    tt.style.left = `${Math.min((x ?? r.left) + 8, window.innerWidth - 360)}px`;
    tt.style.top = `${(y ?? r.bottom) + 10}px`;
  };
  const card = (el) => {
    if (el.dataset.term && data.terms[el.dataset.term]) {
      const t = data.terms[el.dataset.term];
      return `<span class="tt-term">${t.name}</span>${t.def}<span class="tt-src">${t.src || ""}</span>`;
    }
    if (el.dataset.sym && data.symbols[el.dataset.sym]) {
      const s = data.symbols[el.dataset.sym];
      return `<span class="tt-term">${s.name}</span>${s.def}`;
    }
    return null;
  };
  const target = (e) => e.target.closest?.("[data-term], [data-sym]");
  document.addEventListener("pointerover", (e) => { const el = target(e); const h = el && card(el); if (h) show(el, h, e.clientX, e.clientY); });
  document.addEventListener("pointerout", (e) => { if (target(e)) tt.hidden = true; });
  document.addEventListener("focusin", (e) => { const el = target(e); const h = el && card(el); if (h) show(el, h); });
  document.addEventListener("focusout", () => { tt.hidden = true; });
}
