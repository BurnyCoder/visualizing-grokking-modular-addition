// colors.js — colour maps and theme-aware colours shared by every chart.
//
// Local explanation: `rdbu(t)` (diverging, for signed values centred at 0) and `viridis(t)` (sequential, for
// non-negative values) map t ∈ [0, 1] to RGB by linear interpolation between published anchor colours;
// `cssVar(name)` reads the current theme's colour tokens from main.css; `freqColor(k)` gives each frequency
// one fixed colour across the whole site so "frequency 14" always looks the same; `cyclic(t)` is a perceptually
// uniform cyclic map (same colour at t = 0 and 1) for positions on a clock.
// Global context: canvas heatmaps and Plotly charts use these so both themes and all figures stay consistent.
// Sources: ColorBrewer RdBu 11-class (https://colorbrewer2.org/#type=diverging&scheme=RdBu&n=11);
// matplotlib viridis anchor values (https://github.com/matplotlib/matplotlib/blob/main/lib/matplotlib/_cm_listed.py);
// matplotlib "twilight" cyclic map (Smith & van der Walt; sampled at 13 evenly spaced points from matplotlib 3.11.2,
// https://matplotlib.org/stable/users/explain/colors/colormaps.html#cyclic): perceptually uniform and cyclic, unlike
// the HSL hue wheel, whose equal steps look unequal (lightness varies with hue).

const RDBU = ["#67001f", "#b2182b", "#d6604d", "#f4a582", "#fddbc7", "#f7f7f7", "#d1e5f0", "#92c5de", "#4393c3", "#2166ac", "#053061"].reverse(); // low = blue, high = red
const VIRIDIS = ["#440154", "#482878", "#3e4989", "#31688e", "#26828e", "#1f9e89", "#35b779", "#6ece58", "#b5de2b", "#fde725"];
const TWILIGHT = ["#e2d9e2", "#b3c6ce", "#7ba1c2", "#6276ba", "#5e43a5", "#4e186f", "#2f1436", "#581647", "#8e2c50", "#b25652", "#c6896c", "#d4bcac", "#e2d9e2"];

const hexToRgb = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16)); // "#rrggbb" → [r, g, b]
const RDBU_RGB = RDBU.map(hexToRgb);
const VIRIDIS_RGB = VIRIDIS.map(hexToRgb);
const TWILIGHT_RGB = TWILIGHT.map(hexToRgb);

/** Interpolate a list of RGB anchors at t ∈ [0, 1]. */
function interp(anchors, t) {
  const x = Math.min(Math.max(t, 0), 1) * (anchors.length - 1); // position along the anchor list
  const i = Math.min(Math.floor(x), anchors.length - 2), f = x - i;
  const a = anchors[i], b = anchors[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

/** Diverging map: t = 0.5 is white (value 0), t → 0 blue (negative), t → 1 red (positive). */
export const rdbu = (t) => interp(RDBU_RGB, t);
/** Sequential map for non-negative quantities. */
export const viridis = (t) => interp(VIRIDIS_RGB, t);
/** Cyclic map: position t ∈ [0, 1) around a circle (t and t + 1 get the same colour). */
export const cyclic = (t) => interp(TWILIGHT_RGB, t - Math.floor(t));
/** CSS colour of clock position i of n (e.g. number i on a 113-hour clock). */
export const clockColor = (i, n) => `rgb(${cyclic(i / n).map(Math.round).join(",")})`;

/** Current value of a CSS custom property, e.g. cssVar("--ink"). */
export function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

const PALETTE = ["#d1495b", "#2e86ab", "#edae49", "#00798c", "#8d6a9f", "#66a182", "#e07a5f", "#3d405b", "#c44536", "#2a9d8f"];
/** A fixed colour for frequency k (same k → same colour on every page). */
export const freqColor = (k) => PALETTE[(k * 7) % PALETTE.length];
/** A fixed colour per run id. */
export function runColor(id) {
  if (id === "paper") return "#b4532a";
  let h = 0;
  for (const ch of id) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
  return PALETTE[h % PALETTE.length];
}
