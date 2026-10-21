// data.js — load the exported manifest, binary arrays and model weights (with caching).
//
// Local explanation: `manifest()` fetches data/manifest.json once; `array(runId, name)` fetches one .bin file,
// decodes it according to its manifest entry (f32 | f16 | i16 | u8q = uint8 linearly quantised) and returns
// {data: Float32Array, shape}; `model(runId, step)` returns a GrokModel for the stored checkpoint nearest to
// `step`. Results are cached so moving the scrubber back and forth does not refetch.
// Global context: the only place that knows the data format written by src/grokviz/export.py.
// Sources: fetch + ArrayBuffer (https://developer.mozilla.org/docs/Web/API/Response/arrayBuffer).
import { GrokModel, decodeF16 } from "../model/grokmodel.js";

const ROOT = (window.GROKVIZ && window.GROKVIZ.root) || "./"; // relative path from this page to the site root
const cache = new Map(); // url → Promise
let manifestPromise = null;

/** Fetch a URL once and cache the promise (concurrent callers share one request). */
function once(url, kind) {
  if (!cache.has(url)) {
    cache.set(url, fetch(ROOT + url).then((r) => {
      if (!r.ok) throw new Error(`${r.status} ${url}`);
      return kind === "json" ? r.json() : r.arrayBuffer();
    }));
  }
  return cache.get(url);
}

/** The site manifest (runs, arrays, layout, basis names). */
export function manifest() {
  manifestPromise ||= once("data/manifest.json", "json");
  return manifestPromise;
}

/** Manifest entry of one run. */
export async function run(runId) {
  const m = await manifest();
  const r = m.runs.find((x) => x.id === runId);
  if (!r) throw new Error(`unknown run ${runId}`);
  return r;
}

/** Decode raw bytes according to a manifest array entry into a Float32Array. */
export function decode(buf, entry) {
  switch (entry.dtype) {
    case "f32": return new Float32Array(buf);
    case "f16": return decodeF16(new Uint16Array(buf));
    case "i16": return Float32Array.from(new Int16Array(buf));
    case "u8q": { // linear de-quantisation: min + q/255·(max − min)
      const q = new Uint8Array(buf), out = new Float32Array(q.length), lo = entry.min, span = entry.max - entry.min;
      for (let i = 0; i < q.length; i++) out[i] = lo + (q[i] / 255) * span;
      return out;
    }
    default: throw new Error(`dtype ${entry.dtype}`);
  }
}

const decoded = new Map(); // "run/name" → Promise<{data, shape, entry}>
/** Load + decode one exported array of a run. */
export function array(runId, name) {
  const key = `${runId}/${name}`;
  if (!decoded.has(key)) {
    decoded.set(key, run(runId).then(async (r) => {
      const entry = r.arrays[name];
      if (!entry) throw new Error(`run ${runId} has no array ${name}`);
      const buf = await once(`data/${entry.file}`, "bin"); // manifest paths are relative to data/
      return { data: decode(buf, entry), shape: entry.shape, entry };
    }));
  }
  return decoded.get(key);
}

/** Column of a packed per-checkpoint scalar table, e.g. column(run, "ckpt_scalars", "loss_test"). */
export async function column(runId, name, col) {
  const a = await array(runId, name);
  const cols = a.entry.columns;
  const j = cols.indexOf(col);
  if (j < 0) throw new Error(`no column ${col} in ${name}`);
  const n = a.shape[0], w = a.shape[1], out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = a.data[i * w + j];
  return out;
}

/** Row i of an array with shape [T, ...] as a flat Float32Array view. */
export function row(a, i) {
  const size = a.shape.slice(1).reduce((x, y) => x * y, 1);
  return a.data.subarray(i * size, (i + 1) * size);
}

/** Index of the stored step closest to `step` in a sorted list of steps. */
export function nearestIndex(steps, step) {
  let lo = 0, hi = steps.length - 1;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (steps[mid] < step) lo = mid + 1; else hi = mid; } // first ≥ step
  if (lo > 0 && Math.abs(steps[lo - 1] - step) <= Math.abs(steps[lo] - step)) lo -= 1;
  return lo;
}

const models = new Map(); // "run@step" → Promise<GrokModel>
/** GrokModel for the stored weight checkpoint of `runId` nearest to `step`. */
export async function model(runId, step) {
  const r = await run(runId);
  const s = r.weight_steps[nearestIndex(r.weight_steps, step)];
  const key = `${runId}@${s}`;
  if (!models.has(key)) {
    models.set(key, array(runId, `w_${s}`).then((a) => Object.assign(new GrokModel(a.data, r.param_layout, r.p || 113), { step: s })));
    if (models.size > 24) models.delete(models.keys().next().value); // keep memory bounded (LRU-ish)
  }
  return models.get(key);
}

/** Train mask (1 = training pair) for a split seed, as a Uint8Array over a*p+b. */
export async function trainMask(splitSeed) {
  const m = await manifest();
  return new Uint8Array(await once(`data/${m.train_masks[String(splitSeed)]}`, "bin"));
}
