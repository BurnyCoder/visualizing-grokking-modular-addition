// client.js — run the network's heavy per-snapshot computations in model/worker.js (main-thread fallback).
//
// Local explanation: `buildModel(flat, layout, p)` sends a copy of the weights to one shared module worker, which
// returns the precomputed tables; the model is then assembled here with GrokModel.fromTables (cheap).
// `ablationBasis(flat, layout, p)` returns every input's logits and their Fourier blocks for the ablation playground.
// Requests are matched to replies by id. If workers are unavailable or the worker fails, the same work is done on
// this thread instead.
// Global context: core/data.js uses this for every weight snapshot, so scrubbing, playing or ablating keeps the page
// responsive (the precompute takes ~0.1 s per snapshot, the ablation basis ~2 s).
// Sources: Worker constructor with type "module" and import.meta.url (https://developer.mozilla.org/docs/Web/API/Worker/Worker).
import { GrokModel } from "./grokmodel.js";
import { logitBlocks } from "./ablation.js";

let worker = null; // the shared worker; false once it proved unusable
let nextId = 0;
const pending = new Map(); // request id → {resolve, reject}

/** The shared worker, created on first use (null when workers cannot run here). */
function getWorker() {
  if (worker !== null) return worker || null;
  try {
    worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
    worker.onmessage = (e) => {
      const { id, result, error } = e.data, req = pending.get(id);
      pending.delete(id);
      if (error) req?.reject(new Error(error)); else req?.resolve(result);
    };
    worker.onerror = (e) => { // the worker itself failed (e.g. module workers unsupported): fall back for good
      e.preventDefault?.();
      worker = false;
      for (const req of pending.values()) req.reject(new Error("model worker failed"));
      pending.clear();
    };
  } catch {
    worker = false;
  }
  return worker || null;
}

/** Run `op` on weights `flat` in the worker; resolves with its result, or null when there is no usable worker. */
async function inWorker(op, flat, layout, p) {
  const w = getWorker();
  if (!w) return null;
  try {
    return await new Promise((resolve, reject) => {
      const id = nextId++;
      pending.set(id, { resolve, reject });
      w.postMessage({ id, op, flat, layout, p }); // structured clone copies the weights; the page keeps its own
    });
  } catch {
    return null; // fall through to the main thread
  }
}

/** A GrokModel for weights `flat` (Float32Array), with its tables computed in the worker when possible. */
export async function buildModel(flat, layout, p) {
  const tables = await inWorker("tables", flat, layout, p);
  return tables ? GrokModel.fromTables(flat, layout, p, tables) : new GrokModel(flat, layout, p);
}

/** {L: logits [p·p·p], blocks: their Fourier blocks} for the ablation playground, from the worker when possible. */
export async function ablationBasis(flat, layout, p) {
  const result = await inWorker("ablation", flat, layout, p);
  if (result) return result;
  const L = new GrokModel(flat, layout, p).logitsGrid();
  return { L, blocks: logitBlocks(L, p) };
}
