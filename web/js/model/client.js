// client.js — build GrokModels with their tables computed in model/worker.js (main-thread fallback).
//
// Local explanation: `buildModel(flat, layout, p)` sends a copy of the weights to one shared module worker, which
// returns the precomputed tables; the model is then assembled here with GrokModel.fromTables (cheap). Requests are
// matched to replies by id. If workers are unavailable or the worker fails, the model is built on this thread as before.
// Global context: core/data.js uses this for every weight snapshot, so scrubbing or playing through training keeps
// the page responsive (the precompute takes ~0.1 s per snapshot).
// Sources: Worker constructor with type "module" and import.meta.url (https://developer.mozilla.org/docs/Web/API/Worker/Worker).
import { GrokModel } from "./grokmodel.js";

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

/** A GrokModel for weights `flat` (Float32Array), with its tables computed in the worker when possible. */
export async function buildModel(flat, layout, p) {
  const w = getWorker();
  if (w) {
    try {
      const tables = await new Promise((resolve, reject) => {
        const id = nextId++;
        pending.set(id, { resolve, reject });
        w.postMessage({ id, op: "tables", flat, layout, p }); // structured clone copies the weights; the page keeps its own
      });
      return GrokModel.fromTables(flat, layout, p, tables);
    } catch { /* fall through to the main thread */ }
  }
  return new GrokModel(flat, layout, p);
}
