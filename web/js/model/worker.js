// worker.js — a Web Worker that precomputes GrokModel tables off the main thread.
//
// Local explanation: receives {id, op: "tables", flat, layout, p}, builds a GrokModel from the weights (its precompute
// is the ~45-million-multiply-add part) and posts back {id, result: model.toTables()}; errors come back as {id, error}.
// Global context: core/data.js `model()` asks this worker for every new weight snapshot, so playing a run through
// training never freezes the page while a snapshot's tables are computed; model/client.js handles the messaging.
// Sources: module workers (https://developer.mozilla.org/docs/Web/API/Worker/Worker#type); structured clone of typed
// arrays (https://developer.mozilla.org/docs/Web/API/Web_Workers_API/Structured_clone_algorithm).
import { GrokModel } from "./grokmodel.js";

self.onmessage = (e) => {
  const { id, op, flat, layout, p } = e.data;
  try {
    if (op !== "tables") throw new Error(`unknown op ${op}`);
    self.postMessage({ id, result: new GrokModel(flat, layout, p).toTables() });
  } catch (err) {
    self.postMessage({ id, error: String(err) });
  }
};
