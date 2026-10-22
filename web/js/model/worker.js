// worker.js — a Web Worker for the network's heavy, per-snapshot computations, off the main thread.
//
// Local explanation: receives {id, op, flat, layout, p}. op "tables": builds a GrokModel from the weights (its
// precompute is the ~45-million-multiply-add part) and returns model.toTables(). op "ablation": returns the logits of
// every answer for every input (model.logitsGrid, ~740 million multiply-adds) and their per-answer Fourier blocks
// (model/ablation.js logitBlocks), the inputs of the ablation playground; the two big arrays are transferred, not
// copied. Errors come back as {id, error}.
// Global context: core/data.js asks this worker for every new weight snapshot, so playing or ablating never freezes the
// page; model/client.js handles the messaging and the main-thread fallback.
// Sources: module workers (https://developer.mozilla.org/docs/Web/API/Worker/Worker#type); transferable objects
// (https://developer.mozilla.org/docs/Web/API/Web_Workers_API/Transferable_objects).
import { GrokModel } from "./grokmodel.js";
import { logitBlocks } from "./ablation.js";

self.onmessage = (e) => {
  const { id, op, flat, layout, p } = e.data;
  try {
    const net = new GrokModel(flat, layout, p);
    if (op === "tables") { self.postMessage({ id, result: net.toTables() }); return; }
    if (op === "ablation") {
      const L = net.logitsGrid(), blocks = logitBlocks(L, p);
      self.postMessage({ id, result: { L, blocks } }, [L.buffer, blocks.buffer]);
      return;
    }
    throw new Error(`unknown op ${op}`);
  } catch (err) {
    self.postMessage({ id, error: String(err) });
  }
};
