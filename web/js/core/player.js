// player.js — plays a run through its checkpoints on one store (the page's or an animated figure's scope).
//
// Local explanation: a Player keeps a fractional checkpoint index `pos`; each animation frame it advances `pos` by the
// elapsed time × speed (core/anim.js `advance`, so slow frames are skipped, never queued) and publishes the nearest
// stored step as `step` plus the interpolated `stepf` and `playing: true`. Pausing publishes the stored frame
// (`stepf: null`). It offers play / pause / seek (with a 400 ms eased tween) / stepBy / speed / loop mode / range and
// phase presets, and follows outside step changes (a click on a chart pauses it and moves it there). Under reduced
// motion it moves by whole checkpoints (2 per second) and seeks instantly. UIs subscribe with `onChange`.
// Global context: the dashboard's player bar, the chapter scrubber, scroll-played figures ({{anim:id}}) and the tour
// all drive one of these; the figures only see the store's `step` / `stepf` / `playing`.
// No DOM at import time (clock and motion preference are injected), so tests/js/unit/anim.test.mjs drives it with a
// fake clock. Sources: requestAnimationFrame timestamps (https://developer.mozilla.org/docs/Web/API/Window/requestAnimationFrame).
import { advance, ease, nearest, rangePresets, stepAt } from "./anim.js";

const SEEK_MS = 400; // duration of an animated seek
const REDUCED_RATE = 2; // checkpoints per second when the reader prefers reduced motion
const MAX_DT = 0.25; // seconds; caps the jump after a stalled frame or a hidden tab

export class Player {
  /**
   * @param {object} store  core/store.js store (owns step, stepf, playing)
   * @param {{raf?: Function, now?: Function, reduced?: Function}} o  frame scheduler, clock (ms), reduced-motion test
   */
  constructor(store, { raf = (f) => requestAnimationFrame(f), now = () => performance.now(), reduced = () => false } = {}) {
    Object.assign(this, { store, raf, now, reduced });
    this.steps = null; // stored checkpoint steps of the current run
    this.presets = {}; // named ranges (core/anim.js rangePresets)
    this.range = [0, 0]; // [lo, hi] checkpoint indices being played
    this.pos = 0; // fractional checkpoint index
    this.speed = 1;
    this.loop = "loop"; // "loop" | "pingpong" | "once"
    this.playing = false;
    this.listeners = []; // UI callbacks
    this.published = null; // last step this player wrote (to tell its own updates from outside changes)
    this.frame = null; // pending animation-frame flag
    store.subscribe(["step"], (s) => { // an outside change (click, typed step, another control) wins
      if (!this.steps || s.step === this.published) return;
      this.tween = null;
      this.pos = nearest(this.steps, s.step);
      this.published = this.steps[this.pos];
      if (this.playing) this.pause(); else this.emit();
    }, { immediate: false });
  }

  /** Use a run's checkpoint steps and phases (pauses; the step snaps to the run's nearest stored checkpoint). */
  setSteps(steps, phases, range = "grokking") {
    this.steps = steps;
    this.phases = phases;
    this.presets = rangePresets(steps, phases);
    this.setRange(range);
    this.seek(nearest(steps, this.store.state.step));
  }

  /** Call fn(player) after every change of play state, position, speed, loop or range. Returns an unsubscribe. */
  onChange(fn) {
    this.listeners.push(fn);
    return () => this.listeners.splice(this.listeners.indexOf(fn), 1);
  }

  emit() { for (const fn of this.listeners.slice()) fn(this); }

  play() {
    if (!this.steps || this.playing) return;
    const [lo, hi] = this.range;
    if (this.pos >= hi || this.pos < lo) this.pos = lo; // restart from the beginning of the range
    this.state = { pos: this.pos, dir: 1, hold: 0 };
    this.acc = 0; // fractional checkpoints accumulated under reduced motion
    this.playing = true;
    this.last = null;
    this.publish();
    this.schedule();
  }

  pause() {
    if (!this.playing && !this.tween) return;
    this.playing = false;
    this.tween = null;
    this.pos = Math.round(this.pos); // stop on a stored frame
    this.publish();
  }

  toggle() { if (this.playing) this.pause(); else this.play(); }

  /** Go to checkpoint index `pos` (eased over 400 ms when tween is true and motion is allowed); pauses playback. */
  seek(pos, { tween = false } = {}) {
    if (!this.steps) return;
    pos = Math.min(Math.max(pos, 0), this.steps.length - 1);
    this.playing = false;
    if (tween && !this.reduced() && Math.abs(pos - this.pos) > 0.5) {
      this.tween = { from: this.pos, to: Math.round(pos), t0: this.now() };
      this.schedule();
      return;
    }
    this.tween = null;
    this.pos = Math.round(pos);
    this.publish();
  }

  /** Seek to the stored checkpoint nearest to training step `step`. */
  seekStep(step, opts) { if (this.steps) this.seek(nearest(this.steps, step), opts); }

  /** Pause and move by n checkpoints. */
  stepBy(n) { this.seek(Math.round(this.pos) + n); }

  setSpeed(x) { this.speed = x; this.emit(); }
  setLoop(mode) { this.loop = mode; this.emit(); }

  /** Play only between checkpoint indices lo and hi (a preset name or [lo, hi]); play() starts at lo when outside. */
  setRange(r) {
    const range = typeof r === "string" ? this.presets[r] || this.presets.whole : r;
    this.range = [Math.max(0, range[0]), Math.min(this.steps.length - 1, range[1])];
    this.rangeName = typeof r === "string" && this.presets[r] ? r : null;
    this.emit();
  }

  /** Select a preset range (e.g. "circuit") and move to its start. */
  jumpTo(name) {
    this.setRange(name);
    this.seek(this.range[0], { tween: true });
  }

  schedule() { if (!this.frame) { this.frame = true; this.raf((t) => { this.frame = null; this.tick(t ?? this.now()); }); } }

  /** One animation frame: finish a seek tween, or advance playback by the elapsed time. */
  tick(t) {
    if (this.tween) {
      const { from, to, t0 } = this.tween, f = Math.min((t - t0) / SEEK_MS, 1);
      this.pos = from + (to - from) * ease(f);
      if (f >= 1) { this.tween = null; this.pos = to; }
      this.publish();
      if (this.tween) this.schedule();
      return;
    }
    if (!this.playing) return;
    const dt = this.last == null ? 0 : Math.min((t - this.last) / 1000, MAX_DT);
    this.last = t;
    const [lo, hi] = this.range;
    if (this.reduced()) { // whole checkpoints only, at a calm fixed rate
      this.acc += dt * REDUCED_RATE;
      const n = Math.floor(this.acc);
      this.acc -= n;
      this.pos = this.pos + n > hi ? (this.loop === "once" ? hi : lo) : this.pos + n;
      if (this.loop === "once" && this.pos >= hi) { this.pause(); return; }
    } else {
      this.state = advance({ ...this.state, pos: this.pos }, dt, { lo, hi, speed: this.speed, loop: this.loop });
      this.pos = this.state.pos;
      if (this.state.done) { this.pause(); return; }
    }
    this.publish();
    this.schedule();
  }

  /** Write the current position into the store (interpolated step only while moving with motion allowed). */
  publish() {
    const { step, stepf } = stepAt(this.steps, this.pos);
    const moving = this.playing || !!this.tween;
    this.published = step;
    this.store.set({ step, stepf: moving ? (this.reduced() ? step : stepf) : null, playing: this.playing });
    this.emit();
  }
}
