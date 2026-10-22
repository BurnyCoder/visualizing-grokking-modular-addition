// index.js — registry: widget id → lazy loader of its implementation (same pattern as figures/index.js).
// The site build parses the quoted ids below to check that every {{widget:id}} in the content exists.
const mod = (path, id) => () => import(path).then((m) => {
  if (!m.widgets[id]) throw new Error(`widget ${id} missing in ${path}`);
  return m.widgets[id];
});

export const WIDGETS = {
  "clock": mod("./basics.js", "clock"),
  "dot-product": mod("./basics.js", "dot-product"),
  "matmul-anim": mod("./basics.js", "matmul-anim"),
  "softmax": mod("./basics.js", "softmax"),
  "gd-toy": mod("./basics.js", "gd-toy"),
  "adamw-toy": mod("./basics.js", "adamw-toy"),
  "gini-toy": mod("./basics.js", "gini-toy"),
  "unit-circle": mod("./trig.js", "unit-circle"),
  "phasor-wave": mod("./trig.js", "phasor-wave"),
  "angle-addition": mod("./trig.js", "angle-addition"),
  "interference-toy": mod("./trig.js", "interference-toy"),
  "add-angles": mod("./trig.js", "add-angles"),
  "trig-outer-product": mod("./trig.js", "trig-outer-product"),
  "dft-p5": mod("./fourier.js", "dft-p5"),
  "periodic-sparse": mod("./fourier.js", "periodic-sparse"),
  "basis-2d": mod("./fourier.js", "basis-2d"),
  "relu-multiplier": mod("./fourier.js", "relu-multiplier"),
  "scrubber": mod("./model.js", "scrubber"),
  "forward-walkthrough": mod("./model.js", "forward-walkthrough"),
};
