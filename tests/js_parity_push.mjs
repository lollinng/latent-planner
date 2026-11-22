// Reads {states: [[ax,ay,bx,by]...], act: [[dx,dy]...]} from argv[2]; writes next states and frames (base64).
import { readFileSync } from "node:fs";
import { render, stepState } from "../web/src/env_push.js";
const inp = JSON.parse(readFileSync(process.argv[2], "utf8"));
const next = inp.states.map((s, k) => stepState(s, inp.act[k]));
const frames = inp.states.map((s) => Buffer.from(render(s)).toString("base64"));
process.stdout.write(JSON.stringify({ next, frames }));
