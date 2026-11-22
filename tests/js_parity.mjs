// Reads {pos: [[x,y]...], act: [[ax,ay]...]} from argv[2], writes next positions and frames (base64) to stdout.
import { readFileSync } from "node:fs";
import { render, stepPos } from "../web/src/env.js";
const inp = JSON.parse(readFileSync(process.argv[2], "utf8"));
const next = inp.pos.map((p, k) => stepPos(p, inp.act[k]));
const frames = inp.pos.map((p) => Buffer.from(render(p)).toString("base64"));
process.stdout.write(JSON.stringify({ next, frames }));
