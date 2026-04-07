import { eulerToMatrix } from './rotation';
import type { Primitive } from './store';
import { ollamaChatUrl } from './devServiceUrl';
import { chairSqForPrompt, laptopSqForPrompt, planeSqForPrompt, sofaSqForPrompt } from './verifiedSqPresets';

function ollamaUrl(): string {
  return ollamaChatUrl('http://localhost:11434/api/chat');
}
/** Ollama tag. Default `gemma4:e2b` (~7 GB). Other Gemma 4 tags: `gemma4:e4b` / `gemma4:latest` (larger), `gemma4:e2b-it-q8_0` (higher-quality quant). */
const MODEL = 'gemma4:e2b';

const LLM_VERIFIED_EXAMPLES = `
REFERENCE FITS (real data — large eulerDeg is normal here)
  The chair/sofa/plane JSON below is in the editor’s Y-up frame (converted from Z-up fits). Parts can have eulerDeg far from [0,0,0].
  When you GENERATE NEW objects from scratch, do NOT blindly copy those euler patterns: prefer clear translations and eulerDeg [0,0,0] unless a tilt is obviously needed.

AUTHORING EXAMPLE (hand layout — simple euler, good template)
  The laptop below was authored in Y-up: thin keyboard/trackpad slabs slightly above the base, screen lid translated back/up with a small negative X euler for an open hinge, hinge bar as shapes [2,2] cylinder-like. Mimic this style for articulated objects.

Few-shots: three NPZ-derived fits (chair / sofa / plane) plus the laptop authoring example.

User: "a chair"
${JSON.stringify(chairSqForPrompt)}

User: "a sofa"
${JSON.stringify(sofaSqForPrompt)}

User: "an airplane"
${JSON.stringify(planeSqForPrompt)}

User: "a laptop"
${JSON.stringify(laptopSqForPrompt)}
`;

const SQ_PARAMETER_GUIDE = `
SUPERQUADRIC PARAMETERS — EDITOR-VERIFIED CONVENTION (Y-up)

Each primitive: { name, scales, shapes, translation, eulerDeg }
Keep eulerDeg aligned with the intended pose (the app stores a 3×3 rotation matrix derived from eulerDeg).

scales [A, B, C] — Half-extents along local axes after rotation (total size ≈ 2× each half-extent).

shapes [e1, e2] — Exponents in this editor's parametric superquadric (see mesh code). They control corner
  sharpness and silhouette; very small e1 (e.g. 0.05) is used with specific e2 and rotation for cylinders and cubes.

CANONICAL BASIC SHAPES (match the UI "Add Primitive" presets — copy these tuples exactly)
  Ball:
    scales [1,1,1], shapes [1,1], eulerDeg [0,0,0]
  Ellipsoid (elongated ball; same exponents, different half-extents):
    scales [0.5,0.5,1], shapes [1,1], eulerDeg [0,0,0]
  Cylinder (upright along world Y):
    scales [1,1,1], shapes [0.05,1], eulerDeg [90,0,0]
  Cube (working pose for a box-like superquadric):
    scales [1,1,1], shapes [0.05,0.05], eulerDeg [90,0,0]
  Astroid / star-shaped superquadric:
    scales [1,1,1], shapes [4,4], eulerDeg [0,0,0]

ROTATION
  Cylinder and cube use eulerDeg [90,0,0] so the intrinsic surface lines up usefully in Y-up. Changing rotation
  without retuning shapes/scales can change the apparent category.

COMPOSITE PARTS
  Flat slabs: one small half-extent, e.g. scales [0.8,0.02,0.5] with cube-like shapes [0.05,0.05] and eulerDeg [90,0,0].
  Vertical legs/columns: cylinder recipe [0.05,1] + [90,0,0], then increase the Y half-extent (middle scale).
  Fitted NPZ examples in the few-shots may use other e1,e2 (~0.4); treat those as data, not as the basic presets.

IMPLICIT FORM (reference)
  (|x/A|^(2/e2) + |y/B|^(2/e2))^(e2/e1) + |z/C|^(2/e1) = 1

translation: [x, y, z] — World center. Y-up.
eulerDeg: [rx, ry, rz] — ZYX Euler degrees.
`;

const SYSTEM_PROMPT = `You are a 3D Geometry Expert decomposing objects into Superquadric Primitives.
Reply ONLY with valid JSON: {"primitives":[...]} matching the schema. No markdown.

COORDINATE SYSTEM (Y-Up)
- Y = Up/Down (Ground is Y=0).
- X = Right/Left.
- Z = Forward/Back.
` + SQ_PARAMETER_GUIDE + `
DECOMPOSITION STRATEGY
1. Define the "Core" (e.g., car body, chair seat). Place it near [0, Core.scale.y, 0].
2. Calculate "Child" positions relative to the Core's dimensions to ensure they touch.
   * Stacking Rule: To place Part B on top of Part A: B.translation.y = A.translation.y + A.scale.y + B.scale.y.
3. Use Symmetry: For pairs (legs, wheels, wings), use ±X translations with identical scales.
4. Overlap: Ensure parts overlap by ~5% (0.01-0.02 units) so they appear connected.

RULES
- Names must be descriptive (e.g., "front_left_leg", not "part_1").
- 4-10 primitives per object for detail.
- For basic shapes use the canonical eulerDeg from the guide ([0,0,0] for ball/astroid; [90,0,0] for cylinder/cube). Add non-zero euler only for hinges, tilts, or re-orienting parts.
` + LLM_VERIFIED_EXAMPLES;

const EDIT_SYSTEM_PROMPT = `You EDIT an existing superquadric scene from the user's instruction.
Reply ONLY with valid JSON: {"primitives":[...]} — same schema as creation. No markdown, no code fences.
` + SQ_PARAMETER_GUIDE + `
The user message contains CURRENT_SCENE_JSON (Y-up, half-extents), EDIT_INSTRUCTION, and optional FOCUS_PART_NAMES.
If an IMAGE is attached, it is a screenshot of the current 3D viewport (Y-up; parts are color-coded). Use it together with the JSON to understand what to change. The JSON numbers remain authoritative for geometry; the image helps with spatial intent (which part is which, overall proportions).

CRITICAL — YOU MUST ACTUALLY EDIT
- Do NOT return CURRENT_SCENE_JSON unchanged. You MUST change scales, translation, eulerDeg, and/or shapes on the affected parts with clear, non-trivial numeric changes.
- "More round / ball-like" → shapes toward [1,1] with eulerDeg [0,0,0]; "cylindrical" → [0.05,1] with eulerDeg [90,0,0]; "boxy / cube-like" → [0.05,0.05] with eulerDeg [90,0,0]; "star / astroid" → toward [4,4].
- "Taller / wider / longer" → change the relevant scale axis AND adjust translations so parts stay connected (stacking rule: child.y = parent.y + parent.scale[1] + child.scale[1]).
- Reclining / hinge / articulated motion: tilt parts with eulerDeg (often X rotation for backrest vs seat); shift translations so touching faces stay aligned.
- Adding new parts: give them descriptive names and sensible geometry.
- Removing parts: simply omit them from the output.

FOCUS CONSTRAINT (HARD RULE)
- If FOCUS_PART_NAMES is non-empty, you may ONLY modify the listed parts. All other parts MUST be returned with EXACTLY their original values — do not change a single number on non-focused parts. This is enforced programmatically; any changes to non-focused parts will be discarded.
- If FOCUS_PART_NAMES is empty ([]), you may modify any or all parts freely.

RULES
- Output the FULL updated scene: one object per remaining part, with updated numbers where needed.
- Preserve part "name" strings exactly when keeping a part; only change names if splitting/merging.
- Do not add commentary outside JSON.

COORDINATE SYSTEM: Y-up, X right/left, Z forward/back. eulerDeg is [rx, ry, rz] in degrees (ZYX order used by the editor).`;

const RESPONSE_SCHEMA = {
  type: 'object',
  properties: {
    primitives: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          name: { type: 'string' },
          scales: { type: 'array', items: { type: 'number' } },
          shapes: { type: 'array', items: { type: 'number' } },
          translation: { type: 'array', items: { type: 'number' } },
          eulerDeg: { type: 'array', items: { type: 'number' } },
        },
        required: ['name', 'scales', 'shapes', 'translation', 'eulerDeg'],
      },
    },
  },
  required: ['primitives'],
};

interface RawPrimitive {
  name: string;
  scales: number[];
  shapes: number[];
  translation: number[];
  eulerDeg: number[];
}

let genCounter = 0;

function toPrimitive(raw: RawPrimitive): Primitive {
  const euler: [number, number, number] = [
    raw.eulerDeg[0] ?? 0,
    raw.eulerDeg[1] ?? 0,
    raw.eulerDeg[2] ?? 0,
  ];
  return {
    id: `gen_${++genCounter}_${Date.now()}`,
    name: raw.name || `Part ${genCounter}`,
    visible: true,
    scales: [
      Math.max(0.01, raw.scales[0] ?? 1),
      Math.max(0.01, raw.scales[1] ?? 1),
      Math.max(0.01, raw.scales[2] ?? 1),
    ],
    shapes: [
      Math.max(0.05, raw.shapes[0] ?? 2),
      Math.max(0.05, raw.shapes[1] ?? 2),
    ],
    translation: [
      raw.translation[0] ?? 0,
      raw.translation[1] ?? 0,
      raw.translation[2] ?? 0,
    ],
    rotation: eulerToMatrix(euler),
    eulerDeg: euler,
  };
}

/** Strip ids — what the LLM sees for the current scene. */
function roundForLlm(value: number): number {
  return Math.round(value * 1000) / 1000;
}

export function primitivesToLlmJson(primitives: Primitive[]) {
  return {
    primitives: primitives.map(p => ({
      name: p.name,
      scales: p.scales.map(roundForLlm),
      shapes: p.shapes.map(roundForLlm),
      translation: p.translation.map(roundForLlm),
      eulerDeg: p.eulerDeg.map(roundForLlm),
    })),
  };
}

/**
 * Match LLM output to previous primitives by name (first match wins per output row).
 * New names get fresh ids from toPrimitive.
 * Parts present in `previous` but missing from the model output are appended unchanged (avoids accidental scene loss).
 */
export function mergeEditedPrimitives(previous: Primitive[], rawList: RawPrimitive[]): Primitive[] {
  const pool = [...previous];
  const out: Primitive[] = [];
  for (const raw of rawList) {
    const euler: [number, number, number] = [
      raw.eulerDeg[0] ?? 0,
      raw.eulerDeg[1] ?? 0,
      raw.eulerDeg[2] ?? 0,
    ];
    const idx = pool.findIndex(p => p.name === raw.name);
    if (idx >= 0) {
      const prev = pool.splice(idx, 1)[0];
      out.push({
        ...prev,
        scales: [
          Math.max(0.01, raw.scales[0] ?? 1),
          Math.max(0.01, raw.scales[1] ?? 1),
          Math.max(0.01, raw.scales[2] ?? 1),
        ],
        shapes: [
          Math.max(0.05, raw.shapes[0] ?? 2),
          Math.max(0.05, raw.shapes[1] ?? 2),
        ],
        translation: [
          raw.translation[0] ?? 0,
          raw.translation[1] ?? 0,
          raw.translation[2] ?? 0,
        ],
        eulerDeg: euler,
        rotation: eulerToMatrix(euler),
      });
    } else {
      out.push(toPrimitive(raw));
    }
  }
  for (const p of pool) {
    out.push(p);
  }
  return out;
}

const TIMEOUT_MS = 120_000;

/** Ollama sometimes returns 200 with no assistant text (cold load, schema+vision quirks). */
function extractAssistantContent(raw: unknown): string {
  if (!raw || typeof raw !== 'object') return '';
  const message = (raw as { message?: unknown }).message;
  if (!message || typeof message !== 'object') return '';
  const c = (message as { content?: unknown }).content;
  if (typeof c === 'string') return c.trim();
  if (Array.isArray(c)) {
    return c
      .map(part => {
        if (typeof part === 'string') return part;
        if (part && typeof part === 'object' && 'text' in part) {
          const t = (part as { text?: unknown }).text;
          return typeof t === 'string' ? t : '';
        }
        return '';
      })
      .join('')
      .trim();
  }
  return '';
}

function describeOllamaEmptyResponse(json: unknown): string {
  if (!json || typeof json !== 'object') return 'response was not JSON';
  const o = json as Record<string, unknown>;
  const bits: string[] = [];
  if ('done' in o) bits.push(`done=${String(o.done)}`);
  if ('done_reason' in o) bits.push(`done_reason=${String(o.done_reason)}`);
  if ('eval_count' in o) bits.push(`eval_count=${String(o.eval_count)}`);
  if ('prompt_eval_count' in o) bits.push(`prompt_eval_count=${String(o.prompt_eval_count)}`);
  if ('error' in o) bits.push(`error=${JSON.stringify(o.error).slice(0, 280)}`);
  const msg = o.message;
  if (msg && typeof msg === 'object') {
    const m = msg as Record<string, unknown>;
    if (typeof m.content === 'string') bits.push(`content_len=${m.content.length}`);
  }
  return bits.join('; ') || JSON.stringify(json).slice(0, 400);
}

async function fetchChatJson(body: object): Promise<unknown> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
  try {
    const res = await fetch(ollamaUrl(), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    if (!res.ok) {
      const text = await res.text().catch(() => '');
      throw new Error(`Ollama returned ${res.status}: ${text.slice(0, 200)}`);
    }
    return await res.json();
  } catch (err) {
    if (err instanceof Error && err.message.startsWith('Ollama returned')) throw err;
    if (err instanceof DOMException && err.name === 'AbortError') {
      throw new Error('Generation timed out (>2 min). The model may be too slow or still loading. Try again.');
    }
    throw new Error(
      'Cannot reach Ollama. Make sure it is running:\n  ollama serve\n  ollama pull gemma4:e2b'
    );
  } finally {
    clearTimeout(timer);
  }
}

async function chatOllama(
  system: string,
  user: string,
  opts?: { userImagesBase64?: string[] }
): Promise<string> {
  const userMsg: {
    role: 'user';
    content: string;
    images?: string[];
  } = { role: 'user', content: user };
  const imgs = opts?.userImagesBase64?.filter(Boolean);
  if (imgs && imgs.length > 0) {
    userMsg.images = imgs;
  }

  // Full JSON Schema + vision can yield empty `message.content` on some Ollama/model builds; loose "json" still matches our prompts.
  const hasImages = !!(imgs && imgs.length > 0);
  const body = {
    model: MODEL,
    messages: [{ role: 'system' as const, content: system }, userMsg],
    stream: false,
    format: (hasImages ? 'json' : RESPONSE_SCHEMA) as typeof RESPONSE_SCHEMA | 'json',
  };

  let json: unknown = await fetchChatJson(body);
  let content = extractAssistantContent(json);
  if (!content) {
    await new Promise<void>(r => setTimeout(r, 1500));
    json = await fetchChatJson(body);
    content = extractAssistantContent(json);
  }
  if (!content) {
    throw new Error(
      `Empty response from Ollama (${describeOllamaEmptyResponse(json)}). ` +
        'Often fixed by retrying after the model loads, or by turning off “Include viewport screenshot” if the multimodal path misbehaves.'
    );
  }
  return content;
}

const GEOM_EPS = 1e-5;
function near(a: number, b: number): boolean {
  return Math.abs(a - b) < GEOM_EPS;
}
function vec3eq(
  a: readonly number[] | undefined,
  b: readonly number[] | undefined
): boolean {
  for (let i = 0; i < 3; i++) {
    if (!near(a?.[i] ?? 0, b?.[i] ?? 0)) return false;
  }
  return true;
}
function vec2eq(
  a: readonly number[] | undefined,
  b: readonly number[] | undefined
): boolean {
  for (let i = 0; i < 2; i++) {
    if (!near(a?.[i] ?? 0, b?.[i] ?? 0)) return false;
  }
  return true;
}

/** True if every part has the same geometry fields (by name). */
function primitivesGeometryEqual(a: Primitive[], b: Primitive[]): boolean {
  if (a.length !== b.length) return false;
  const byName = new Map(b.map(p => [p.name, p]));
  for (const p of a) {
    const q = byName.get(p.name);
    if (!q) return false;
    if (!vec3eq(p.scales, q.scales)) return false;
    if (!vec2eq(p.shapes, q.shapes)) return false;
    if (!vec3eq(p.translation, q.translation)) return false;
    if (!vec3eq(p.eulerDeg, q.eulerDeg)) return false;
  }
  return true;
}

/** Strip markdown fences and leading prose so JSON.parse succeeds. */
function extractJsonPayload(s: string): string {
  let t = s.trim();
  if (t.startsWith('```')) {
    const firstNl = t.indexOf('\n');
    if (firstNl >= 0) t = t.slice(firstNl + 1);
    const close = t.lastIndexOf('```');
    if (close >= 0) t = t.slice(0, close);
  }
  t = t.trim();
  try {
    JSON.parse(t);
    return t;
  } catch {
    const lb = t.indexOf('{');
    const rb = t.lastIndexOf('}');
    if (lb >= 0 && rb > lb) return t.slice(lb, rb + 1);
    return t;
  }
}

function parsePrimitivesJson(content: string): RawPrimitive[] {
  const payload = extractJsonPayload(content);
  let parsed: { primitives: RawPrimitive[] };
  try {
    parsed = JSON.parse(payload);
  } catch {
    throw new Error(`Model returned invalid JSON: ${content.slice(0, 200)}`);
  }
  if (!parsed.primitives || !Array.isArray(parsed.primitives) || parsed.primitives.length === 0) {
    throw new Error('Model returned no primitives');
  }
  return parsed.primitives;
}

export async function generateFromText(prompt: string): Promise<Primitive[]> {
  const content = await chatOllama(SYSTEM_PROMPT, prompt);
  return parsePrimitivesJson(content).map(toPrimitive);
}

export interface EditFromTextOptions {
  /** Part names that may be modified; all others are locked. Empty = everything editable. */
  focusNames?: string[];
  /**
   * Raw PNG base64 strings (no `data:` prefix) for multimodal models (e.g. Gemma 4).
   * Typically one screenshot of the viewport.
   */
  viewportImagesBase64?: string[];
}

/**
 * Hard-lock: restore geometry of non-focused parts from `original`.
 * Only parts whose name IS in `focusNames` keep their model-edited values.
 */
function enforceFocusLock(
  result: Primitive[],
  original: Primitive[],
  focusNames: string[]
): Primitive[] {
  if (focusNames.length === 0) return result;
  const focusSet = new Set(focusNames);
  const origByName = new Map(original.map(p => [p.name, p]));
  return result.map(p => {
    if (focusSet.has(p.name)) return p;
    const orig = origByName.get(p.name);
    if (!orig) return p;
    return { ...orig };
  });
}

export interface EditFromTextResult {
  primitives: Primitive[];
  /** True when merged geometry matches `current` (model echoed the scene or made no numeric edits). */
  unchanged: boolean;
}

export async function editFromText(
  instruction: string,
  current: Primitive[],
  options?: EditFromTextOptions
): Promise<EditFromTextResult> {
  if (current.length === 0) {
    throw new Error('Nothing to edit — add or generate primitives first.');
  }
  const focus = (options?.focusNames ?? []).filter(Boolean);
  const hasImage = (options?.viewportImagesBase64?.length ?? 0) > 0;
  const userContent = [
    ...(hasImage
      ? ['An IMAGE of the current 3D viewport is attached (color-coded parts).', '']
      : []),
    'CURRENT_SCENE_JSON:',
    JSON.stringify(primitivesToLlmJson(current)),
    '',
    'EDIT_INSTRUCTION:',
    instruction.trim(),
    '',
    focus.length > 0
      ? 'FOCUS_PART_NAMES (ONLY modify these parts — all others MUST keep their exact original values):'
      : 'FOCUS_PART_NAMES (empty — you may modify any part):',
    JSON.stringify(focus),
  ].join('\n');

  const content = await chatOllama(EDIT_SYSTEM_PROMPT, userContent, {
    userImagesBase64: options?.viewportImagesBase64,
  });
  const rawList = parsePrimitivesJson(content);
  let merged = mergeEditedPrimitives(current, rawList);
  merged = enforceFocusLock(merged, current, focus);
  const unchanged = primitivesGeometryEqual(current, merged);
  return { primitives: merged, unchanged };
}
