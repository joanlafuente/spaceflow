/**
 * Source numbers match examples/superquadrics/*_sq.npz (pipeline / fit convention).
 * Those assets are effectively Z-up; we convert world pose to Y-up (Three.js) for the LLM
 * so few-shot examples match the editor.
 */
import { applyWorldBasisZUpToYUp } from '../mesh/npzImport';
import { eulerToMatrix, matrixToEuler } from './rotation';

function r6(n: number): number {
  return Math.round(n * 1e6) / 1e6;
}

type Prim = {
  name: string;
  scales: readonly [number, number, number];
  shapes: readonly [number, number];
  translation: readonly [number, number, number];
  eulerDeg: readonly [number, number, number];
};

function primZUpToYUp(p: Prim) {
  const R = eulerToMatrix([p.eulerDeg[0], p.eulerDeg[1], p.eulerDeg[2]]);
  const { rotation, translation } = applyWorldBasisZUpToYUp(R, [
    p.translation[0],
    p.translation[1],
    p.translation[2],
  ]);
  const eulerDeg = matrixToEuler(rotation);
  return {
    name: p.name,
    scales: [p.scales[0], p.scales[1], p.scales[2]] as [number, number, number],
    shapes: [p.shapes[0], p.shapes[1]] as [number, number],
    translation: [r6(translation[0]), r6(translation[1]), r6(translation[2])] as [number, number, number],
    eulerDeg: [r6(eulerDeg[0]), r6(eulerDeg[1]), r6(eulerDeg[2])] as [number, number, number],
  };
}

function presetZUpToYUp<const T extends { primitives: readonly Prim[] }>(preset: T) {
  return {
    primitives: preset.primitives.map(primZUpToYUp),
  };
}

/** Raw Z-up (matches NPZ / extract script) — for reference only; use *ForPrompt exports in the app. */
export const chairSqZUpForPrompt = {
  primitives: [
    { name: 'chair_sq_0', scales: [0.020238, 0.246944, 0.012734] as const, shapes: [0.553361, 0.400066] as const, translation: [0.243503, -0.24232, -0.200469] as const, eulerDeg: [-88.931828, -3.839556, -91.71084] as const },
    { name: 'chair_sq_1', scales: [0.018802, 0.45529, 0.009253] as const, shapes: [0.543329, 0.400133] as const, translation: [0.242976, 0.238389, 0.014216] as const, eulerDeg: [-89.528466, -0.892911, -90.90105] as const },
    { name: 'chair_sq_2', scales: [0.004843, 0.021809, 0.234984] as const, shapes: [0.400071, 0.969185] as const, translation: [-0.02767, 0.24768, -0.022461] as const, eulerDeg: [-90.112776, -2.065676, -90.101749] as const },
    { name: 'chair_sq_3', scales: [0.018814, 0.437928, 0.007229] as const, shapes: [0.402872, 0.400001] as const, translation: [-0.244636, 0.243751, 0.012136] as const, eulerDeg: [-90.728434, -0.352932, -91.552831] as const },
    { name: 'chair_sq_4', scales: [0.242746, 0.044417, 0.003787] as const, shapes: [0.400367, 0.4] as const, translation: [-0.241443, 0.007865, -0.005142] as const, eulerDeg: [-90.704968, 1.214688, -91.139648] as const },
    { name: 'chair_sq_5', scales: [0.019444, 0.264667, 0.012161] as const, shapes: [0.552516, 0.40039] as const, translation: [-0.245519, -0.239169, -0.216552] as const, eulerDeg: [-90.847656, -1.692174, -90.669375] as const },
    { name: 'chair_sq_6', scales: [0.004798, 0.02703, 0.200347] as const, shapes: [0.400068, 0.424382] as const, translation: [0.005322, -0.230983, -0.007053] as const, eulerDeg: [-89.94428, -0.584911, -89.821639] as const },
    { name: 'chair_sq_7', scales: [0.007744, 0.073928, 0.267075] as const, shapes: [0.400089, 0.400955] as const, translation: [-0.006302, 0.256255, 0.426675] as const, eulerDeg: [-89.380597, -5.019466, -90.141117] as const },
    { name: 'chair_sq_8', scales: [0.269015, 0.021576, 0.252046] as const, shapes: [0.400259, 0.4] as const, translation: [0.003451, -0.005037, 0.038579] as const, eulerDeg: [-89.974583, 0.487424, -90.990189] as const },
    { name: 'chair_sq_9', scales: [0.221396, 0.026363, 0.004318] as const, shapes: [0.403497, 0.400006] as const, translation: [0.250641, 0.009467, -0.014289] as const, eulerDeg: [-92.449918, -2.156727, -89.431766] as const },
  ],
} as const;

export const sofaSqZUpForPrompt = {
  primitives: [
    { name: 'sofa_sq_0', scales: [0.165236, 0.103787, 0.022057] as const, shapes: [0.400005, 0.4] as const, translation: [0.349773, -0.027548, -0.047916] as const, eulerDeg: [-90.094917, 0.639484, -89.512179] as const },
    { name: 'sofa_sq_1', scales: [0.042, 0.144, 0.336] as const, shapes: [0.40001, 0.400003] as const, translation: [0.006061, 0.122474, 0.021664] as const, eulerDeg: [-90.501743, -4.953707, -89.903013] as const },
    { name: 'sofa_sq_2', scales: [0.154028, 0.053498, 0.336] as const, shapes: [0.40011, 0.400002] as const, translation: [0.008166, -0.022785, -0.074396] as const, eulerDeg: [-90.331879, 1.430377, -89.765973] as const },
    { name: 'sofa_sq_3', scales: [0.165236, 0.103787, 0.022057] as const, shapes: [0.400005, 0.4] as const, translation: [-0.323016, -0.031767, -0.048812] as const, eulerDeg: [-90.094917, 0.639484, -89.512179] as const },
  ],
} as const;

export const planeSqZUpForPrompt = {
  primitives: [
    { name: 'plane_sq_0', scales: [0.174, 0.437928, 0.042] as const, shapes: [0.402872, 0.400001] as const, translation: [-0.016253, -0.380808, 0.257946] as const, eulerDeg: [-90.804213, -25.074484, -91.21648] as const },
    { name: 'plane_sq_1', scales: [0.184, 0.052, 1.0] as const, shapes: [0.400068, 0.424382] as const, translation: [0.008036, -1.10289, 0.001848] as const, eulerDeg: [-89.944283, -0.584909, -89.821636] as const },
    { name: 'plane_sq_2', scales: [0.184, 0.042, 0.438] as const, shapes: [0.400089, 0.400955] as const, translation: [-0.007566, -0.257128, 0.471766] as const, eulerDeg: [-89.380595, -5.019466, -90.141117] as const },
    { name: 'plane_sq_3', scales: [0.794, 0.336, 0.276] as const, shapes: [1.06, 0.88] as const, translation: [-0.015726, -1.114554, 0.029138] as const, eulerDeg: [-89.974586, 0.487427, -90.990188] as const },
  ],
} as const;

/** Y-up — use in system prompt and anywhere the LLM should see editor-aligned poses */
export const chairSqForPrompt = presetZUpToYUp(chairSqZUpForPrompt);
export const sofaSqForPrompt = presetZUpToYUp(sofaSqZUpForPrompt);
export const planeSqForPrompt = presetZUpToYUp(planeSqZUpForPrompt);

/**
 * Hand-authored Y-up layout (not from NPZ). Shows simple eulerDeg, stacked thin parts, hinge tilt on the lid.
 * Validated against the editor; included so smaller local LLMs mimic this style.
 */
export const laptopSqForPrompt = {
  primitives: [
    { name: 'laptop_base', scales: [0.4, 0.02, 0.3] as [number, number, number], shapes: [0.2, 0.2] as [number, number], translation: [0, 0.01, 0] as [number, number, number], eulerDeg: [0, 0, 0] as [number, number, number] },
    { name: 'laptop_screen_lid', scales: [0.4, 0.28, 0.01] as [number, number, number], shapes: [0.2, 0.2] as [number, number], translation: [0, 0.25, -0.28] as [number, number, number], eulerDeg: [-15, 0, 0] as [number, number, number] },
    { name: 'keyboard_area', scales: [0.35, 0.005, 0.15] as [number, number, number], shapes: [0.1, 0.1] as [number, number], translation: [0, 0.025, -0.05] as [number, number, number], eulerDeg: [0, 0, 0] as [number, number, number] },
    { name: 'trackpad', scales: [0.1, 0.005, 0.06] as [number, number, number], shapes: [0.2, 0.2] as [number, number], translation: [0, 0.025, 0.15] as [number, number, number], eulerDeg: [0, 0, 0] as [number, number, number] },
    { name: 'hinge_bar', scales: [0.38, 0.015, 0.015] as [number, number, number], shapes: [2, 2] as [number, number], translation: [0, 0.02, -0.15] as [number, number, number], eulerDeg: [0, 0, 0] as [number, number, number] },
  ],
} as const;
