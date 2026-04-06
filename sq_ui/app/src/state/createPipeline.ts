import type { Primitive } from './store';
import { generateWithSuperdec, type SuperdecGenerateOptions } from './superdec';
import { generatePointCloudFromText, type TrellisGenerateOptions } from './trellis';

export interface CreateFromTextViaSuperdecOptions {
  name?: string;
  trellis?: Omit<TrellisGenerateOptions, 'prompt' | 'name'>;
  superdec?: Omit<SuperdecGenerateOptions, 'file' | 'name'>;
}

export interface CreateFromTextViaSuperdecResult {
  primitives: Primitive[];
  pointCount: number;
  primitiveCount: number;
  trellisRunId: string;
  superdecRunId: string;
  metadata?: {
    trellis?: Record<string, unknown>;
    superdec?: Record<string, unknown>;
  };
}

function sanitizeName(name: string): string {
  return name.trim().replace(/[^a-zA-Z0-9_-]/g, '_') || 'superquadrics';
}

export async function createFromTextViaSuperdec(
  prompt: string,
  options?: CreateFromTextViaSuperdecOptions
): Promise<CreateFromTextViaSuperdecResult> {
  const baseName = sanitizeName(options?.name || prompt);
  const trellis = await generatePointCloudFromText({
    prompt,
    name: baseName,
    pointCount: options?.trellis?.pointCount ?? 4096,
    normalize: options?.trellis?.normalize ?? true,
    preferMesh: options?.trellis?.preferMesh ?? false,
    seed: options?.trellis?.seed ?? 1,
    sparseSteps: options?.trellis?.sparseSteps,
    slatSteps: options?.trellis?.slatSteps,
    cfgStrength: options?.trellis?.cfgStrength,
    slatCfgStrength: options?.trellis?.slatCfgStrength,
  });

  const superdec = await generateWithSuperdec({
    file: trellis.file,
    name: baseName,
    zUp: options?.superdec?.zUp ?? false,
    normalize: options?.superdec?.normalize ?? true,
    lmOptimization: options?.superdec?.lmOptimization ?? false,
    maxPrimitives: options?.superdec?.maxPrimitives ?? 16,
    existThreshold: options?.superdec?.existThreshold ?? 0.5,
  });

  return {
    primitives: superdec.primitives,
    pointCount: trellis.pointCount,
    primitiveCount: superdec.primitiveCount,
    trellisRunId: trellis.runId,
    superdecRunId: superdec.runId,
    metadata: {
      trellis: trellis.metadata,
      superdec: superdec.metadata,
    },
  };
}
