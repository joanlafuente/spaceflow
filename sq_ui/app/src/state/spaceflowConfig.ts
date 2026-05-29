export const DEFAULT_LOW_CONTROL_BBOX_MARGIN = 0.05;
export const LOW_CONTROL_BBOX_MARGIN_MIN = 0;
export const LOW_CONTROL_BBOX_MARGIN_MAX = 0.5;
export const LOW_CONTROL_BBOX_MARGIN_STEP = 0.01;

export function clampLowControlBBoxMargin(value: number): number {
  if (!Number.isFinite(value)) return DEFAULT_LOW_CONTROL_BBOX_MARGIN;
  return Math.min(
    LOW_CONTROL_BBOX_MARGIN_MAX,
    Math.max(LOW_CONTROL_BBOX_MARGIN_MIN, value),
  );
}
