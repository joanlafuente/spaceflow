import type { SVGProps } from 'react';

type IconProps = SVGProps<SVGSVGElement> & {
  size?: number;
  strokeWidth?: number;
};

function IconBase({ size = 16, strokeWidth = 2, children, ...props }: IconProps) {
  return (
    <svg
      aria-hidden="true"
      fill="none"
      height={size}
      stroke="currentColor"
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth={strokeWidth}
      viewBox="0 0 24 24"
      width={size}
      {...props}
    >
      {children}
    </svg>
  );
}

export function UndoIcon(props: IconProps) {
  return (
    <IconBase {...props}>
      <path d="M9 14 4 9l5-5" />
      <path d="M4 9h9.5a6.5 6.5 0 1 1 0 13H11" />
    </IconBase>
  );
}

export function RedoIcon(props: IconProps) {
  return (
    <IconBase {...props}>
      <path d="m15 14 5-5-5-5" />
      <path d="M20 9h-9.5a6.5 6.5 0 1 0 0 13H13" />
    </IconBase>
  );
}

export function CircleIcon(props: IconProps) {
  return (
    <IconBase {...props}>
      <circle cx="12" cy="12" r="7" />
      <path d="M12 5v14" opacity="0.35" />
      <path d="M5 12h14" opacity="0.35" />
    </IconBase>
  );
}

export function TableIcon(props: IconProps) {
  return (
    <IconBase {...props}>
      <path d="M4 8h16" />
      <path d="M5 8l2 11" />
      <path d="M19 8l-2 11" />
      <path d="M9 8v11" />
      <path d="M15 8v11" />
      <path d="M7 19h10" />
    </IconBase>
  );
}

export function ChairIcon(props: IconProps) {
  return (
    <IconBase {...props}>
      <path d="M7 4h10v7H7z" />
      <path d="M6 13h12" />
      <path d="M8 13v7" />
      <path d="M16 13v7" />
      <path d="M5 20h4" />
      <path d="M15 20h4" />
    </IconBase>
  );
}

export function RotateIcon(props: IconProps) {
  return (
    <IconBase {...props}>
      <path d="M20 12a8 8 0 1 1-2.35-5.65" />
      <path d="M20 4v6h-6" />
    </IconBase>
  );
}

export function InvertControlIcon(props: IconProps) {
  return (
    <IconBase {...props}>
      <path d="M7 7h10" />
      <path d="m14 4 3 3-3 3" />
      <path d="M17 17H7" />
      <path d="m10 14-3 3 3 3" />
      <circle cx="7" cy="7" r="2.5" />
      <circle cx="17" cy="17" r="2.5" />
    </IconBase>
  );
}

export function SparklesIcon(props: IconProps) {
  return (
    <IconBase {...props}>
      <path d="M12 3l1.7 4.3L18 9l-4.3 1.7L12 15l-1.7-4.3L6 9l4.3-1.7z" />
      <path d="M5 15l.8 2.2L8 18l-2.2.8L5 21l-.8-2.2L2 18l2.2-.8z" />
      <path d="M19 14l.6 1.4L21 16l-1.4.6L19 18l-.6-1.4L17 16l1.4-.6z" />
    </IconBase>
  );
}

export function XIcon(props: IconProps) {
  return (
    <IconBase {...props}>
      <path d="M18 6 6 18" />
      <path d="m6 6 12 12" />
    </IconBase>
  );
}

export function SunIcon(props: IconProps) {
  return (
    <IconBase {...props}>
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2" />
      <path d="M12 20v2" />
      <path d="m4.93 4.93 1.41 1.41" />
      <path d="m17.66 17.66 1.41 1.41" />
      <path d="M2 12h2" />
      <path d="M20 12h2" />
      <path d="m6.34 17.66-1.41 1.41" />
      <path d="m19.07 4.93-1.41 1.41" />
    </IconBase>
  );
}

export function MoonIcon(props: IconProps) {
  return (
    <IconBase {...props}>
      <path d="M20.5 14.2A8.5 8.5 0 0 1 9.8 3.5 7 7 0 1 0 20.5 14.2z" />
    </IconBase>
  );
}

export function UploadIcon(props: IconProps) {
  return (
    <IconBase {...props}>
      <path d="M12 16V4" />
      <path d="m7 9 5-5 5 5" />
      <path d="M4 20h16" />
    </IconBase>
  );
}

export function DownloadIcon(props: IconProps) {
  return (
    <IconBase {...props}>
      <path d="M12 4v12" />
      <path d="m7 11 5 5 5-5" />
      <path d="M4 20h16" />
    </IconBase>
  );
}

export function AlertTriangleIcon(props: IconProps) {
  return (
    <IconBase {...props}>
      <path d="M10.3 4.5 2.5 18a2 2 0 0 0 1.7 3h15.6a2 2 0 0 0 1.7-3L13.7 4.5a2 2 0 0 0-3.4 0z" />
      <path d="M12 9v4" />
      <path d="M12 17h.01" />
    </IconBase>
  );
}

export function GripIcon(props: IconProps) {
  return (
    <IconBase {...props}>
      <path d="M9 5h.01" />
      <path d="M15 5h.01" />
      <path d="M9 12h.01" />
      <path d="M15 12h.01" />
      <path d="M9 19h.01" />
      <path d="M15 19h.01" />
    </IconBase>
  );
}

export function EyeIcon(props: IconProps) {
  return (
    <IconBase {...props}>
      <path d="M2.5 12s3.5-6 9.5-6 9.5 6 9.5 6-3.5 6-9.5 6-9.5-6-9.5-6z" />
      <circle cx="12" cy="12" r="3" />
    </IconBase>
  );
}

export function EyeOffIcon(props: IconProps) {
  return (
    <IconBase {...props}>
      <path d="m3 3 18 18" />
      <path d="M10.6 10.6a2 2 0 0 0 2.8 2.8" />
      <path d="M9.5 5.3A10.5 10.5 0 0 1 12 5c6 0 9.5 7 9.5 7a17 17 0 0 1-2.1 3.1" />
      <path d="M6.6 6.8C3.9 8.6 2.5 12 2.5 12s3.5 7 9.5 7c1.5 0 2.9-.4 4.1-1" />
    </IconBase>
  );
}

export function CopyIcon(props: IconProps) {
  return (
    <IconBase {...props}>
      <rect height="13" rx="2" width="13" x="8" y="8" />
      <path d="M5 16H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2v1" />
    </IconBase>
  );
}

export function TrashIcon(props: IconProps) {
  return (
    <IconBase {...props}>
      <path d="M3 6h18" />
      <path d="M8 6V4h8v2" />
      <path d="M19 6l-1 14H6L5 6" />
      <path d="M10 11v5" />
      <path d="M14 11v5" />
    </IconBase>
  );
}

export function PlusIcon(props: IconProps) {
  return (
    <IconBase {...props}>
      <path d="M12 5v14" />
      <path d="M5 12h14" />
    </IconBase>
  );
}

export function DiamondIcon(props: IconProps) {
  return (
    <IconBase {...props}>
      <path d="m12 3 9 9-9 9-9-9z" />
    </IconBase>
  );
}
