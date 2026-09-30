import { create } from 'zustand';
import type { NpzSpaceflowMetadata } from '../mesh/npzImport';

export type SpaceflowTextureMode = 'text' | 'image';

interface ImportedNpzMetadataEvent {
  id: number;
  metadata: NpzSpaceflowMetadata;
}

interface SpaceflowUiState {
  textureMode: SpaceflowTextureMode;
  setTextureMode: (mode: SpaceflowTextureMode) => void;
  importedNpzMetadata: ImportedNpzMetadataEvent | null;
  setImportedNpzMetadata: (metadata: NpzSpaceflowMetadata) => void;
}

export const useSpaceflowUiStore = create<SpaceflowUiState>((set) => ({
  textureMode: 'text',
  setTextureMode: (mode) => set({ textureMode: mode }),
  importedNpzMetadata: null,
  setImportedNpzMetadata: (metadata) => set(state => ({
    importedNpzMetadata: {
      id: (state.importedNpzMetadata?.id ?? 0) + 1,
      metadata,
    },
  })),
}));
