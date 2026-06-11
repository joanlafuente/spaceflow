import { create } from 'zustand';

export type SpaceflowTextureMode = 'text' | 'image';

interface SpaceflowUiState {
  textureMode: SpaceflowTextureMode;
  setTextureMode: (mode: SpaceflowTextureMode) => void;
}

export const useSpaceflowUiStore = create<SpaceflowUiState>((set) => ({
  textureMode: 'text',
  setTextureMode: (mode) => set({ textureMode: mode }),
}));
