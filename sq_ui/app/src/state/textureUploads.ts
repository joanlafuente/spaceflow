import { create } from 'zustand';

interface TextureUploadState {
  localTextureImageFiles: Record<string, File | null>;
  setLocalTextureImageFile: (primitiveId: string, file: File | null) => void;
  clearLocalTextureImageFile: (primitiveId: string) => void;
  clearAllLocalTextureImageFiles: () => void;
}

export const useTextureUploadStore = create<TextureUploadState>((set) => ({
  localTextureImageFiles: {},
  setLocalTextureImageFile: (primitiveId, file) => set((state) => ({
    localTextureImageFiles: {
      ...state.localTextureImageFiles,
      [primitiveId]: file,
    },
  })),
  clearLocalTextureImageFile: (primitiveId) => set((state) => {
    const next = { ...state.localTextureImageFiles };
    delete next[primitiveId];
    return { localTextureImageFiles: next };
  }),
  clearAllLocalTextureImageFiles: () => set({ localTextureImageFiles: {} }),
}));
