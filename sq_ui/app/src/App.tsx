import { useEffect } from 'react';
import TopBar from './ui/TopBar';
import PrimitiveList from './ui/PrimitiveList';
import Viewport from './ui/Viewport';
import ParameterPanel from './ui/ParameterPanel';
import { useStore } from './state/store';
import './App.css';

export default function App() {
  const undo = useStore(s => s.undo);
  const redo = useStore(s => s.redo);
  const selectedId = useStore(s => s.selectedId);
  const removePrimitive = useStore(s => s.removePrimitive);
  const duplicatePrimitive = useStore(s => s.duplicatePrimitive);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).tagName === 'INPUT') return;

      if ((e.ctrlKey || e.metaKey) && e.key === 'z' && !e.shiftKey) {
        e.preventDefault();
        undo();
      } else if ((e.ctrlKey || e.metaKey) && e.key === 'z' && e.shiftKey) {
        e.preventDefault();
        redo();
      } else if ((e.ctrlKey || e.metaKey) && e.key === 'Z') {
        e.preventDefault();
        redo();
      } else if ((e.key === 'Delete' || e.key === 'Backspace') && selectedId) {
        e.preventDefault();
        removePrimitive(selectedId);
      } else if (e.key === 'd' && selectedId && !e.ctrlKey && !e.metaKey) {
        e.preventDefault();
        duplicatePrimitive(selectedId);
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [undo, redo, selectedId, removePrimitive, duplicatePrimitive]);

  return (
    <div className="app">
      <TopBar />
      <div className="workspace">
        <PrimitiveList />
        <Viewport />
        <ParameterPanel />
      </div>
    </div>
  );
}
