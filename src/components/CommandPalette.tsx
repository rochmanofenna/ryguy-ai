'use client';

import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface CommandPaletteProps {
  isOpen: boolean;
  onClose: () => void;
  onCommand: (command: string) => void;
}

interface Command {
  id: string;
  title: string;
  description: string;
  category: string;
}

const commands: Command[] = [
  {
    id: 'about',
    title: 'About Ryan',
    description: 'My story and background',
    category: 'Personal'
  },
  {
    id: 'experience',
    title: 'Experience',
    description: 'Interactive work timeline',
    category: 'Personal'
  },
  {
    id: 'portfolio',
    title: 'Portfolio',
    description: 'Live trading P&L dashboard',
    category: 'Trading'
  },
  {
    id: 'stack',
    title: 'System Architecture',
    description: 'View trading system stack',
    category: 'Technical'
  },
  {
    id: 'gpu',
    title: 'GPU Demo',
    description: 'Monte Carlo simulation demo',
    category: 'Technical'
  },
  {
    id: 'skills',
    title: 'Skills',
    description: 'Proven technical skills with validation',
    category: 'Personal'
  },
  {
    id: 'projects',
    title: 'Projects',
    description: 'Browse all projects',
    category: 'Personal'
  },
  {
    id: 'cv',
    title: 'Download CV',
    description: 'Get my resume in PDF format',
    category: 'Personal'
  },
  {
    id: 'clear',
    title: 'Clear Terminal',
    description: 'Clear the terminal screen',
    category: 'System'
  }
];

export default function CommandPalette({ isOpen, onClose, onCommand }: CommandPaletteProps) {
  const [search, setSearch] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  const filteredCommands = commands.filter(cmd =>
    cmd.title.toLowerCase().includes(search.toLowerCase()) ||
    cmd.description.toLowerCase().includes(search.toLowerCase()) ||
    cmd.id.toLowerCase().includes(search.toLowerCase())
  );

  useEffect(() => {
    if (isOpen) {
      inputRef.current?.focus();
      setSearch('');
      setSelectedIndex(0);
    }
  }, [isOpen]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setSelectedIndex((prev) =>
        prev < filteredCommands.length - 1 ? prev + 1 : 0
      );
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setSelectedIndex((prev) =>
        prev > 0 ? prev - 1 : filteredCommands.length - 1
      );
    } else if (e.key === 'Enter' && filteredCommands.length > 0) {
      e.preventDefault();
      const selected = filteredCommands[selectedIndex];
      onCommand(selected.id);
      onClose();
    } else if (e.key === 'Escape') {
      e.preventDefault();
      onClose();
    }
  };

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/50 z-50"
            onClick={onClose}
          />
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full max-w-2xl bg-terminal-bg border border-terminal-accent/30 rounded-lg shadow-2xl z-50"
          >
            <div className="p-4 border-b border-terminal-accent/20">
              <input
                ref={inputRef}
                type="text"
                value={search}
                onChange={(e) => {
                  setSearch(e.target.value);
                  setSelectedIndex(0);
                }}
                onKeyDown={handleKeyDown}
                placeholder="Type a command or search..."
                className="w-full bg-transparent text-terminal-text outline-none placeholder-terminal-muted"
              />
            </div>
            <div className="max-h-96 overflow-y-auto">
              {filteredCommands.length === 0 ? (
                <div className="p-4 text-terminal-muted text-center">
                  No commands found
                </div>
              ) : (
                <div className="p-2">
                  {Object.entries(
                    filteredCommands.reduce((acc, cmd) => {
                      if (!acc[cmd.category]) acc[cmd.category] = [];
                      acc[cmd.category].push(cmd);
                      return acc;
                    }, {} as Record<string, Command[]>)
                  ).map(([category, cmds]) => (
                    <div key={category} className="mb-4">
                      <div className="text-terminal-muted text-xs uppercase px-2 mb-2">
                        {category}
                      </div>
                      {cmds.map((cmd, index) => {
                        const globalIndex = filteredCommands.indexOf(cmd);
                        return (
                          <div
                            key={cmd.id}
                            className={`px-2 py-2 rounded cursor-pointer transition-colors ${
                              globalIndex === selectedIndex
                                ? 'bg-terminal-accent/20 text-terminal-text'
                                : 'hover:bg-terminal-accent/10'
                            }`}
                            onClick={() => {
                              onCommand(cmd.id);
                              onClose();
                            }}
                            onMouseEnter={() => setSelectedIndex(globalIndex)}
                          >
                            <div className="flex items-center justify-between">
                              <div>
                                <div className="text-sm font-medium">
                                  {cmd.title}
                                </div>
                                <div className="text-xs text-terminal-muted">
                                  {cmd.description}
                                </div>
                              </div>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  ))}
                </div>
              )}
            </div>
            <div className="p-3 border-t border-terminal-accent/20 text-xs text-terminal-muted">
              <span className="mr-4">↑↓ Navigate</span>
              <span className="mr-4">↵ Select</span>
              <span>ESC Close</span>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}