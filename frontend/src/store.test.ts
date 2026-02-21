import { describe, it, expect, vi, beforeEach } from 'vitest';

// Stub localStorage before importing store
const storageMock = { getItem: vi.fn(), setItem: vi.fn(), removeItem: vi.fn() };
vi.stubGlobal('localStorage', storageMock);

import { getState, dispatch, subscribe, initStore } from './store.ts';

describe('store', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    storageMock.getItem.mockReturnValue(null);
    initStore();
  });

  it('returns initial state', () => {
    const s = getState();
    expect(s.turnPhase).toBe('idle');
    expect(s.dice).toHaveLength(5);
    expect(s.totalScore).toBe(0);
  });

  it('dispatch updates state', () => {
    dispatch({ type: 'ROLL' });
    const s = getState();
    expect(s.turnPhase).toBe('rolled');
    expect(s.rerollsRemaining).toBe(2);
  });

  it('subscribe is called on dispatch', () => {
    const fn = vi.fn();
    subscribe(fn);
    dispatch({ type: 'ROLL' });
    expect(fn).toHaveBeenCalledTimes(1);
    const [state, prev, action] = fn.mock.calls[0];
    expect(state.turnPhase).toBe('rolled');
    expect(prev.turnPhase).toBe('idle');
    expect(action.type).toBe('ROLL');
  });

  it('unsubscribe stops notifications', () => {
    const fn = vi.fn();
    const unsub = subscribe(fn);
    dispatch({ type: 'ROLL' });
    expect(fn).toHaveBeenCalledTimes(1);
    unsub();
    dispatch({ type: 'TOGGLE_DEBUG' });
    expect(fn).toHaveBeenCalledTimes(1); // no more calls
  });

  it('multiple listeners all get notified', () => {
    const fn1 = vi.fn();
    const fn2 = vi.fn();
    subscribe(fn1);
    subscribe(fn2);
    dispatch({ type: 'TOGGLE_DEBUG' });
    expect(fn1).toHaveBeenCalledTimes(1);
    expect(fn2).toHaveBeenCalledTimes(1);
  });

  it('dispatch persists state via saveState', () => {
    dispatch({ type: 'ROLL' });
    expect(storageMock.setItem).toHaveBeenCalled();
  });
});
