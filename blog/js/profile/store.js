/**
 * Profile store — Flux-like state management for the profiling quiz.
 * Persists answers and progress to localStorage so refreshes don't lose data.
 */

const STORAGE_KEY = 'yatzy-profile-state';

let state = initialState();
const listeners = new Set();

function initialState() {
  return {
    phase: 'loading',    // 'loading' | 'intro' | 'answering' | 'complete'
    scenarios: [],
    currentIndex: 0,
    answers: [],          // {scenarioId, actionId, responseTimeMs}
    profile: null,        // {theta, beta, gamma, d, ci_theta, ci_beta, ci_gamma, ci_d}
    profileHistory: [],   // incremental estimates after each answer (from #5 onward)
    error: null,
  };
}

function saveToStorage(state) {
  try {
    const data = {
      phase: state.phase,
      currentIndex: state.currentIndex,
      answers: state.answers,
      profile: state.profile,
      profileHistory: state.profileHistory,
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
  } catch (_) { /* quota exceeded or private mode */ }
}

function loadFromStorage() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    return JSON.parse(raw);
  } catch (_) { return null; }
}

function clearStorage() {
  try { localStorage.removeItem(STORAGE_KEY); } catch (_) {}
}

export function getState() { return state; }

export function dispatch(action) {
  const prev = state;
  state = reducer(state, action);
  if (state !== prev) {
    // Persist on meaningful state changes
    if (['START_QUIZ', 'ANSWER', 'CLEAR_ANSWER', 'ADVANCE', 'GO_TO', 'UPDATE_PROFILE'].includes(action.type)) {
      saveToStorage(state);
    }
    if (action.type === 'RESET') {
      clearStorage();
    }
    listeners.forEach(fn => fn(state, prev, action));
  }
}

export function subscribe(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

function reducer(state, action) {
  switch (action.type) {
    case 'SCENARIOS_LOADED': {
      // Restore saved progress if available
      const saved = loadFromStorage();
      if (saved && saved.answers && saved.answers.length > 0) {
        const phase = saved.phase === 'complete' ? 'complete'
          : saved.phase === 'answering' ? 'answering' : 'intro';
        return {
          ...state,
          phase,
          scenarios: action.scenarios,
          currentIndex: saved.currentIndex,
          answers: saved.answers,
          profile: saved.profile,
          profileHistory: saved.profileHistory || [],
          error: null,
        };
      }
      return {
        ...state,
        phase: 'intro',
        scenarios: action.scenarios,
        error: null,
      };
    }

    case 'LOAD_ERROR':
      return { ...state, phase: 'loading', error: action.error };

    case 'START_QUIZ':
      return {
        ...state,
        phase: 'answering',
        currentIndex: 0,
        answers: [],
        profile: null,
        profileHistory: [],
      };

    case 'ANSWER': {
      // Record answer (or update existing) but don't advance yet
      const existing = state.answers.findIndex(a => a.scenarioId === action.answer.scenarioId);
      const answers = [...state.answers];
      if (existing >= 0) {
        answers[existing] = action.answer;
      } else {
        answers.push(action.answer);
      }
      return { ...state, answers };
    }

    case 'CLEAR_ANSWER': {
      const answers = state.answers.filter(a => a.scenarioId !== action.scenarioId);
      return { ...state, answers };
    }

    case 'ADVANCE': {
      const nextIndex = state.currentIndex + 1;
      const isComplete = nextIndex >= state.scenarios.length;
      return {
        ...state,
        currentIndex: isComplete ? state.currentIndex : nextIndex,
        phase: isComplete ? 'complete' : 'answering',
      };
    }

    case 'GO_TO': {
      const idx = Math.max(0, Math.min(action.index, state.scenarios.length - 1));
      return { ...state, currentIndex: idx, phase: 'answering' };
    }

    case 'UPDATE_PROFILE':
      return {
        ...state,
        profile: action.profile,
        profileHistory: [...state.profileHistory, action.profile],
      };

    case 'RESET':
      return {
        ...state,
        phase: state.scenarios.length > 0 ? 'intro' : 'loading',
        currentIndex: 0,
        answers: [],
        profile: null,
        profileHistory: [],
        error: null,
      };

    default:
      return state;
  }
}
