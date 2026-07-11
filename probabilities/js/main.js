import { initProbTool } from './render.js';
import { initScoreTool } from './score.js';

const probRoot = document.getElementById('prob-root');
if (probRoot) initProbTool(probRoot);

const scoreRoot = document.getElementById('score-root');
if (scoreRoot) initScoreTool(scoreRoot);
