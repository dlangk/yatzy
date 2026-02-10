declare global {
  interface Window {
    __API_BASE_URL__?: string;
  }
}

export const API_BASE_URL: string =
  window.__API_BASE_URL__ ||
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ||
  'http://localhost:9000';
