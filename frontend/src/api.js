// Centralized API configuration

// You can later make this dynamic with environment variables if needed
const BASE_URL = "http://localhost:8000";

export const API = {
  ask: `${BASE_URL}/ask`,
  // If you add more endpoints, define them here:
  // stats: `${BASE_URL}/stats`,
  // players: `${BASE_URL}/players`,
};
