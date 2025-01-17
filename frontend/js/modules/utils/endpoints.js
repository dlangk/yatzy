export const API_BASE_URL = window.API_BASE_URL || "http://localhost:9000";

export const API_ENDPOINTS = {
    BASE_URL: API_BASE_URL,
    AVAILABLE_CATEGORIES: `${API_BASE_URL}/available_categories`,
    SUGGEST_OPTIMAL_ACTION: `${API_BASE_URL}/suggest_optimal_action`,
    EVALUATE_USER_ACTION: `${API_BASE_URL}/evaluate_user_action`,
    EVALUATE_ACTIONS: `${API_BASE_URL}/evaluate_actions`,
    GET_HISTOGRAM: `${API_BASE_URL}/score_histogram`
};

export function buildUrlWithParams(baseUrl, params) {
    const url = new URL(baseUrl);
    Object.entries(params).forEach(([key, value]) => url.searchParams.append(key, value));
    return url.toString();
}

export async function getJsonRequest(url, options = {}) {
    try {
        console.log(`Fetching: ${url} with options:`, options);
        const response = await fetch(url, options);

        if (!response.ok) {
            console.error(`Error fetching ${url}: ${response.status} - ${response.statusText}`);
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const data = await response.json();
        console.log(`Successfully fetched ${url}:`, data);
        return data;
    } catch (error) {
        console.error(`Error during fetch ${url}:`, error);
        throw error;
    }
}

export async function postJsonRequest(url, data) {
    console.log("Sending POST request to:", url, data);
    try {
        const response = await fetch(url, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(data)
        });
        if (!response.ok) {
            throw new Error(`Server responded with ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        console.warn("API call failed:", error);
        return null;
    }
}