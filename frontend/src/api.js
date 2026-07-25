import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const client = axios.create({
  baseURL: API_BASE_URL,
  timeout: 15000,
});

export async function checkHealth() {
  const { data } = await client.get("/health");
  return data;
}

export async function predictArticle({ title, text }) {
  const { data } = await client.post("/predict", { title, text });
  return data;
}
