"""
LLM API Client for Attack Generation
Handles API calls to Google, OpenAI, and Flash providers
"""

import os

import requests  # type: ignore[import-untyped]
from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    """Client for interacting with various LLM APIs"""

    def __init__(self):
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.flash_api_key = os.getenv("API_FLASH_API_KEY")
        self.langsmith_api_key = os.getenv("LANGSMITH_API_KEY")

    def generate_with_google(self, prompt: str, model: str = "gemini-1.5-pro") -> str:
        """Generate text using Google Gemini API"""
        if not self.google_api_key:
            return "Google API key not configured"

        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            headers = {"Content-Type": "application/json"}
            data = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.9,
                    "topK": 40,
                    "topP": 0.95,
                    "maxOutputTokens": 1024,
                },
            }

            response = requests.post(
                f"{url}?key={self.google_api_key}", headers=headers, json=data, timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                return result["candidates"][0]["content"]["parts"][0]["text"]
            else:
                return f"API Error: {response.status_code} - {response.text}"

        except Exception as e:
            return f"Error calling Google API: {str(e)}"

    def generate_with_openai(self, prompt: str, model: str = "gpt-3.5-turbo") -> str:
        """Generate text using OpenAI API"""
        if not self.openai_api_key:
            return "OpenAI API key not configured"

        try:
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json",
            }
            data = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.9,
                "max_tokens": 1024,
            }

            response = requests.post(url, headers=headers, json=data, timeout=30)

            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                return f"API Error: {response.status_code} - {response.text}"

        except Exception as e:
            return f"Error calling OpenAI API: {str(e)}"

    def generate_with_flash(self, prompt: str) -> str:
        """Generate text using Flash API"""
        if not self.flash_api_key:
            return "Flash API key not configured"

        try:
            url = "https://api.flashinfer.ai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.flash_api_key}",
                "Content-Type": "application/json",
            }
            data = {
                "model": "flashinfer-llama-3.1-8b-instruct",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.9,
                "max_tokens": 1024,
            }

            response = requests.post(url, headers=headers, json=data, timeout=30)

            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                return f"API Error: {response.status_code} - {response.text}"

        except Exception as e:
            return f"Error calling Flash API: {str(e)}"


# Singleton instance
_llm_client = None


def get_llm_client() -> LLMClient:
    """Get or create singleton LLMClient instance"""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
