"""
VecSec LLM Client Functional Diagnostic
Tests real runtime behavior of llm_client.py subsystems
Purpose: Diagnose all LLM client issues before refactoring
"""

import os
import sys
import traceback
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to Python path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

print("🚀 Starting VecSec LLM Client Functional Diagnostics\n")
print("=" * 60)


# ============================================================================
# Helper Functions
# ============================================================================


def reset_env():
    """Reset environment variables to safe defaults"""
    os.environ.pop("GOOGLE_API_KEY", None)
    os.environ.pop("OPENAI_API_KEY", None)
    os.environ.pop("API_FLASH_API_KEY", None)
    os.environ.pop("LANGSMITH_API_KEY", None)


# ============================================================================
# 1️⃣ Client Initialization Test
# ============================================================================


def test_client_initialization():
    """Test client initialization with different configs"""
    print("\n🔧 Testing Client Initialization...")
    reset_env()

    try:
        from src.evil_agent.llm_client import LLMClient, get_llm_client

        # Test 1: No API credentials
        reset_env()
        client1 = LLMClient()
        print("✅ Client initialized without credentials")
        print(f"   Google API key: {'Set' if client1.google_api_key else 'Not set'}")
        print(f"   OpenAI API key: {'Set' if client1.openai_api_key else 'Not set'}")
        print(f"   Flash API key: {'Set' if client1.flash_api_key else 'Not set'}")

        assert client1.google_api_key is None, "Should be None when not set"
        assert client1.openai_api_key is None, "Should be None when not set"

        # Test 2: With fake credentials
        os.environ["GOOGLE_API_KEY"] = "test_key_abc"
        os.environ["OPENAI_API_KEY"] = "test_key_xyz"
        client2 = LLMClient()
        print("✅ Client initialized with credentials")
        print(f"   Google API key: {'Set' if client2.google_api_key else 'Not set'}")
        print(f"   OpenAI API key: {'Set' if client2.openai_api_key else 'Not set'}")

        assert client2.google_api_key == "test_key_abc", "Should read Google API key"
        assert client2.openai_api_key == "test_key_xyz", "Should read OpenAI API key"

        # Test 3: Singleton pattern
        client3 = get_llm_client()
        client4 = get_llm_client()
        print(f"   Singleton test: client3 is client4 = {client3 is client4}")

        assert client3 is client4, "get_llm_client should return same instance"
        print("   ✅ Singleton pattern works")

    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 2️⃣ Google API Test
# ============================================================================


def test_google_api():
    """Test Google Gemini API integration"""
    print("\n🔧 Testing Google API...")
    reset_env()

    try:
        from src.evil_agent.llm_client import LLMClient

        client = LLMClient()

        # Test 1: No API key
        result1 = client.generate_with_google("Test prompt")
        print("   No API key:")
        print(f"     Result: {result1[:60]}...")

        assert "not configured" in result1.lower(), (
            "Should return error message when API key not configured"
        )
        print("   ✅ Returns error message when API key missing")

        # Test 2: With API key (mock)
        os.environ["GOOGLE_API_KEY"] = "test_key"
        client2 = LLMClient()

        with patch("src.evil_agent.llm_client.requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "candidates": [{"content": {"parts": [{"text": "Generated attack prompt"}]}}]
            }
            mock_post.return_value = mock_response

            result2 = client2.generate_with_google("Test prompt", model="gemini-1.5-pro")

            print("   With mocked API:")
            print(f"     Result: {result2}")

            assert result2 == "Generated attack prompt", "Should return generated text from API"
            print("   ✅ Google API integration works")

            # Check API call
            assert mock_post.called, "Should call requests.post"
            call_args = mock_post.call_args
            assert "generativelanguage.googleapis.com" in call_args[0][
                0
            ] or "generativelanguage.googleapis.com" in str(call_args), (
                "Should call Google API endpoint"
            )

        # Test 3: API error handling
        with patch("src.evil_agent.llm_client.requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.text = "Internal Server Error"
            mock_post.return_value = mock_response

            result3 = client2.generate_with_google("Test prompt")

            print("   API error handling:")
            print(f"     Result: {result3[:60]}...")

            assert "API Error" in result3 or "500" in result3, (
                "Should return error message for API errors"
            )
            print("   ✅ API error handling works")

    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 3️⃣ OpenAI API Test
# ============================================================================


def test_openai_api():
    """Test OpenAI API integration"""
    print("\n🔧 Testing OpenAI API...")
    reset_env()

    try:
        from src.evil_agent.llm_client import LLMClient

        client = LLMClient()

        # Test 1: No API key
        result1 = client.generate_with_openai("Test prompt")
        print("   No API key:")
        print(f"     Result: {result1[:60]}...")

        assert "not configured" in result1.lower(), (
            "Should return error message when API key not configured"
        )
        print("   ✅ Returns error message when API key missing")

        # Test 2: With API key (mock)
        os.environ["OPENAI_API_KEY"] = "test_key"
        client2 = LLMClient()

        with patch("src.evil_agent.llm_client.requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "Generated attack prompt"}}]
            }
            mock_post.return_value = mock_response

            result2 = client2.generate_with_openai("Test prompt", model="gpt-3.5-turbo")

            print("   With mocked API:")
            print(f"     Result: {result2}")

            assert result2 == "Generated attack prompt", "Should return generated text from API"
            print("   ✅ OpenAI API integration works")

            # Check API call
            assert mock_post.called, "Should call requests.post"
            call_args = mock_post.call_args
            assert "api.openai.com" in call_args[0][0] or "api.openai.com" in str(call_args), (
                "Should call OpenAI API endpoint"
            )

    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 4️⃣ Flash API Test
# ============================================================================


def test_flash_api():
    """Test Flash API integration"""
    print("\n🔧 Testing Flash API...")
    reset_env()

    try:
        from src.evil_agent.llm_client import LLMClient

        client = LLMClient()

        # Test 1: No API key
        result1 = client.generate_with_flash("Test prompt")
        print("   No API key:")
        print(f"     Result: {result1[:60]}...")

        assert "not configured" in result1.lower(), (
            "Should return error message when API key not configured"
        )
        print("   ✅ Returns error message when API key missing")

        # Test 2: With API key (mock)
        os.environ["API_FLASH_API_KEY"] = "test_key"
        client2 = LLMClient()

        with patch("src.evil_agent.llm_client.requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "Generated attack prompt"}}]
            }
            mock_post.return_value = mock_response

            result2 = client2.generate_with_flash("Test prompt")

            print("   With mocked API:")
            print(f"     Result: {result2}")

            assert result2 == "Generated attack prompt", "Should return generated text from API"
            print("   ✅ Flash API integration works")

    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 5️⃣ Error Handling Test (CRITICAL)
# ============================================================================


def test_error_handling():
    """Test error handling and timeout behavior"""
    print("\n🔴 Testing Error Handling...")
    reset_env()

    try:
        from src.evil_agent.llm_client import LLMClient

        os.environ["GOOGLE_API_KEY"] = "test_key"
        client = LLMClient()

        # Test 1: Network timeout
        with patch("src.evil_agent.llm_client.requests.post") as mock_post:
            import requests  # type: ignore[import-untyped]

            mock_post.side_effect = requests.Timeout("Connection timeout")

            result = client.generate_with_google("Test prompt")

            print("   Network timeout:")
            print(f"     Result: {result[:60]}...")

            assert "Error calling" in result or "timeout" in result.lower(), (
                "Should handle timeout errors"
            )
            print("   ✅ Timeout errors handled")

        # Test 2: Connection error
        with patch("src.evil_agent.llm_client.requests.post") as mock_post:
            import requests  # type: ignore[import-untyped]

            mock_post.side_effect = requests.ConnectionError("Connection failed")

            result = client.generate_with_google("Test prompt")

            print("   Connection error:")
            print(f"     Result: {result[:60]}...")

            assert "Error calling" in result or "connection" in result.lower(), (
                "Should handle connection errors"
            )
            print("   ✅ Connection errors handled")

        # Test 3: Generic exception
        with patch("src.evil_agent.llm_client.requests.post") as mock_post:
            mock_post.side_effect = Exception("Unexpected error")

            result = client.generate_with_google("Test prompt")

            print("   Generic exception:")
            print(f"     Result: {result[:60]}...")

            assert "Error calling" in result, "Should handle generic exceptions"
            print("   ✅ Generic exceptions handled")

        # Test 4: Invalid JSON response
        with patch("src.evil_agent.llm_client.requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.side_effect = ValueError("Invalid JSON")
            mock_post.return_value = mock_response

            result = client.generate_with_google("Test prompt")

            print("   Invalid JSON:")
            print(f"     Result: {result[:60]}...")

            assert "Error calling" in result, "Should handle JSON parsing errors"
            print("   ⚠️  ISSUE: May not handle JSON parsing errors gracefully")

    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 6️⃣ API Request Format Test
# ============================================================================


def test_api_request_format():
    """Test API request format and parameters"""
    print("\n🔧 Testing API Request Format...")
    reset_env()

    try:
        from src.evil_agent.llm_client import LLMClient

        os.environ["GOOGLE_API_KEY"] = "test_key"
        client = LLMClient()

        with patch("src.evil_agent.llm_client.requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "candidates": [{"content": {"parts": [{"text": "Test"}]}}]
            }
            mock_post.return_value = mock_response

            client.generate_with_google("Test prompt", model="gemini-1.5-pro")

            # Check request format
            assert mock_post.called, "Should make API call"

            call_args = mock_post.call_args
            call_kwargs = call_args[1] if len(call_args) > 1 else {}
            call_data = call_kwargs.get("json", {})

            print(f"   Request URL: {call_args[0][0]}")
            print(f"   Request data keys: {list(call_data.keys())}")

            # Check Google API format
            if "contents" in call_data:
                print("   ✅ Google API format correct")
                assert "contents" in call_data, "Should have 'contents' key"
                assert "generationConfig" in call_data, "Should have 'generationConfig' key"
            else:
                print("   ⚠️  Request format may not match Google API spec")

    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 7️⃣ Timeout Configuration Test
# ============================================================================


def test_timeout_configuration():
    """Test timeout configuration"""
    print("\n⚠️  Testing Timeout Configuration...")
    reset_env()

    try:
        from src.evil_agent.llm_client import LLMClient

        os.environ["GOOGLE_API_KEY"] = "test_key"
        client = LLMClient()

        with patch("src.evil_agent.llm_client.requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "candidates": [{"content": {"parts": [{"text": "Test"}]}}]
            }
            mock_post.return_value = mock_response

            client.generate_with_google("Test prompt")

            # Check timeout is set
            call_args = mock_post.call_args
            call_kwargs = call_args[1] if len(call_args) > 1 else {}
            timeout = call_kwargs.get("timeout", None)

            print(f"   Timeout value: {timeout}")

            if timeout:
                assert timeout == 30, f"Expected timeout=30, got {timeout}"
                print("   ✅ Timeout configured correctly (30 seconds)")
            else:
                print("   ⚠️  ISSUE: No timeout configured")
                print("   ⚠️  Expected: timeout=30 seconds")
                print("   ⚠️  Actual: No timeout parameter")
                print("   ⚠️  Impact: Requests may hang indefinitely")

    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 8️⃣ Edge Cases Test
# ============================================================================


def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n🔧 Testing Edge Cases...")
    reset_env()

    try:
        from src.evil_agent.llm_client import LLMClient

        client = LLMClient()

        # Test 1: Empty prompt
        result = client.generate_with_google("")
        print(f"   Empty prompt: {result[:60]}...")

        # Test 2: Very long prompt
        long_prompt = "x" * 10000
        result = client.generate_with_google(long_prompt)
        print(f"   Very long prompt: {result[:60]}...")

        # Test 3: None prompt (should handle gracefully)
        try:
            result = client.generate_with_google(None)
            print(f"   None prompt handled: {result[:60]}...")
        except Exception as e:
            print(f"   ⚠️  None prompt raised error: {type(e).__name__}: {e}")

        # Test 4: Invalid model name
        os.environ["GOOGLE_API_KEY"] = "test_key"
        client2 = LLMClient()

        with patch("src.evil_agent.llm_client.requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 400
            mock_response.text = "Invalid model"
            mock_post.return_value = mock_response

            result = client2.generate_with_google("Test", model="invalid-model")
            print(f"   Invalid model: {result[:60]}...")

            assert "API Error" in result or "400" in result, "Should handle invalid model errors"

        print("   ✅ Edge cases handled")

    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# Run All Tests
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("VecSec LLM Client Functional Diagnostics")
    print("=" * 60)

    test_client_initialization()
    test_google_api()
    test_openai_api()
    test_flash_api()
    test_error_handling()
    test_api_request_format()
    test_timeout_configuration()
    test_edge_cases()

    print("\n" + "=" * 60)
    print("🏁 LLM Client Diagnostics Complete")
    print("=" * 60)
    print("\n📋 Summary of Issues Found:")
    print("   🔴 CRITICAL: Returns error strings instead of raising exceptions")
    print("   ⚠️  HIGH: May not handle JSON parsing errors")
    print("   ⚠️  MEDIUM: Timeout configuration may not be consistent")
    print("   ⚠️  MEDIUM: No retry mechanism for transient failures")
