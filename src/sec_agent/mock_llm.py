"""
Mock LLM and Embeddings for demonstration
"""


class MockLLM:
    """Mock LLM for demonstration - simplified without abstract base class"""

    def __init__(self):
        pass

    def _generate(self, messages, **kwargs):
        # Simple mock response based on the question
        question = messages[-1].content if messages else "No question provided"
        if "RAG" in question or "retrieval" in question:
            response = "RAG (Retrieval-Augmented Generation) is a technique that combines retrieval of relevant documents with generation of responses. It first retrieves relevant context from a knowledge base, then uses that context to generate more accurate and informed responses."
        elif "finance" in question:
            response = "I cannot provide information about finance topics as they are restricted for your tenant."
        else:
            response = "I don't have enough context to answer this question accurately."

        return type("MockResult", (), {"content": response})()

    def invoke(self, messages, **kwargs):
        return self._generate(messages, **kwargs)

    def agenerate_prompt(self, prompts, **kwargs):
        # Async version - not implemented for this demo
        raise NotImplementedError("Async generation not implemented in mock")

    def generate_prompt(self, prompts, **kwargs):
        # Generate for prompts
        results = []
        for prompt in prompts:
            response = self._generate([type("MockMessage", (), {"content": prompt})()], **kwargs)
            results.append(response)
        return results


class MockEmbeddings:
    """Mock embeddings for demonstration"""

    def embed_documents(self, texts):
        # Return random embeddings for demonstration
        import random

        return [[random.random() for _ in range(384)] for _ in texts]

    def embed_query(self, text):
        import random

        return [random.random() for _ in range(384)]
