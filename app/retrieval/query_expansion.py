"""
Query expansion module.
Uses LLM-based query reformulation to generate related search formulations
and overcome the vocabulary mismatch problem.
"""

import logging
from typing import List, Optional
from app.config import settings

logger = logging.getLogger(__name__)

EXPANSION_SYSTEM_PROMPT = """You are an AI search query optimization assistant.
Your task is to generate alternative search query formulations for enterprise document retrieval.
Generate {num_queries} diverse search queries related to the original query.
Include synonyms, related technical terms, and alternative question formulations.
Output only the generated queries, one per line. Do not include numbers, bullet points, or commentary."""


class QueryExpander:
    """
    Expands a user's original query into multiple lexical and semantic search variants.
    """

    def __init__(self, llm=None):
        """
        Initializes QueryExpander with an optional LLM instance.
        """
        self.llm = llm

    def _get_llm(self):
        """
        Initializes ChatOpenAI if not explicitly provided and an API key is configured.
        """
        if self.llm is not None:
            return self.llm

        if settings.OPENAI_API_KEY:
            try:
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(
                    model=settings.LLM_MODEL,
                    temperature=0.3,
                    api_key=settings.OPENAI_API_KEY,
                    base_url=settings.OPENAI_BASE_URL
                )
            except Exception as e:
                logger.warning(f"Could not initialize ChatOpenAI: {e}")
                return None
        return None

    def expand(self, query: str, num_queries: Optional[int] = None) -> List[str]:
        """
        Expands the original query into a list of query formulations.
        Guarantees that the original query is always returned at index 0.
        """
        if not query or not query.strip():
            return []

        cleaned_query = query.strip()
        count = num_queries or settings.QUERY_EXPANSION_COUNT
        llm = self._get_llm()

        # 1. If LLM is available, perform prompt-based expansion
        if llm is not None:
            try:
                prompt = (
                    f"{EXPANSION_SYSTEM_PROMPT.format(num_queries=count)}\n\n"
                    f"Original Query: {cleaned_query}\n"
                    f"Alternative Queries:"
                )
                response = llm.invoke(prompt)
                content = response.content if hasattr(response, "content") else str(response)

                variants = []
                for line in content.strip().splitlines():
                    cleaned_line = line.strip().lstrip("0123456789.-*• ")
                    if cleaned_line and cleaned_line.lower() != cleaned_query.lower():
                        variants.append(cleaned_line)

                if variants:
                    return [cleaned_query] + variants[:count]
            except Exception as e:
                logger.warning(f"LLM query expansion failed: {e}. Using fallback.")

        # 2. Resilient fallback when offline or no API key is present
        fallback_variants = self._fallback_expansion(cleaned_query, count)
        return [cleaned_query] + fallback_variants[:count]

    def _fallback_expansion(self, query: str, count: int) -> List[str]:
        """
        Deterministic heuristic fallback providing synonyms for common enterprise terms
        to ensure local execution and testing never crash without an external LLM.
        """
        q_lower = query.lower()
        variants = []

        if "refund" in q_lower:
            variants.extend([
                "customer refund eligibility and policy deadline",
                "money back guarantee and cancellation window",
                "how many days to request a refund"
            ])
        elif "leave" in q_lower or "vacation" in q_lower or "pto" in q_lower:
            variants.extend([
                "employee paid time off vacation accrual",
                "sick leave and parental leave guidelines",
                "annual leave rollover rules"
            ])
        elif "security" in q_lower or "encryption" in q_lower:
            variants.extend([
                "data protection SOC2 compliance and TLS standards",
                "AES-256 encryption at rest and customer managed keys",
                "zero data retention security architecture"
            ])
        elif "pricing" in q_lower or "cost" in q_lower:
            variants.extend([
                "subscription tiers Starter Pro Business Enterprise",
                "monthly versus annual commitment pricing discount",
                "token overage rates and billing tiers"
            ])
        else:
            variants.extend([
                f"{query} details and specifications",
                f"guidelines regarding {query}",
                f"official policy for {query}"
            ])

        return [v for v in variants if v.lower() != q_lower][:count]
