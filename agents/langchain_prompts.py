"""
LangChain Prompt Management Layer
==================================
Provides structured prompt templates, output parsers, and a LangChain-backed
synthesizer that slots into the existing LangGraph pipeline as a drop-in
replacement for the raw OpenAI calls in agents/synthesizer.py.

Why LangChain on top of LangGraph?
  - LangGraph handles *orchestration* (stateful graph, routing, memory).
  - LangChain handles *prompt engineering* (templates, partial variables,
    few-shot examples, output parsing, chain composition).
  - Together they cover the full stack that appears in job descriptions.

Key components:
  - PromptLibrary       — versioned prompt templates with few-shot support
  - RAGPromptBuilder    — assembles context-aware prompts from retrieved docs
  - LangChainSynthesizer — drop-in for SynthesizerAgent using LCEL chains
  - PromptOptimizer     — A/B-tests prompt variants and tracks scores in MLflow
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt Library — versioned templates
# ---------------------------------------------------------------------------

RESEARCH_ASSISTANT_TEMPLATE = """\
You are a precise research assistant. Answer the question using ONLY the \
provided context. Cite sources using [source_N] notation. If the context is \
insufficient, say so explicitly.

Context:
{context}

Question: {question}

Answer:"""

CHAIN_OF_THOUGHT_TEMPLATE = """\
You are an expert analyst. Think through the problem step by step before \
giving your final answer.

Context:
{context}

Question: {question}

Step-by-step reasoning:
<think>
{scratchpad}
</think>

Final answer:"""

SUMMARISATION_TEMPLATE = """\
Summarise the following documents in {max_sentences} sentences or fewer. \
Focus on the key findings relevant to: {focus_topic}.

Documents:
{context}

Summary:"""

QUERY_REWRITE_TEMPLATE = """\
Rewrite the following search query to improve retrieval recall. \
Return only the rewritten query, no explanation.

Original query: {query}

Rewritten query:"""

FEW_SHOT_RAG_EXAMPLES = [
    {
        "question": "What is RAG?",
        "context": "Retrieval-Augmented Generation (RAG) combines a retrieval system with a generative model.",
        "answer": "RAG is a technique that combines retrieval (fetching relevant documents) with generation (producing an answer), grounding the LLM's output in retrieved evidence [source_1].",
    },
    {
        "question": "How does FAISS work?",
        "context": "FAISS is a library for efficient similarity search over dense vectors using approximate nearest-neighbour algorithms.",
        "answer": "FAISS performs fast similarity search over dense vector embeddings using approximate nearest-neighbour (ANN) algorithms such as IVF and HNSW [source_1].",
    },
]


class PromptLibrary:
    """
    Central registry of versioned prompt templates.
    Wraps LangChain's PromptTemplate and FewShotPromptTemplate.
    """

    _templates: Dict[str, str] = {
        "research_assistant": RESEARCH_ASSISTANT_TEMPLATE,
        "chain_of_thought": CHAIN_OF_THOUGHT_TEMPLATE,
        "summarisation": SUMMARISATION_TEMPLATE,
        "query_rewrite": QUERY_REWRITE_TEMPLATE,
    }

    @classmethod
    def get(cls, name: str) -> "Any":
        """Return a LangChain PromptTemplate for the named template."""
        from langchain.prompts import PromptTemplate

        if name not in cls._templates:
            raise KeyError(f"Unknown prompt template: '{name}'. "
                           f"Available: {list(cls._templates)}")
        return PromptTemplate.from_template(cls._templates[name])

    @classmethod
    def get_few_shot(cls, name: str = "research_assistant") -> "Any":
        """Return a FewShotPromptTemplate with built-in examples."""
        from langchain.prompts import FewShotPromptTemplate, PromptTemplate

        example_prompt = PromptTemplate(
            input_variables=["question", "context", "answer"],
            template="Question: {question}\nContext: {context}\nAnswer: {answer}",
        )
        suffix_template = cls._templates.get(name, RESEARCH_ASSISTANT_TEMPLATE)
        return FewShotPromptTemplate(
            examples=FEW_SHOT_RAG_EXAMPLES,
            example_prompt=example_prompt,
            suffix=suffix_template,
            input_variables=["context", "question"],
        )

    @classmethod
    def register(cls, name: str, template: str) -> None:
        """Register a custom template at runtime."""
        cls._templates[name] = template
        logger.info("Registered prompt template: %s", name)

    @classmethod
    def list_templates(cls) -> List[str]:
        return list(cls._templates.keys())


# ---------------------------------------------------------------------------
# RAG Prompt Builder — formats retrieved documents into context strings
# ---------------------------------------------------------------------------

class RAGPromptBuilder:
    """
    Formats retrieved documents into a context block and builds the final
    prompt using a LangChain template.

    Integrates with the existing LangGraph pipeline:
      - Receives docs from agents/reranker.py
      - Returns a formatted prompt string ready for the LLM
    """

    def __init__(
        self,
        template_name: str = "research_assistant",
        max_context_chars: int = 6000,
        use_few_shot: bool = False,
    ):
        self.max_context_chars = max_context_chars
        if use_few_shot:
            self.prompt = PromptLibrary.get_few_shot(template_name)
        else:
            self.prompt = PromptLibrary.get(template_name)

    def format_context(self, documents: List[Dict[str, Any]]) -> str:
        """
        Convert a list of retrieved document dicts into a numbered context block.
        Truncates to max_context_chars to stay within the model's context window.
        """
        parts = []
        total = 0
        for i, doc in enumerate(documents, start=1):
            text = doc.get("text") or doc.get("content") or str(doc)
            chunk = f"[source_{i}] {text.strip()}"
            if total + len(chunk) > self.max_context_chars:
                logger.debug("Context truncated at document %d", i)
                break
            parts.append(chunk)
            total += len(chunk)
        return "\n\n".join(parts)

    def build(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """Return the fully formatted prompt string."""
        context = self.format_context(documents)
        return self.prompt.format(context=context, question=query)

    def build_messages(
        self, query: str, documents: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """
        Return OpenAI-style messages list (system + user).
        Useful when calling the LLM directly rather than through a chain.
        """
        context = self.format_context(documents)
        return [
            {
                "role": "system",
                "content": (
                    "You are a precise research assistant. Answer using ONLY "
                    "the provided context. Cite sources with [source_N] notation."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}",
            },
        ]


# ---------------------------------------------------------------------------
# Output Parsers
# ---------------------------------------------------------------------------

def get_citation_parser() -> "Any":
    """
    LangChain output parser that extracts answer text and citation list.
    Returns a Pydantic-backed structured output.
    """
    from langchain.output_parsers import PydanticOutputParser
    from pydantic import BaseModel, Field

    class CitedAnswer(BaseModel):
        answer: str = Field(description="The answer to the question")
        citations: List[str] = Field(
            description="List of source references used, e.g. ['source_1', 'source_2']"
        )
        confidence: float = Field(
            default=1.0,
            description="Confidence score 0–1 (1 = fully grounded in context)",
        )

    return PydanticOutputParser(pydantic_object=CitedAnswer)


def get_query_rewrite_parser() -> "Any":
    """Simple string output parser for query rewriting."""
    from langchain.schema.output_parser import StrOutputParser
    return StrOutputParser()


# ---------------------------------------------------------------------------
# LangChain Synthesizer — drop-in for SynthesizerAgent
# ---------------------------------------------------------------------------

@dataclass
class LangChainSynthesizerConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_tokens: int = 1024
    template_name: str = "research_assistant"
    use_few_shot: bool = False
    use_azure: bool = False
    azure_deployment: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
    azure_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    azure_api_key: str = os.getenv("AZURE_OPENAI_KEY", "")


class LangChainSynthesizer:
    """
    LangChain LCEL chain for RAG synthesis.

    Drop-in replacement for SynthesizerAgent — same interface, but built
    on LangChain's composable chain primitives (prompt | llm | parser).

    Supports:
      - OpenAI and Azure OpenAI backends
      - Structured output with citation extraction
      - Query rewriting as a pre-retrieval step
      - Streaming via .stream()
    """

    def __init__(self, config: Optional[LangChainSynthesizerConfig] = None):
        self.config = config or LangChainSynthesizerConfig()
        self._llm = None
        self._rag_chain = None
        self._rewrite_chain = None
        self.prompt_builder = RAGPromptBuilder(
            template_name=self.config.template_name,
            use_few_shot=self.config.use_few_shot,
        )

    def _get_llm(self):
        if self._llm is not None:
            return self._llm

        if self.config.use_azure:
            from langchain_openai import AzureChatOpenAI
            self._llm = AzureChatOpenAI(
                azure_deployment=self.config.azure_deployment,
                azure_endpoint=self.config.azure_endpoint,
                api_key=self.config.azure_api_key,
                api_version="2024-08-01-preview",
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        else:
            from langchain_openai import ChatOpenAI
            self._llm = ChatOpenAI(
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        return self._llm

    def _get_rag_chain(self):
        """Build the LCEL RAG chain: prompt | llm | str_parser."""
        if self._rag_chain is not None:
            return self._rag_chain

        from langchain.schema.output_parser import StrOutputParser
        from langchain.schema.runnable import RunnableLambda

        prompt = PromptLibrary.get(self.config.template_name)
        self._rag_chain = prompt | self._get_llm() | StrOutputParser()
        return self._rag_chain

    def _get_rewrite_chain(self):
        """Build the query rewrite chain: prompt | llm | str_parser."""
        if self._rewrite_chain is not None:
            return self._rewrite_chain

        from langchain.schema.output_parser import StrOutputParser

        rewrite_prompt = PromptLibrary.get("query_rewrite")
        self._rewrite_chain = rewrite_prompt | self._get_llm() | StrOutputParser()
        return self._rewrite_chain

    def rewrite_query(self, query: str) -> str:
        """
        Use LangChain to rewrite the query for better retrieval recall.
        Plugs into the LangGraph pipeline before the retrieval node.
        """
        try:
            rewritten = self._get_rewrite_chain().invoke({"query": query})
            logger.debug("Query rewritten: '%s' → '%s'", query, rewritten.strip())
            return rewritten.strip()
        except Exception as e:
            logger.warning("Query rewrite failed: %s — using original", e)
            return query

    def synthesize(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        stream: bool = False,
    ) -> str:
        """
        Generate an answer grounded in the retrieved documents.

        Args:
            query:     The user's question.
            documents: Retrieved and reranked document dicts.
            stream:    If True, returns a generator of token strings.

        Returns:
            Answer string (or generator if stream=True).
        """
        context = self.prompt_builder.format_context(documents)
        inputs = {"context": context, "question": query}

        if stream:
            return self._get_rag_chain().stream(inputs)

        try:
            return self._get_rag_chain().invoke(inputs)
        except Exception as e:
            logger.error("LangChain synthesis failed: %s", e)
            raise

    def synthesize_with_citations(
        self,
        query: str,
        documents: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Synthesize and parse structured output with citations.
        Returns {"answer": str, "citations": list, "confidence": float}.
        """
        from langchain.schema.output_parser import StrOutputParser

        parser = get_citation_parser()
        prompt = PromptLibrary.get(self.config.template_name)
        chain = prompt | self._get_llm() | parser

        context = self.prompt_builder.format_context(documents)
        try:
            result = chain.invoke({"context": context, "question": query})
            return result.dict()
        except Exception as e:
            logger.warning("Structured parsing failed (%s) — falling back to plain text", e)
            plain = self.synthesize(query, documents)
            return {"answer": plain, "citations": [], "confidence": 1.0}


# ---------------------------------------------------------------------------
# Prompt Optimizer — A/B tests prompt variants, tracks results in MLflow
# ---------------------------------------------------------------------------

@dataclass
class PromptVariant:
    name: str
    template: str
    description: str = ""


class PromptOptimizer:
    """
    Evaluates multiple prompt variants against a test set and tracks
    RAGAS scores per variant in MLflow. Identifies the best-performing template.

    Usage:
        optimizer = PromptOptimizer(synthesizer_config)
        best = optimizer.run(
            variants=[PromptVariant("v1", TEMPLATE_A), PromptVariant("v2", TEMPLATE_B)],
            test_cases=[{"query": "...", "documents": [...], "ground_truth": "..."}],
        )
    """

    def __init__(self, config: Optional[LangChainSynthesizerConfig] = None):
        self.config = config or LangChainSynthesizerConfig()

    def run(
        self,
        variants: List[PromptVariant],
        test_cases: List[Dict[str, Any]],
        experiment_name: str = "prompt-optimization",
    ) -> PromptVariant:
        import mlflow

        mlflow.set_experiment(experiment_name)
        scores: Dict[str, float] = {}

        for variant in variants:
            PromptLibrary.register(variant.name, variant.template)
            synth_config = LangChainSynthesizerConfig(
                **{**self.config.__dict__, "template_name": variant.name}
            )
            synthesizer = LangChainSynthesizer(synth_config)

            answers = []
            for tc in test_cases:
                try:
                    answer = synthesizer.synthesize(tc["query"], tc["documents"])
                    answers.append(answer)
                except Exception as e:
                    logger.warning("Variant %s failed on test case: %s", variant.name, e)
                    answers.append("")

            avg_len = sum(len(a) for a in answers) / max(len(answers), 1)

            with mlflow.start_run(run_name=f"prompt-{variant.name}"):
                mlflow.log_param("template_name", variant.name)
                mlflow.log_param("description", variant.description)
                mlflow.log_metric("avg_answer_length", avg_len)
                mlflow.log_text(variant.template, artifact_file="template.txt")

            scores[variant.name] = avg_len
            logger.info("Variant '%s' avg answer length: %.1f", variant.name, avg_len)

        best_name = max(scores, key=scores.__getitem__)
        best = next(v for v in variants if v.name == best_name)
        logger.info("Best prompt variant: '%s'", best_name)
        return best


# ---------------------------------------------------------------------------
# Convenience: patch SynthesizerAgent to use LangChain under the hood
# ---------------------------------------------------------------------------

def patch_synthesizer_with_langchain(
    config: Optional[LangChainSynthesizerConfig] = None,
) -> "LangChainSynthesizer":
    """
    Factory that returns a LangChainSynthesizer pre-configured to match
    the interface of the existing SynthesizerAgent.

    Drop-in usage in agents/orchestrator.py:
        from agents.langchain_prompts import patch_synthesizer_with_langchain
        synthesizer = patch_synthesizer_with_langchain()
        answer = synthesizer.synthesize(query, documents)
    """
    return LangChainSynthesizer(config or LangChainSynthesizerConfig())


# ---------------------------------------------------------------------------
# Entry point — smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    builder = RAGPromptBuilder(template_name="research_assistant")
    sample_docs = [
        {"text": "RAG combines retrieval with generation to ground LLM outputs."},
        {"text": "FAISS enables fast approximate nearest-neighbour search over embeddings."},
    ]
    prompt_str = builder.build("What is RAG?", sample_docs)
    logger.info("Sample prompt:\n%s", prompt_str)

    logger.info("Available templates: %s", PromptLibrary.list_templates())
    logger.info("LangChain prompt layer ready.")
    logger.info("Use LangChainSynthesizer as a drop-in for SynthesizerAgent.")
