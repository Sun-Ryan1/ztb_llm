from typing import Any, List, Optional
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import LLMResult, Generation
from core_modules.llm_integration import LLMEngine
from pydantic import Field


class CustomLLM(BaseLLM):
    """将现有的 LLMEngine 适配为 LangChain 的 LLM"""
    
    llm_engine: LLMEngine = Field(..., description="The LLM engine instance")

    def __init__(self, llm_engine: LLMEngine):
        super().__init__(llm_engine=llm_engine)

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager = None,
        **kwargs
    ) -> LLMResult:
        generations = []
        for prompt in prompts:
            answer = self.llm_engine.generate(
                prompt,
                max_new_tokens=kwargs.get("max_new_tokens", 512),
                temperature=kwargs.get("temperature", 0.0),
                do_sample=False
            )
            generations.append([Generation(text=answer)])
        return LLMResult(generations=generations)

    @property
    def _llm_type(self) -> str:
        return "custom_qwen"