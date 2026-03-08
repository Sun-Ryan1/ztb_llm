"""
提示工程模块
============
独立管理各类提示模板，支持基础问答、多轮对话、专业领域模板，
并提供上下文格式化和系统指令集成。
"""

import json
import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class TemplateType(str, Enum):
    """模板类型枚举"""
    BASIC = "basic"               # 基础问答
    CONVERSATION = "conversation" # 多轮对话
    PROFESSIONAL = "professional" # 专业领域（招投标）


@dataclass
class PromptTemplate:
    """
    提示模板类
    template: 模板字符串，使用 {变量名} 占位符
    system_prompt: 可选的系统指令（部分模型可单独传入）
    variables: 模板中需要的变量列表（自动提取）
    """
    name: str
    template: str
    template_type: TemplateType = TemplateType.BASIC
    system_prompt: Optional[str] = None
    description: str = ""
    version: str = "1.0"
    variables: List[str] = field(default_factory=list)

    def __post_init__(self):
        import re
        self.variables = re.findall(r'\{([^{}]+)\}', self.template)

    def format(self, **kwargs) -> str:
        missing = [v for v in self.variables if v not in kwargs]
        if missing:
            logger.warning(f"模板 {self.name} 缺少变量: {missing}")
        return self.template.format(**kwargs)

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "template": self.template,
            "type": self.template_type.value,
            "system_prompt": self.system_prompt,
            "description": self.description,
            "version": self.version,
            "variables": self.variables,
        }


class PromptManager:
    """
    提示模板管理器
    """
    def __init__(self, templates: Optional[Dict[str, PromptTemplate]] = None):
        self._templates: Dict[str, PromptTemplate] = templates or {}
        self._load_default_templates()

    def _load_default_templates(self):
        # 基础问答模板
        basic = PromptTemplate(
            name="basic_qa",
            template_type=TemplateType.BASIC,
            template="{question}",
            description="基础问答，直接输入问题",
        )
        self.add_template(basic)

        # 专业领域模板（招投标）- 简洁版，系统指令独立
        professional = PromptTemplate(
            name="tender_professional",
            template_type=TemplateType.PROFESSIONAL,
            system_prompt="""你是聚焦招投标采购全流程的专业智能问答系统，需严格依据《招标投标法》《政府采购法》等法规，精准解答政策合规、业务操作、物资产品、电子系统操作等领域问题。

# 回答要求
1. 准确性：严格依据相关法规和政策，确保信息准确无误
2. 完整性：全面覆盖问题要点，提供详细的分析和解释
3. 专业性：正确使用专业术语，体现专业知识和分析能力
4. 清晰性：语言流畅，逻辑清晰，结构合理
请严格依据以下规则回答：
- 只使用“相关知识”中提供的信息，不要编造。
- 答案必须与“相关知识”中的原文完全一致，直接复制原文，不要添加任何额外文字或解释。
- 不要输出“答：”等前缀，只输出答案本身。""",
            template="""相关知识：
{context}

问题：{question}
答案：""",
            description="招投标领域专业问答模板（简洁版）",
        )
        self.add_template(professional)

        # 多轮对话模板（带历史）
        conversation = PromptTemplate(
            name="conversation_history",
            template_type=TemplateType.CONVERSATION,
            system_prompt="你是一个有帮助的助手。",
            template="""{system_prompt}

{history}
用户：{question}
助手：""",
            description="带对话历史的多轮模板",
        )
        self.add_template(conversation)

    def add_template(self, template: PromptTemplate):
        self._templates[template.name] = template
        logger.debug(f"添加模板: {template.name}")

    def get_template(self, name: str) -> Optional[PromptTemplate]:
        return self._templates.get(name)

    def list_templates(self) -> List[str]:
        return list(self._templates.keys())

    def remove_template(self, name: str):
        if name in self._templates:
            del self._templates[name]

    def save_to_file(self, filepath: str):
        data = {name: tmpl.to_dict() for name, tmpl in self._templates.items()}
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"模板已保存到 {filepath}")

    def load_from_file(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        for name, tmpl_dict in data.items():
            tmpl_dict["template_type"] = TemplateType(tmpl_dict["type"])
            template = PromptTemplate(**tmpl_dict)
            self._templates[name] = template
        logger.info(f"从 {filepath} 加载 {len(data)} 个模板")


# ==================== 上下文格式化工具 ====================

def format_context(
    documents: List[str],
    format_type: str = "numbered",
    max_docs: int = 5,
    doc_prefix: str = "信息",
) -> str:
    if not documents:
        return ""
    docs = documents[:max_docs]
    if format_type == "plain":
        return "\n".join(docs)
    elif format_type == "markdown":
        return "\n\n".join([f"> {doc}" for doc in docs])
    else:
        return "\n".join([f"{doc_prefix}{i+1}: {doc}" for i, doc in enumerate(docs)])


def format_conversation_history(
    history: List[Dict[str, str]],
    user_key: str = "user",
    assistant_key: str = "assistant",
    max_turns: int = 10,
) -> str:
    lines = []
    for turn in history[-max_turns:]:
        if user_key in turn:
            lines.append(f"用户：{turn[user_key]}")
        if assistant_key in turn:
            lines.append(f"助手：{turn[assistant_key]}")
    return "\n".join(lines)


if __name__ == "__main__":
    manager = PromptManager()
    tmpl = manager.get_template("tender_professional")
    if tmpl:
        docs = [
            "上海仓祥绿化工程有限公司的注册地址是上海市松江区叶榭镇叶旺路1号三楼。"
        ]
        context = format_context(docs, format_type="numbered")
        question = "上海仓祥绿化工程有限公司的注册地址"
        full_prompt = f"{tmpl.system_prompt}\n\n{tmpl.template.format(context=context, question=question)}"
        print("生成的提示词：\n")
        print(full_prompt)
    manager.save_to_file("./prompt_templates.json")
    new_manager = PromptManager()
    new_manager.load_from_file("./prompt_templates.json")
    print(f"已加载模板：{new_manager.list_templates()}")