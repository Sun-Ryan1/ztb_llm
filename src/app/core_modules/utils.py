"""
工具函数模块
包含文档选择、公司名称提取等通用函数
"""
import re
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def extract_company_name(query: str) -> Optional[str]:
    """从查询中提取公司名称（支持常见后缀）"""
    pattern = r'([\u4e00-\u9fa5]+(?:有限公司|有限责任公司|股份有限公司|分公司|子公司|集团公司|公司))'
    match = re.search(pattern, query)
    if match:
        return match.group(1)
    return None


def select_best_document(docs: List[Dict[str, Any]], query: str) -> Optional[Dict[str, Any]]:
    """根据查询类型和公司名称智能选择最佳文档"""
    if not docs:
        return None

    docs_sorted = sorted(docs, key=lambda x: x.get('similarity', 0), reverse=True)
    company_name = extract_company_name(query)

    type_rules = {
        "供应商": {"keywords": ["供应商", "供货商"], "threshold": 0.2},
        "注册地址": {"keywords": ["注册地址", "地址", "在哪", "位置", "位于"], "threshold": 0.6},
        "法定代表人": {"keywords": ["法定代表人", "法人代表"], "threshold": 0.5},
        "价格": {"keywords": ["价格", "售价", "多少钱"], "threshold": 0.3},
        "中标供应商": {"keywords": ["中标供应商", "中标单位"], "threshold": 0.4},
        "采购方": {"keywords": ["采购方", "招标人", "采购人"], "threshold": 0.35},
        "规格": {"keywords": ["规格", "型号", "技术参数", "参数"], "threshold": 0.3},
        "法规": {"keywords": ["法", "主要内容", "规定", "条例"], "threshold": 0.3},
    }

    query_type = "general"
    for t, rule in type_rules.items():
        if any(kw in query for kw in rule["keywords"]):
            query_type = t
            break

    # ===== 特殊处理：地址查询优先选择包含"注册地址"的文档 =====
    if query_type == "注册地址":
        precise_docs = [d for d in docs_sorted if "注册地址" in d.get('content', '')]
        if precise_docs:
            precise_docs.sort(key=lambda x: x.get('similarity', 0), reverse=True)
            best = precise_docs[0]
            if best.get('similarity', 0) > type_rules["注册地址"]["threshold"]:
                return best

    # ---------- 1. 公司名称精确匹配 ----------
    if company_name:
        if query_type in type_rules:
            keywords = type_rules[query_type]["keywords"]
            company_docs = [d for d in docs_sorted if company_name in d.get('content', '') and any(kw in d.get('content', '') for kw in keywords)]
        else:
            # 对于 general 类型，不要求关键词，但按内容长度排序
            company_docs = [d for d in docs_sorted if company_name in d.get('content', '')]
        if company_docs:
            if query_type == "general":
                # 对于 general 类型，优先选内容最长的
                company_docs.sort(key=lambda x: len(x.get('content', '')), reverse=True)
                best = company_docs[0]
                sim = best.get('similarity', 0)
                if sim > 0.3:
                    logger.info(f"公司名称匹配（general选最长） (company={company_name}, sim={sim})")
                    return best
            else:
                company_docs.sort(key=lambda x: x.get('similarity', 0), reverse=True)
                best = company_docs[0]
                sim = best.get('similarity', 0)
                if sim > 0.3:
                    logger.info(f"公司名称精确匹配 (company={company_name}, sim={sim})")
                    return best

    # ---------- 2. 按查询类型的关键词筛选 ----------
    if query_type in type_rules:
        rule = type_rules[query_type]

        if query_type == "注册地址":
            address_docs = [d for d in docs_sorted if any(kw in d.get('content', '') for kw in rule["keywords"])]
            if address_docs:
                address_docs.sort(key=lambda x: x.get('similarity', 0), reverse=True)
                best = address_docs[0]
                if best.get('similarity', 0) > rule["threshold"]:
                    return best

        elif query_type == "供应商":
            # 简化：直接返回相似度最高的文档
            filtered_docs = [d for d in docs_sorted if any(kw in d.get('content', '') for kw in rule["keywords"])]
            if filtered_docs:
                valid_docs = [d for d in filtered_docs if d.get('similarity', 0) > rule["threshold"]]
                if valid_docs:
                    valid_docs.sort(key=lambda x: x.get('similarity', 0), reverse=True)
                    best = valid_docs[0]
                    logger.info(f"供应商查询选相似度最高文档 (sim={best['similarity']})")
                    return best

        elif query_type == "法规":
            main_docs = [d for d in docs_sorted if "主要内容" in d.get('content', '')]
            if main_docs:
                main_docs.sort(key=lambda x: x.get('similarity', 0), reverse=True)
                best = main_docs[0]
                if best.get('similarity', 0) > rule["threshold"]:
                    return best
            filtered_docs = [d for d in docs_sorted if any(kw in d.get('content', '') for kw in rule["keywords"])]
            if filtered_docs:
                filtered_docs.sort(key=lambda x: x.get('similarity', 0), reverse=True)
                best = filtered_docs[0]
                if best.get('similarity', 0) > rule["threshold"]:
                    return best

        else:
            filtered_docs = [d for d in docs_sorted if any(kw in d.get('content', '') for kw in rule["keywords"])]
            if filtered_docs:
                filtered_docs.sort(key=lambda x: x.get('similarity', 0), reverse=True)
                best = filtered_docs[0]
                if best.get('similarity', 0) > rule["threshold"]:
                    return best
            else:
                if query_type == "规格":
                    return None

    # ---------- 3. 通用查询 ----------
    if query_type == "general":
        candidate_docs = [d for d in docs_sorted if d.get('similarity', 0) > 0.5]
        if candidate_docs:
            candidate_docs.sort(key=lambda x: len(x.get('content', '')), reverse=True)
            best = candidate_docs[0]
            logger.info(f"通用查询选最长文档 (sim={best['similarity']})")
            return best
        else:
            if docs_sorted:
                top_docs = docs_sorted[:5]
                top_docs.sort(key=lambda x: len(x.get('content', '')), reverse=True)
                best = top_docs[0]
                logger.info(f"通用查询无高相似度，选相似度前5中最长 (sim={best['similarity']})")
                return best

    # ---------- 4. 高相似度文档保底 ----------
    if docs_sorted[0].get('similarity', 0) > 0.7:
        logger.info(f"高相似度文档 (sim={docs_sorted[0]['similarity']})")
        return docs_sorted[0]

    return None