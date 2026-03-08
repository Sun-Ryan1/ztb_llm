#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""向量数据库查询器（召回恢复版
"""线程安全的查询缓存系统（支持向量缓存、查询类型统计、自动过期）"""def __init__(self, max_size: int = CONFIG["cache_max_size"],
                 expire_time: int = CONFIG["cache_expire_time"]):
        self.max_size = max_size
        self.expire_time = expire_time
        self.cache = OrderedDict()
        self.vector_cache = OrderedDict()
        self.keyword_cache = OrderedDict()
        self.lock = threading.Lock()

        self.stats = {
            "hits": 0, "misses": 0,
            "vector_cache_hits": 0, "vector_cache_misses": 0,
            "keyword_cache_hits": 0, "keyword_cache_misses": 0,
            "evictions": 0, "expired_evictions": 0,
            "total_requests": 0,
            "cache_size": 0, "max_cache_size": 0,
            "writes": 0,
            "last_clear_time": time.time(),
            "query_type_stats": defaultdict(lambda: {"hits": 0, "misses": 0, "requests": 0, "hit_rate": 0.0}),
        }

        self.clean_thread = threading.Thread(target=self._periodic_clean, daemon=True)
        self.clean_thread.start()

    @staticmethod
    def _generate_key(query: str, params: Dict) -> str:
        try:
            safe_params = {}
            for k, v in params.items():
                if isinstance(v, (dict, list, tuple)):
                    safe_params[k] = json.dumps(v, sort_keys=True, ensure_ascii=False, default=str)
                else:
                    safe_params[k] = str(v)
            key_str = f"{query}_{json.dumps(safe_params, sort_keys=True)}"
            return hashlib.md5(key_str.encode()).hexdigest()
        except:
            return hashlib.md5(f"{query}_{time.time()}".encode()).hexdigest()

    @staticmethod
    def _simple_key(text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def get(self, query: str, params: Dict) -> Optional[Any]:
        with self.lock:
            self.stats["total_requests"] += 1
            key = self._generate_key(query, params)
            query_type = self._classify_query(query)

            if key in self.cache:
                entry = self.cache[key]
                if time.time()
"""
        混合检索（召回恢复版）：
        - 候选集仅来自向量检索，确保召回质量
        - BM25 仅用于对向量候选文档进行重排（得分融合）
        - BM25 得分只计算一次，通过映射表快速获取
        - 候选集内归一化 BM25
        - 保留精确匹配奖励
        """if candidate_multiple is None:
            candidate_multiple = CONFIG["hybrid_search_candidate_multiple"]

        candidate_k = top_k * candidate_multiple

        # 1. 仅向量检索获取候选集
        vec_docs = self._vector_search_optimized(query_text, candidate_k, where_filter, 0.0)

        if not vec_docs:
            return []

        # 2. 获取查询分词（缓存）
        if query_text not in self._query_tokens_cache:
            self._query_tokens_cache[query_text] = jieba.lcut(query_text)
        query_tokens = self._query_tokens_cache[query_text]

        # 3. 一次性计算 BM25 得分（整个索引）
        bm25_scores_all = self.bm25_index.get_scores(query_tokens) if self.bm25_index else None

        # 4. 构建候选文档的 BM25 得分映射（仅针对向量候选集）
        bm25_scores = {}
        if bm25_scores_all is not None:
            for doc in vec_docs:
                doc_id = doc["id"]
                if doc_id in self.bm25_doc_id_to_idx:
                    idx = self.bm25_doc_id_to_idx[doc_id]
                    bm25_scores[doc_id] = bm25_scores_all[idx]

        # 5. 候选集内归一化 BM25
        if bm25_scores:
            bm25_vals = list(bm25_scores.values())
            min_b = min(bm25_vals)
            max_b = max(bm25_vals)
            range_b = max_b
"""已废弃 - 不再使用"""
        return []

    # ---------- 向量检索 ----------
    def _vector_search_optimized(self, query_text: str, top_k: int,
                                 where_filter: Optional[Dict] = None,
                                 min_similarity: Optional[float] = None) -> List[Dict]:
        try:
            query_emb = self._encode_query(query_text)
            query_params = {
                "query_embeddings": [query_emb.tolist()],
                "n_results": max(1, top_k),
                "include": ["documents", "metadatas", "distances"]
            }
            if where_filter:
                query_params["where"] = where_filter

            results = self.collection.query(**query_params)

            docs = results.get("documents", [[]])[0] or []
            metas = results.get("metadatas", [[]])[0] or [{}] * len(docs)
            dists = results.get("distances", [[]])[0] or [1.0] * len(docs)
            ids = results.get("ids", [[]])[0] or [f"doc_{i}" for i in range(len(docs))]

            retrieved = []
            for i, (doc, meta, dist, doc_id) in enumerate(zip(docs, metas, dists, ids)):
                if not doc:
                    continue
                sim = max(0.0, 1 - dist)
                if min_similarity and sim < min_similarity:
                    continue
                retrieved.append({
                    "id": doc_id,
                    "content": doc,
                    "similarity": round(sim, 4),
                    "metadata": meta or {},
                    "rank": i+1,
                    "content_preview": doc[:200] + "..." if len(doc) > 200 else doc,
                    "retrieval_method": "vector_search"
                })
            return retrieved
        except Exception as e:
            logger.error(f"向量检索失败: {e}")
            return []

    # ---------- 统一候选框架（已废弃，保留兼容）----------
    def _get_candidates(self, query_text: str, candidate_multiple: int,
                        where_filter: Optional[Dict] = None) -> List[Dict]:
        logger.warning("_get_candidates 已废弃，请使用 _hybrid_search")
        top_k_cand = CONFIG["default_top_k"] * candidate_multiple
        return self._hybrid_search(query_text, top_k_cand, where_filter, 0.0,
                                  candidate_multiple=CONFIG["hybrid_search_candidate_multiple"])

    # ---------- 专用检索方法（均调用 _hybrid_search）----------
    def _credit_code_exact_match(self, query_text: str, top_k: int,
                                 where_filter: Optional[Dict] = None,
                                 min_similarity: Optional[float] = None) -> List[Dict]:
        match = re.search(r'91[0-9A-Z]{16}', query_text.upper())
        if not match:
            return self._hybrid_search(query_text, top_k, where_filter, min_similarity,
                                      candidate_multiple=CONFIG["credit_code_candidate_multiple"])
        credit_code = match.group(0)
        exact_where = {"credit_code": {"$eq": credit_code}}
        if where_filter:
            exact_where.update(where_filter)
        exact_results = self._vector_search_optimized(credit_code, top_k, exact_where, 0.0)
        exact_matched = []
        for doc in exact_results:
            if credit_code in doc.get("content", "").upper() or \
               credit_code in doc.get("id", "").upper():
                doc["similarity"] = CONFIG["credit_code_exact_match_score"]
                doc["exact_match"] = True
                exact_matched.append(doc)
        if exact_matched:
            return exact_matched[:top_k]
        candidates = self._vector_search_optimized(
            credit_code,
            top_k * CONFIG["credit_code_candidate_multiple"],
            where_filter, 0.0
        )
        return candidates[:top_k]

    def _address_search(self, query_text: str, top_k: int,
                        where_filter: Optional[Dict] = None,
                        min_similarity: Optional[float] = None) -> List[Dict]:
        candidates = self._hybrid_search(query_text,
                                        top_k * CONFIG["address_candidate_multiple"],
                                        where_filter, 0.0,
                                        candidate_multiple=CONFIG["hybrid_search_candidate_multiple"])
        if not candidates:
            return []
        components = self._parse_address_components(query_text)
        scored = []
        for doc in candidates:
            content = doc.get("content", "")
            vector_score = doc.get("similarity", 0.0)
            addr_score = self._calculate_address_similarity(content, components)
            total = (vector_score * CONFIG["query_type_weights"]["address"]["vector"] +
                    addr_score * CONFIG["query_type_weights"]["address"]["bm25"])
            total = min(total, 1.0)
            new_doc = doc.copy()
            new_doc.update({
                "similarity": round(total, 4),
                "vector_score": round(vector_score, 4),
                "address_score": round(addr_score, 4),
                "retrieval_method": "address_optimized"
            })
            scored.append(new_doc)
        scored.sort(key=lambda x: x["similarity"], reverse=True)
        unique = self._deduplicate_results(scored)
        if min_similarity:
            unique = [d for d in unique if d["similarity"] >= min_similarity]
        return unique[:top_k]

    def _parse_address_components(self, text: str) -> Dict[str, str]:
        comp = {}
        m = re.search(r'([\u4e00-\u9fa5]+[省])', text)
        if m: comp['province'] = m.group(1)
        m = re.search(r'([\u4e00-\u9fa5]+[市])', text)
        if m: comp['city'] = m.group(1)
        m = re.search(r'([\u4e00-\u9fa5]+[县区])', text)
        if m: comp['district'] = m.group(1)
        m = re.search(r'([\u4e00-\u9fa5]+[乡镇街道])', text)
        if m: comp['town'] = m.group(1)
        m = re.search(r'([\u4e00-\u9fa5]+[路街道路巷弄])', text)
        if m: comp['road'] = m.group(1)
        m = re.search(r'(\d+[号院附之]*)', text)
        if m: comp['number'] = m.group(1)
        return comp

    def _normalize_address(self, address: str) -> str:
        norm = re.sub(r'[^\u4e00-\u9fa50-9]', '', address)
        norm = norm.replace('大道', '路').replace('大街', '街')
        norm = norm.replace('工业区', '园区').replace('新区', '区')
        norm = norm.replace('经济开发区', '开发区').replace('高新技术产业开发区', '高新区')
        norm = norm.replace('号院', '号').replace('号附', '号').replace('号之', '号')
        return norm

    def _calculate_address_similarity(self, content: str, components: Dict[str, str]) -> float:
        norm_content = self._normalize_address(content)
        score = 0.0
        weights = CONFIG["address_component_weights"]
        for comp, val in components.items():
            if val and val in norm_content:
                score += weights.get(comp, 0.0)
        full_addr = ''.join(components.values())
        if full_addr and full_addr in norm_content:
            score += CONFIG["address_exact_match_bonus"]
        return min(score, 1.0)

    def _business_scope_search(self, query_text: str, top_k: int,
                               where_filter: Optional[Dict] = None,
                               min_similarity: Optional[float] = None) -> List[Dict]:
        candidates = self._hybrid_search(query_text,
                                        top_k * CONFIG["business_scope_candidate_multiple"],
                                        where_filter, 0.0,
                                        candidate_multiple=CONFIG["hybrid_search_candidate_multiple"])
        if not candidates:
            return []
        core_words = self._extract_business_core_words(query_text)
        scored = []
        for doc in candidates:
            content = doc.get("content", "")
            vector_score = doc.get("similarity", 0.0)
            bm25_score = doc.get("bm25_score", 0.0)
            keyword_score = 0.0
            matched = 0
            for word in core_words:
                if word in content:
                    matched += 1
                    keyword_score += 0.15
                    if word in self.business_terms or word in self.industry_terms:
                        keyword_score += 0.1
            if core_words:
                coverage = matched / len(core_words)
                keyword_score += coverage * 0.2
            total = (vector_score * 0.2 + bm25_score * 0.2 + keyword_score * 0.6)
            total = min(total, 1.0)
            new_doc = doc.copy()
            new_doc.update({
                "similarity": round(total, 4),
                "keyword_score": round(keyword_score, 4),
                "retrieval_method": "business_scope_optimized"
            })
            scored.append(new_doc)
        scored.sort(key=lambda x: x["similarity"], reverse=True)
        unique = self._deduplicate_results(scored)
        if min_similarity:
            unique = [d for d in unique if d["similarity"] >= min_similarity]
        return unique[:top_k]

    def _extract_business_core_words(self, text: str) -> List[str]:
        cleaned = re.sub(r'经营范围|经营项目|主营|兼营|业务范围|许可项目|一般项目|营业范围', '', text)
        words = []
        for term in self.business_terms:
            if term in cleaned:
                words.append(term)
        tfidf = jieba.analyse.extract_tags(cleaned, topK=5, allowPOS=('n', 'vn'))
        words.extend(tfidf)
        return list(set(words))[:10]

    def _product_keyword_search(self, query_text: str, top_k: int,
                                where_filter: Optional[Dict] = None,
                                min_similarity: Optional[float] = None) -> List[Dict]:
        patterns = [r'生产(什么|哪些)?产品', r'提供(什么|哪些)?产品',
                   r'(什么|哪些)产品', r'产品(有|包括|是)什么']
        if not any(re.search(p, query_text) for p in patterns):
            return self._hybrid_search(query_text, top_k, where_filter, min_similarity,
                                      candidate_multiple=CONFIG["hybrid_search_candidate_multiple"])
        candidates = self._hybrid_search(query_text,
                                        top_k * CONFIG["product_keyword_candidate_multiple"],
                                        where_filter, 0.0,
                                        candidate_multiple=CONFIG["hybrid_search_candidate_multiple"])
        if not candidates:
            return []
        product_words = self._extract_product_specific_words(query_text)
        scored = []
        for doc in candidates:
            content = doc.get("content", "")
            vector_score = doc.get("similarity", 0.0)
            bm25_score = doc.get("bm25_score", 0.0)
            kw_score = 0.0
            matched = []
            for word in product_words:
                if word in content:
                    matched.append(word)
                    kw_score += 0.2
                    if word in self.product_terms:
                        kw_score += 0.1
            exact_bonus = 0.0
            if product_words:
                first = product_words[0]
                if f"产品名称：{first}" in content or f"产品：{first}" in content:
                    exact_bonus += CONFIG["product_keyword_exact_match_bonus"]
            total = (vector_score * 0.3 + bm25_score * 0.2 + kw_score * 0.5 + exact_bonus)
            total = min(total, 1.0)
            new_doc = doc.copy()
            new_doc.update({
                "similarity": round(total, 4),
                "keyword_score": round(kw_score, 4),
                "matched_keywords": matched,
                "retrieval_method": "product_keyword_optimized"
            })
            scored.append(new_doc)
        scored.sort(key=lambda x: x["similarity"], reverse=True)
        unique = self._deduplicate_results(scored)
        if min_similarity:
            unique = [d for d in unique if d["similarity"] >= min_similarity]
        return unique[:top_k]

    def _extract_product_specific_words(self, text: str) -> List[str]:
        cleaned = re.sub(r'什么|哪些|有|包括|是|提供|产品|商品|货物', '', text)
        words = []
        for term in self.product_terms:
            if term in cleaned:
                words.append(term)
        seg = jieba.lcut(cleaned)
        for w in seg:
            if len(w) >= 2 and w not in words and any(c in w for c in ['机','器','材','料','设备','装置','仪器']):
                words.append(w)
        return list(set(words))[:8]

    def _product_supplier_search(self, query_text: str, top_k: int,
                                 where_filter: Optional[Dict] = None,
                                 min_similarity: Optional[float] = None) -> List[Dict]:
        candidates = self._hybrid_search(query_text,
                                        top_k * CONFIG["product_keyword_candidate_multiple"],
                                        where_filter, 0.0,
                                        candidate_multiple=CONFIG["hybrid_search_candidate_multiple"])
        if not candidates:
            return []
        product_words = self._extract_product_specific_words(query_text)
        scored = []
        for doc in candidates:
            content = doc.get("content", "")
            vector_score = doc.get("similarity", 0.0)
            bm25_score = doc.get("bm25_score", 0.0)
            supplier_score = 0.0
            for ind in ['供应', '供货', '销售', '经销', '代理', '厂家', '制造商']:
                if ind in content:
                    supplier_score += 0.08
            product_score = 0.0
            for word in product_words:
                if word in content:
                    product_score += 0.15
            synonym_bonus = 0.0
            if '提供' in query_text and any(s in content for s in ['供应','供货','销售']):
                synonym_bonus += CONFIG["product_synonym_match_bonus"]
            total = (vector_score * 0.4 + bm25_score * 0.2 + product_score * 0.3 + supplier_score + synonym_bonus)
            total = min(total, 1.0)
            new_doc = doc.copy()
            new_doc.update({
                "similarity": round(total, 4),
                "product_score": round(product_score, 4),
                "supplier_score": round(supplier_score, 4),
                "retrieval_method": "product_supplier_optimized"
            })
            scored.append(new_doc)
        scored.sort(key=lambda x: x["similarity"], reverse=True)
        unique = self._deduplicate_results(scored)
        if min_similarity:
            unique = [d for d in unique if d["similarity"] >= min_similarity]
        return unique[:top_k]

    def _legal_representative_search(self, query_text: str, top_k: int,
                                     where_filter: Optional[Dict] = None,
                                     min_similarity: Optional[float] = None) -> List[Dict]:
        name = query_text.strip()
        if re.fullmatch(r'[\u4e00-\u9fa5]{2,4}', name):
            exact_where = {"legal_rep": {"$eq": name}}
            if where_filter:
                exact_where.update(where_filter)
            exact_results = self._vector_search_optimized(name, top_k, exact_where, 0.0)
            if exact_results:
                for d in exact_results:
                    d["similarity"] = CONFIG["exact_query_match_bonus"] + 0.7
                    d["exact_match"] = True
                return exact_results[:top_k]
        candidates = self._hybrid_search(query_text,
                                        top_k * CONFIG["default_candidate_multiple"],
                                        where_filter, 0.0,
                                        candidate_multiple=CONFIG["hybrid_search_candidate_multiple"])
        if not candidates:
            return []
        rep_keywords = ['法定代表人', '法人代表', '法人', '负责人', '代表人']
        scored = []
        for doc in candidates:
            content = doc.get("content", "")
            vector_score = doc.get("similarity", 0.0)
            rep_score = sum(0.1 for kw in rep_keywords if kw in content)
            name_bonus = 0.2 if re.search(r'法定代表人[：:]?([\u4e00-\u9fa5]{2,4})', content) else 0.0
            total = vector_score + rep_score + name_bonus
            total = min(total, 1.0)
            new_doc = doc.copy()
            new_doc.update({
                "similarity": round(total, 4),
                "rep_score": round(rep_score, 4),
                "name_bonus": round(name_bonus, 4),
                "retrieval_method": "legal_representative"
            })
            scored.append(new_doc)
        scored.sort(key=lambda x: x["similarity"], reverse=True)
        unique = self._deduplicate_results(scored)
        if min_similarity:
            unique = [d for d in unique if d["similarity"] >= min_similarity]
        return unique[:top_k]

    def _zhaobiao_search(self, query_text: str, top_k: int,
                         where_filter: Optional[Dict] = None,
                         min_similarity: Optional[float] = None,
                         sub_type: str = "zhaobiao_natural") -> List[Dict]:
        candidates = self._hybrid_search(query_text,
                                        top_k * CONFIG["default_candidate_multiple"],
                                        where_filter, 0.0,
                                        candidate_multiple=CONFIG["hybrid_search_candidate_multiple"])
        if not candidates:
            return []
        subtype_keywords = {
            "zhaobiao_buyer": ["采购方", "买方", "招标人", "采购单位"],
            "zhaobiao_supplier": ["供应商", "投标方", "中标方", "供货单位"],
            "zhaobiao_contract": ["合同", "合约", "协议"],
            "zhaobiao_natural": ["招标", "投标", "采购", "中标"]
        }
        keywords = subtype_keywords.get(sub_type, subtype_keywords["zhaobiao_natural"])
        scored = []
        for doc in candidates:
            content = doc.get("content", "")
            vector_score = doc.get("similarity", 0.0)
            subtype_score = sum(0.08 for kw in keywords if kw in content)
            zhaobiao_score = sum(0.05 for kw in ["招标","投标","采购","中标","标书"] if kw in content)
            total = vector_score + subtype_score + zhaobiao_score
            total = min(total, 1.0)
            new_doc = doc.copy()
            new_doc.update({
                "similarity": round(total, 4),
                "subtype_score": round(subtype_score, 4),
                "zhaobiao_score": round(zhaobiao_score, 4),
                "retrieval_method": f"zhaobiao_{sub_type}"
            })
            scored.append(new_doc)
        scored.sort(key=lambda x: x["similarity"], reverse=True)
        unique = self._deduplicate_results(scored)
        if min_similarity:
            unique = [d for d in unique if d["similarity"] >= min_similarity]
        return unique[:top_k]

    def _law_natural_search(self, query_text: str, top_k: int,
                            where_filter: Optional[Dict] = None,
                            min_similarity: Optional[float] = None) -> List[Dict]:
        candidates = self._hybrid_search(query_text,
                                        top_k * CONFIG["default_candidate_multiple"],
                                        where_filter, 0.0,
                                        candidate_multiple=CONFIG["hybrid_search_candidate_multiple"])
        if not candidates:
            return []
        law_keywords = ['法律', '法规', '规定', '条款', '司法解释', '条例', '办法']
        answer_inds = ['解答', '解析', '说明', '解释', '回答']
        scored = []
        for doc in candidates:
            content = doc.get("content", "")
            vector_score = doc.get("similarity", 0.0)
            law_score = sum(0.07 for kw in law_keywords if kw in content)
            answer_score = sum(0.05 for ind in answer_inds if ind in content[:200])
            total = vector_score + law_score + answer_score
            total = min(total, 1.0)
            new_doc = doc.copy()
            new_doc.update({
                "similarity": round(total, 4),
                "law_score": round(law_score, 4),
                "answer_score": round(answer_score, 4),
                "retrieval_method": "law_natural"
            })
            scored.append(new_doc)
        scored.sort(key=lambda x: x["similarity"], reverse=True)
        unique = self._deduplicate_results(scored)
        if min_similarity:
            unique = [d for d in unique if d["similarity"] >= min_similarity]
        return unique[:top_k]

    def _product_name_search(self, query_text: str, top_k: int,
                             where_filter: Optional[Dict] = None,
                             min_similarity: Optional[float] = None) -> List[Dict]:
        product_name = None
        patterns = [
            r'([\u4e00-\u9fa5]{2,10})(?:型号|规格|产品名|名称)',
            r'叫什么([\u4e00-\u9fa5]{2,10})[\?？]',
            r'产品名[：:]([\u4e00-\u9fa5]{2,10})',
            r'产品名称[：:]([\u4e00-\u9fa5]{2,10})'
        ]
        for p in patterns:
            m = re.search(p, query_text)
            if m:
                for g in m.groups():
                    if g:
                        product_name = g
                        break
            if product_name:
                break
        search_query = f"产品名称 {product_name}" if product_name else query_text
        candidates = self._hybrid_search(search_query,
                                        top_k * CONFIG["default_candidate_multiple"],
                                        where_filter, 0.0,
                                        candidate_multiple=CONFIG["hybrid_search_candidate_multiple"])
        if not candidates:
            return []
        scored = []
        for doc in candidates:
            content = doc.get("content", "")
            vector_score = doc.get("similarity", 0.0)
            name_score = 0.3 if product_name and product_name in content else 0.0
            exact_bonus = 0.2 if product_name and f"产品名称：{product_name}" in content else 0.0
            total = vector_score + name_score + exact_bonus
            total = min(total, 1.0)
            new_doc = doc.copy()
            new_doc.update({
                "similarity": round(total, 4),
                "name_score": round(name_score, 4),
                "exact_bonus": round(exact_bonus, 4),
                "retrieval_method": "product_name"
            })
            scored.append(new_doc)
        scored.sort(key=lambda x: x["similarity"], reverse=True)
        unique = self._deduplicate_results(scored)
        if min_similarity:
            unique = [d for d in unique if d["similarity"] >= min_similarity]
        return unique[:top_k]

    def _legal_title_search(self, query_text: str, top_k: int,
                            where_filter: Optional[Dict] = None,
                            min_similarity: Optional[float] = None) -> List[Dict]:
        legal_names = []
        book_matches = re.findall(r'《([^》]+)》', query_text)
        legal_names.extend(book_matches)
        for p in [r'([^》]+)法', r'([^》]+)条例', r'([^》]+)规定', r'([^》]+)办法']:
            m = re.search(p, query_text)
            if m:
                legal_names.append(m.group(1))
        legal_names = list(set(legal_names))
        candidates = self._hybrid_search(query_text,
                                        top_k * CONFIG["default_candidate_multiple"],
                                        where_filter, 0.0,
                                        candidate_multiple=CONFIG["hybrid_search_candidate_multiple"])
        if not candidates:
            return []
        scored = []
        for doc in candidates:
            content = doc.get("content", "")
            vector_score = doc.get("similarity", 0.0)
            bm25_score = doc.get("bm25_score", 0.0)
            title_score = sum(0.2 for name in legal_names if name in content)
            exact_bonus = 0.0
            for name in legal_names:
                if f"《{name}》" in content or f"{name}法" in content:
                    exact_bonus += CONFIG["legal_title_exact_match_bonus"]
            total = (vector_score * 0.5 + bm25_score * 0.2 + title_score * 0.3 + exact_bonus)
            total = min(total, 1.0)
            new_doc = doc.copy()
            new_doc.update({
                "similarity": round(total, 4),
                "title_score": round(title_score, 4),
                "exact_bonus": round(exact_bonus, 4),
                "retrieval_method": "legal_title"
            })
            scored.append(new_doc)
        scored.sort(key=lambda x: x["similarity"], reverse=True)
        unique = self._deduplicate_results(scored)
        if min_similarity:
            unique = [d for d in unique if d["similarity"] >= min_similarity]
        return unique[:top_k]

    def _natural_language_search(self, query_text: str, top_k: int,
                                 where_filter: Optional[Dict] = None,
                                 min_similarity: Optional[float] = None) -> List[Dict]:
        simplified = re.sub(r'[请问咨询请教]|怎么|如何|为什么|什么|哪些|哪里|吗|呢|啊|呀|\?|？', '', query_text)
        for w in ['的', '了', '在', '和', '与', '及', '等', '相关', '有关', '关于']:
            simplified = simplified.replace(w, '')
        simplified = simplified.strip()
        if not simplified:
            simplified = query_text
        candidates = self._hybrid_search(query_text,
                                        top_k * CONFIG["nl_query_candidate_multiple"],
                                        where_filter, 0.0,
                                        candidate_multiple=CONFIG["hybrid_search_candidate_multiple"])
        if not candidates:
            return []
        query_tokens = jieba.lcut(simplified)
        scored = []
        for doc in candidates:
            content = doc.get("content", "")
            vector_score = doc.get("similarity", 0.0)
            bm25_score = doc.get("bm25_score", 0.0)
            answer_score = 0.0
            if any(ind in content[:200] for ind in ['解答', '答案', '解析', '说明', '介绍', '概述']):
                answer_score += 0.1
            if len(content) > 300 and ('。' in content or '；' in content):
                answer_score += 0.05
            matched = sum(1 for t in query_tokens if t in content)
            match_score = (matched / len(query_tokens)) * 0.2 if query_tokens else 0.0
            total = (vector_score * 0.5 + bm25_score * 0.3 + answer_score + match_score)
            total = min(total, 1.0)
            new_doc = doc.copy()
            new_doc.update({
                "similarity": round(total, 4),
                "answer_score": round(answer_score, 4),
                "match_score": round(match_score, 4),
                "retrieval_method": "natural_language_optimized"
            })
            scored.append(new_doc)
        scored.sort(key=lambda x: x["similarity"], reverse=True)
        unique = self._deduplicate_results(scored)
        if min_similarity:
            unique = [d for d in unique if d["similarity"] >= min_similarity]
        return unique[:top_k]

    # ---------- 辅助工具 ----------
    @staticmethod
    def _deduplicate_results(docs: List[Dict]) -> List[Dict]:
        seen = set()
        unique = []
        for d in docs:
            content_hash = hashlib.md5(d.get('content', '').encode()).hexdigest()[:16]
            key = f"{content_hash}_{d.get('id', '')}"
            if key not in seen:
                seen.add(key)
                unique.append(d)
        return unique

    @staticmethod
    def _is_credit_code(text: str) -> bool:
        return bool(re.search(r'91[0-9a-zA-Z]{16}', text, re.IGNORECASE))

    # ---------- 状态查询与资源释放 ----------
    def test_connection(self) -> Dict[str, Any]:
        result = {
            "model_loaded": self.embedding_model is not None,
            "db_connected": self.chroma_client is not None,
            "collection_exists": self.collection is not None,
            "bm25_index_ready": self.bm25_index is not None,
            "collection_count": self.collection.count() if self.collection else 0,
            "cache_stats": self.query_cache.get_stats() if self.query_cache else {},
            "term_dicts_loaded": {
                "industry": len(self.industry_terms),
                "product": len(self.product_terms),
                "legal": len(self.legal_terms),
                "address": len(self.address_terms),
                "business": len(self.business_terms)
            }
        }
        try:
            test = self._encode_query("测试")
            result["test_encode_success"] = True
        except:
            result["test_encode_success"] = False
        return result

    def close(self) -> None:
        logger.info("释放资源...")
        if self.embedding_model:
            try:
                self.embedding_model.cpu()
            except:
                pass
            del self.embedding_model
        if self.chroma_client:
            try:
                self.chroma_client.close()
            except:
                pass
        self.query_cache.clear()
        self._query_tokens_cache.clear()
        logger.info("资源释放完成")

# ===================== 使用示例 =====================
if __name__ == "__main__":
    queryer = OptimizedVectorDBQuery(
        db_path="/tmp/chroma_db_dsw",
        model_path="/mnt/workspace/data/modelscope/cache/bge-m3/BAAI/bge-m3",
        collection_name="rag_knowledge_base"
    )
    try:
        status = queryer.test_connection()
        print("连接状态:", json.dumps(status, indent=2, ensure_ascii=False))

        test_queries = [
            "上海仓祥绿化工程有限公司的注册地址",
            "经营范围包括哪些内容？",
            "统一社会信用代码91310118MA1J9K8D6D",
            "生产什么产品？",
            "劳动合同法有什么规定？"
        ]
        for q in test_queries:
            print(f"\n🔍 查询: {q}")
            result = queryer.query(q, top_k=3)
            print(f"类型: {result.query_type}, 方法: {result.retrieval_method}, 耗时: {result.retrieval_time:.3f}s")
            for i, doc in enumerate(result.retrieved_documents):
                print(f"  [{i+1}] {doc['similarity']:.4f} - {doc['content_preview'][:80]}...")
    finally:
        queryer.close()