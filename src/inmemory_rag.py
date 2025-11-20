import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from pymilvus import MilvusClient, DataType
import os
from typing import List, Tuple, Dict, Optional


class InMemoryRAG:
    def __init__(
        self,
        embedding_model_name: str = "baai/bge-m3",
        reranker_model_name: str = "baai/bge-reranker-v2-m3",
        milvus_db_path: str = "./milvus_lite_db"
    ):
        """
        初始化InMemoryRAG类
        :param embedding_model_name: 嵌入模型名称（默认baai/bge-m3）
        :param reranker_model_name: 重排模型名称（默认baai/bge-reranker-v2-m3）
        :param milvus_db_path: Milvus Lite数据库存储路径
        """
        # 初始化嵌入模型
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # 初始化重排模型
        self.reranker_model = CrossEncoder(reranker_model_name)
        
        # 初始化Milvus Lite客户端（持久化存储）
        self.milvus_client = MilvusClient(milvus_db_path)
        self.collection_name = "text_embeddings"
        
        # 初始化FAISS索引（内存计算）
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)  # 内积相似度（适合归一化向量）
        faiss.normalize_L2(self.faiss_index)  # 归一化，使内积等价于余弦相似度
        
        # 初始化或加载Milvus集合
        self._init_milvus_collection()
        
        # 加载Milvus数据到FAISS（启动时恢复内存索引）
        self._load_milvus_to_faiss()

    def _init_milvus_collection(self):
        """初始化Milvus集合结构"""
        if not self.milvus_client.has_collection(collection_name=self.collection_name):
            self.milvus_client.create_collection(
                collection_name=self.collection_name,
                schema={
                    "fields": [
                        {"name": "id", "dtype": DataType.INT64, "is_primary": True},
                        {"name": "text", "dtype": DataType.VARCHAR, "max_length": 4096},
                        {"name": "embedding", "dtype": DataType.FLOAT_VECTOR, "dim": self.embedding_dim}
                    ]
                }
            )

    def _load_milvus_to_faiss(self):
        """从Milvus加载所有数据到FAISS索引"""
        if self.milvus_client.has_collection(collection_name=self.collection_name):
            # 查询所有数据（生产环境可分页，此处简化）
            results = self.milvus_client.query(
                collection_name=self.collection_name,
                filter="id >= 0",  # 匹配所有主键
                output_fields=["id", "text", "embedding"]
            )
            
            if results:
                # 提取嵌入向量并添加到FAISS
                embeddings = np.array([item["embedding"] for item in results], dtype=np.float32)
                self.faiss_index.add(embeddings)
                
                # 缓存文本数据（id->text映射）
                self.text_cache = {item["id"]: item["text"] for item in results}
            else:
                self.text_cache = {}
        else:
            self.text_cache = {}

    def generate_embeddings(self, texts: List[str]) -> Tuple[List[np.ndarray], List[int]]:
        """
        接口1：生成文本嵌入并存储到Milvus和FAISS
        :param texts: 待生成嵌入的文本列表
        :return: (嵌入向量列表, 存储的id列表)
        """
        if not texts:
            return [], []
        
        # 生成嵌入向量
        embeddings = self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True  # 归一化，适配内积相似度
        )
        embeddings = [emb.astype(np.float32) for emb in embeddings]
        
        # 获取当前最大id，生成新id
        max_id = max(self.text_cache.keys(), default=-1)
        new_ids = [max_id + 1 + i for i in range(len(texts))]
        
        # 批量插入Milvus
        milvus_data = [
            {"id": new_ids[i], "text": texts[i], "embedding": embeddings[i].tolist()}
            for i in range(len(texts))
        ]
        self.milvus_client.insert(
            collection_name=self.collection_name,
            data=milvus_data
        )
        
        # 更新FAISS索引
        faiss_embeddings = np.stack(embeddings, axis=0)
        self.faiss_index.add(faiss_embeddings)
        
        # 更新文本缓存
        self.text_cache.update({new_ids[i]: texts[i] for i in range(len(texts))})
        
        return embeddings, new_ids

    def similarity_search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Union[str, float, int]]]:
        """
        接口2：相似度匹配（使用FAISS进行内存计算）
        :param query: 查询文本
        :param top_k: 返回前k个匹配结果
        :return: 匹配结果列表，包含text、similarity、id
        """
        if top_k <= 0:
            return []
        
        # 生成查询嵌入
        query_embedding = self.embedding_model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32).reshape(1, -1)
        
        # FAISS搜索（内积相似度，已归一化等价于余弦相似度）
        similarities, indices = self.faiss_index.search(query_embedding, min(top_k, self.faiss_index.ntotal))
        
        # 解析结果
        results = []
        for idx, sim in zip(indices[0], similarities[0]):
            if idx < 0:  # FAISS返回-1表示无匹配
                continue
            # 通过索引映射到原始id（FAISS索引顺序与插入顺序一致，即new_ids顺序）
            # 由于插入时new_ids是连续递增的，FAISS的index=0对应id=0, index=1对应id=1...
            # 这里通过缓存的id列表反向查找（更严谨的实现）
            text_id = list(self.text_cache.keys())[idx] if idx < len(self.text_cache) else None
            if text_id is None:
                continue
            
            results.append({
                "id": text_id,
                "text": self.text_cache[text_id],
                "similarity": float(sim)
            })
        
        return results

    def rerank(
        self,
        query: str,
        candidates: List[str],
        reranker_model: Optional[CrossEncoder] = None
    ) -> List[Dict[str, Union[str, float]]]:
        """
        接口3：重排候选文本
        :param query: 查询文本
        :param candidates: 待重排的候选文本列表
        :param reranker_model: 自定义重排模型（默认使用初始化的bge-reranker）
        :return: 重排后的结果列表，包含text、score（得分越高越相关）
        """
        if not candidates:
            return []
        
        # 使用指定的重排模型或默认模型
        model = reranker_model if reranker_model is not None else self.reranker_model
        
        # 构建查询-候选对
        query_candidate_pairs = [(query, candidate) for candidate in candidates]
        
        # 计算重排得分
        scores = model.predict(query_candidate_pairs).tolist()
        
        # 按得分降序排序
        reranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        # 格式化结果
        return [{"text": text, "score": float(score)} for text, score in reranked]

    def clear_data(self):
        """清空所有数据（Milvus和FAISS）"""
        # 删除Milvus集合
        if self.milvus_client.has_collection(collection_name=self.collection_name):
            self.milvus_client.drop_collection(collection_name=self.collection_name)
        
        # 重置FAISS索引
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        faiss.normalize_L2(self.faiss_index)
        
        # 清空缓存
        self.text_cache = {}
        
        # 重新初始化Milvus集合
        self._init_milvus_collection()


# 使用示例
if __name__ == "__main__":
    # 初始化RAG实例
    rag = InMemoryRAG()
    
    # 示例文本数据
    sample_texts = [
        "人工智能是研究如何使机器模拟人类智能的科学",
        "机器学习是人工智能的一个重要分支，专注于数据驱动的模型训练",
        "深度学习是机器学习的子集，使用多层神经网络进行特征学习",
        "自然语言处理是人工智能的应用领域，专注于计算机与人类语言的交互",
        "计算机视觉是人工智能的另一个应用领域，专注于让计算机理解图像内容"
    ]
    
    # 1. 生成嵌入并存储
    print("=== 生成嵌入 ===")
    embeddings, ids = rag.generate_embeddings(sample_texts)
    print(f"生成了{len(embeddings)}个嵌入，存储ID: {ids}")
    
    # 2. 相似度匹配
    print("\n=== 相似度匹配 ===")
    query = "什么是深度学习？"
    search_results = rag.similarity_search(query, top_k=3)
    for i, res in enumerate(search_results, 1):
        print(f"{i}. 文本: {res['text']}")
        print(f"   相似度: {res['similarity']:.4f}")
        print(f"   ID: {res['id']}\n")
    
    # 3. 重排
    print("=== 重排 ===")
    # 提取相似度匹配的候选文本
    candidates = [res["text"] for res in search_results]
    rerank_results = rag.rerank(query, candidates)
    for i, res in enumerate(rerank_results, 1):
        print(f"{i}. 文本: {res['text']}")
        print(f"   重排得分: {res['score']:.4f}\n")
    
    # 清空数据（可选）
    # rag.clear_data()