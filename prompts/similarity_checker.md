帮我实现一个文本相似度匹配的类InMemoryRAG，实现一下功能:
1. 接口1， 生成embedding, 接受string数组，通过指定的模型来生成embedding, 默认baai_bge_m3。
1. 接口2，相似度匹配，接受一个query, 一个top_k, 返回前k个值，还有相似度.
2. 接口3, reranker, 接受一个query, 一个[str], 一个reranker, 默认baai_bge_reranker_v2_m3。返回reranker后的数组，包含str和得分。
1. 用milvus lite+本地文件做embedding存储，仅用miluvs lite做persistence, 不要用milvus做search.
1. 用SentenceTransformers做模型管理。
1. 用FAISS做内存计算。
1. 只要生成一个文件SimilarityChecker, 不要生成其它文件。