from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import json
import pandas as pd

# --------------------------
# 1. 连接Elasticsearch
# --------------------------
es = Elasticsearch(
    ["http://localhost:9200"],  # ES服务地址，多个节点可写列表
    # 若有账号密码，添加如下配置
    # http_auth=("username", "password"),
    timeout=30,  # 超时时间（秒）
    max_retries=3,
    retry_on_timeout=True
)

# 测试连接
if not es.ping():
    raise ValueError("无法连接到Elasticsearch，请检查服务是否启动")
print("成功连接到Elasticsearch")


# --------------------------
# 2. 创建索引（若不存在）
# --------------------------
index_name = "t2ranking_index"  # 自定义索引名

# 定义索引映射（根据T2Ranking数据字段调整）
mapping = {
    "mappings": {
        "properties": {
            "query_id": {"type": "keyword"},  # 查询ID（不分词）
            "query_text": {"type": "text", "analyzer": "ik_max_word"},  # 查询文本（中文用ik分词）
            "doc_id": {"type": "keyword"},  # 文档ID
            "doc_text": {"type": "text", "analyzer": "ik_max_word"},  # 文档文本
            "relevance": {"type": "integer"},  # 相关性分数（如0-4）
            "split": {"type": "keyword"}  # 数据划分（如train/test）
        }
    }
}

# 若索引不存在则创建
if not es.indices.exists(index=index_name):
    es.indices.create(index=index_name, body=mapping)
    print(f"索引 {index_name} 创建成功")
else:
    print(f"索引 {index_name} 已存在")


# --------------------------
# 3. 读取T2Ranking数据
# --------------------------
def load_t2ranking_data(file_path):
    """读取T2Ranking数据（假设为JSON Lines格式）"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    print(f"成功读取 {len(data)} 条数据")
    return data

# T2Ranking数据文件路径（替换为实际路径）
data_path = "t2ranking_data.jsonl"
t2ranking_data = load_t2ranking_data(data_path)


# --------------------------
# 4. 批量写入Elasticsearch
# --------------------------
def generate_bulk_actions(data, index_name):
    """生成bulk操作的动作列表"""
    for item in data:
        # 确保字段与mapping匹配，可根据实际数据字段调整
        yield {
            "_index": index_name,
            "_source": {
                "query_id": item.get("query_id"),
                "query_text": item.get("query_text"),
                "doc_id": item.get("doc_id"),
                "doc_text": item.get("doc_text"),
                "relevance": item.get("relevance"),
                "split": item.get("split")
            }
        }

# 批量写入（每次批量处理1000条，可根据内存调整）
success, failed = bulk(
    es,
    generate_bulk_actions(t2ranking_data, index_name),
    chunk_size=1000,
    raise_on_error=False  # 出错时不中断，仅记录失败
)

print(f"批量写入完成：成功 {success} 条，失败 {len(failed)} 条")
if failed:
    print("失败详情：", failed)