# Welcome to the Dev Tools Console!
#
# You can use Console to explore the Elasticsearch API. See the Elasticsearch API reference to learn more:
# https://www.elastic.co/guide/en/elasticsearch/reference/current/rest-apis.html
#
# Here are a few examples to get you started.


GET /_cat/indices

# 初始化滚动
POST toutiao_news/_search?scroll=10m&size=1000
{
  "query": { "match_all": {} }
}
GET /toutiao_news/_search
{
  "query": {
    "prefix": {
      "title": {
        "value": "["  // 匹配以“关键词”开头的内容
      }
    }
  }
}

GET /toutiao_news/_search
{
  "query": {
    "match": {
      "title": "人工智能 机器学习"  // 空格分隔多个关键词
    }
  }
}
GET /toutiao_news/_search
{
  "query": {
    "match_phrase": {
      "title": "人工智能应"  // 必须完整出现该短语，词序不变
    }
  }
}

GET /toutiao_news/_search
{
  "query": {
    "match": {
      "title": "["  // 空格分隔多个关键词
    }
  }
}

# Create an index
PUT /my-index


# Add a document to my-index
POST /my-index/_doc
{
    "id": "park_rocky-mountain",
    "title": "Rocky Mountain",
    "description": "Bisected north to south by the Continental Divide, this portion of the Rockies has ecosystems varying from over 150 riparian lakes to montane and subalpine forests to treeless alpine tundra."
}


# Perform a search in my-index
GET /my-index/_search?q="rocky mountain"