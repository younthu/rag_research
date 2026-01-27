# ELK 本地运行说明

本目录提供一个用于本地开发/测试的 ELK（Elasticsearch + Logstash + Kibana）Docker Compose 配置示例。该示例使用 Elasticsearch 8.x 镜像（在文件中示例使用 `8.9.0`），为方便开发默认禁用 xpack 安全。请勿在生产环境直接使用本配置。

文件清单

- `docker-compose.elk.yml`：ELK 单节点 compose（Elasticsearch、Kibana、Logstash）。
- `logstash/pipeline/logstash.conf`：最小的 Logstash pipeline（将输入写到 Elasticsearch 并输出到 stdout），用于测试。

先决条件

- 已安装 Docker Engine 和 Docker Compose（在大多数系统上为 `docker compose` 命令）。
- 主机需要合适的虚拟内存设置，Elasticsearch 通常要求：

```bash
# 需 root 或 sudo
sudo sysctl -w vm.max_map_count=262144
```

启动 ELK

从仓库根或 `es/` 目录执行：

```bash
cd es
docker compose -f docker-compose.elk.yml up -d
```

等待数秒到数十秒让 Elasticsearch 初始化并变为可用。可以查看容器日志判断启动进度：

```bash
docker compose -f docker-compose.elk.yml logs -f elasticsearch
```

验证服务

- Elasticsearch 健康：

```bash
curl -sS http://localhost:9200/_cluster/health?pretty
```

- Kibana UI：在浏览器打开 http://localhost:5601

- 测试 Logstash（从主机向 Logstash 容器 stdin 发一条消息，并观察 Logstash 输出与 Elasticsearch 索引）：

```bash
echo '{"message":"hello elk"}' | docker exec -i logstash /usr/share/logstash/bin/logstash -f /usr/share/logstash/pipeline/logstash.conf
```

或者在容器内以交互模式运行 Logstash：

```bash
docker exec -it logstash /bin/bash
# 然后手工运行 logstash
/usr/share/logstash/bin/logstash -f /usr/share/logstash/pipeline/logstash.conf
```

停止与清理

- 停止容器（保留数据卷）：

```bash
docker compose -f docker-compose.elk.yml down
```

- 停止并删除数据卷（会删除索引/数据）：

```bash
docker compose -f docker-compose.elk.yml down -v
```

版本与安全说明

- 本示例中镜像使用标签 `8.9.0` 作为示例。要使用“最新版本”，可在 `docker-compose.elk.yml` 中把镜像标签替为你想要的版本（或 `latest`），但请注意不同 8.x 版本可能默认启用/更改安全策略。
- Elasticsearch 8.x 在生产模式下默认启用安全（TLS 和内置用户）。本 compose 为方便开发禁用了 `xpack.security.enabled`。如果需要生产级别或测试启用安全的 8.x 配置，我可以帮你：
  - 自动生成证书并注入到各服务，或
  - 使用官方提供的 `elasticsearch-setup-passwords` 等方法初始化内置用户并配置 Kibana/Logstash。

常见问题排查

- 容器启动失败或 Elasticsearch 频繁重启：
  - 检查 `vm.max_map_count`（见上文）。
  - 查看 Elasticsearch 日志：

```bash
docker compose -f docker-compose.elk.yml logs elasticsearch
```

  - 检查 Docker 分配的内存/CPU，Elasticsearch 启动需要足够内存；本示例将 JVM 堆设置为 `-Xms1g -Xmx1g`（可在 `docker-compose.elk.yml` 中修改 `ES_JAVA_OPTS`）。

- 无法访问 Kibana：
  - 确认 Kibana 容器是健康并且依赖的 Elasticsearch 已启动：

```bash
docker compose -f docker-compose.elk.yml ps
```

- Logstash 不发送数据到 Elasticsearch：
  - 检查 Logstash 日志：

```bash
docker compose -f docker-compose.elk.yml logs logstash
```

更多改进建议（我可以代做）

- 将 `docker-compose.elk.yml` 改为多节点 Elasticsearch 集群（3 节点）并保留 Kibana/Logstash。
- 为 Elasticsearch 8.x 配置 TLS 与内置用户（安全模式），并把 Kibana/Logstash 正确绑定到受保护的集群。
- 添加 Beats（Filebeat/Metricbeat）示例配置并在 Compose 中挂载示例数据目录。

如需我替你自动完成其中任何一项，请直接告诉我：例如“启用 8.x 安全并生成证书”，或“把 JVM 堆改成 2GB 并把 Kibana 暴露在 5601 上”。


# 导入数据

1. 导入