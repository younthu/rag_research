"""
语义搜索系统 - 基于 FAISS 和 Sentence Transformers
"""
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Union


class SemanticTitleSearch:
    """使用 FAISS 的语义标题搜索系统"""
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        初始化语义搜索系统
        
        Args:
            model_name: 使用的 sentence-transformers 模型名称
                       使用多语言模型以支持中英文混合搜索
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.titles = []
        
    def build_index(self, titles: List[str]):
        """
        构建 FAISS 索引
        
        Args:
            titles: 标题列表
        """
        print(f"Building index for {len(titles)} titles...")
        self.titles = titles
        
        # 生成标题的 embeddings
        embeddings = self.model.encode(titles, show_progress_bar=True, convert_to_numpy=True)
        embeddings = embeddings.astype('float32')
        
        # 归一化 embeddings (用于余弦相似度)
        faiss.normalize_L2(embeddings)
        
        # 创建 FAISS 索引 (使用内积，因为已归一化，等价于余弦相似度)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        
        print(f"Index built successfully with {self.index.ntotal} vectors")
        
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        搜索最相关的标题
        
        Args:
            query: 查询字符串
            top_k: 返回前 k 个结果
            
        Returns:
            List of (title, score) tuples, sorted by relevance
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # 生成查询的 embedding
        query_embedding = self.model.encode([query], convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # 搜索
        scores, indices = self.index.search(query_embedding, min(top_k, len(self.titles)))
        
        # 返回结果
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.titles):
                results.append((self.titles[idx], float(score)))
        
        return results
    
    def search_with_rank(self, query: str, target_titles: Union[str, List[str]], top_k: int = 10) -> Dict:
        """
        搜索并返回目标标题的排名信息
        
        Args:
            query: 查询字符串
            target_titles: 目标标题（单个字符串或列表）
            top_k: 搜索前 k 个结果
            
        Returns:
            包含排名信息的字典
        """
        if isinstance(target_titles, str):
            target_titles = [target_titles]
        
        results = self.search(query, top_k)
        
        # 查找目标标题的排名
        ranks = {}
        for target in target_titles:
            rank = None
            for i, (title, score) in enumerate(results, 1):
                if title == target:
                    rank = i
                    break
            ranks[target] = {
                "rank": rank,
                "found": rank is not None,
                "score": results[rank - 1][1] if rank else None
            }
        
        return {
            "query": query,
            "results": results,
            "target_ranks": ranks
        }


def load_titles(filepath: str) -> List[str]:
    """加载标题列表"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_test_cases(filepath: str) -> List[Dict]:
    """加载测试用例"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return data['test_cases']


def evaluate_test_cases(searcher: SemanticTitleSearch, test_cases: List[Dict], top_k: int = 10) -> Dict:
    """
    评估所有测试用例
    
    Args:
        searcher: 语义搜索器
        test_cases: 测试用例列表
        top_k: 搜索前 k 个结果
        
    Returns:
        评估结果字典
    """
    print(f"\n{'='*80}")
    print(f"Evaluating {len(test_cases)} test cases...")
    print(f"{'='*80}\n")
    
    results = []
    passed = 0
    failed = 0
    
    for i, case in enumerate(test_cases, 1):
        case_id = case['id']
        query = case['query']
        target = case['target']
        expected_rank = case['target_rank']
        comment = case.get('comment', '')
        
        # 处理多个目标的情况
        targets = target if isinstance(target, list) else [target]
        
        # 执行搜索
        search_result = searcher.search_with_rank(query, targets, top_k)
        
        # 判断是否通过
        passed_case = False
        rank_info = []
        
        for t in targets:
            rank_data = search_result['target_ranks'][t]
            actual_rank = rank_data['rank']
            
            if actual_rank is None:
                rank_info.append(f"{t}: Not found in top {top_k}")
            else:
                rank_info.append(f"{t}: Rank {actual_rank} (score: {rank_data['score']:.4f})")
                
                # 检查是否满足期望
                if expected_rank == "":
                    # 负向测试，不应该找到或排名应该很低
                    if actual_rank is None or actual_rank > 5:
                        passed_case = True
                elif isinstance(expected_rank, str) and expected_rank.startswith("<="):
                    max_rank = int(expected_rank[2:])
                    if actual_rank <= max_rank:
                        passed_case = True
                elif isinstance(expected_rank, int):
                    if actual_rank == expected_rank:
                        passed_case = True
        
        # 对于多目标情况，只要有一个目标满足条件就算通过
        if passed_case or (isinstance(target, list) and any(
            search_result['target_ranks'][t]['rank'] is not None and
            (isinstance(expected_rank, str) and expected_rank.startswith("<=") and 
             search_result['target_ranks'][t]['rank'] <= int(expected_rank[2:]))
            for t in targets
        )):
            passed += 1
            status = "✓ PASS"
        else:
            failed += 1
            status = "✗ FAIL"
        
        result = {
            "case_id": case_id,
            "status": status,
            "query": query,
            "target": target,
            "expected_rank": expected_rank,
            "rank_info": rank_info,
            "top_results": [(title, f"{score:.4f}") for title, score in search_result['results'][:5]],
            "comment": comment
        }
        results.append(result)
        
        # 打印进度
        print(f"[{i}/{len(test_cases)}] {status} - {case_id}")
        print(f"  Query: {query}")
        print(f"  Expected: {expected_rank}, Actual: {', '.join(rank_info)}")
        if status == "✗ FAIL":
            print(f"  Top 3 results: {[title for title, _ in search_result['results'][:3]]}")
        print()
    
    return {
        "total": len(test_cases),
        "passed": passed,
        "failed": failed,
        "pass_rate": passed / len(test_cases) * 100,
        "results": results
    }

def generate_report(evaluation_results: Dict, output_file: str = None, html_output: str = None):
    """
    生成评估报告
    
    Args:
        evaluation_results: 评估结果
        output_file: 文本报告输出文件路径（可选）
        html_output: HTML报告输出文件路径（可选）
    """
    # 生成文本报告
    report = []
    report.append("=" * 80)
    report.append("SEMANTIC SEARCH EVALUATION REPORT")
    report.append("=" * 80)
    report.append("")
    
    # 总体统计
    report.append("## Overall Statistics")
    report.append(f"Total test cases: {evaluation_results['total']}")
    report.append(f"Passed: {evaluation_results['passed']} ✓")
    report.append(f"Failed: {evaluation_results['failed']} ✗")
    report.append(f"Pass rate: {evaluation_results['pass_rate']:.2f}%")
    report.append("")
    
    # 失败的用例
    failed_cases = [r for r in evaluation_results['results'] if r['status'] == "✗ FAIL"]
    if failed_cases:
        report.append("## Failed Cases")
        report.append(f"Total failed: {len(failed_cases)}")
        report.append("")
        for result in failed_cases:
            report.append(f"### {result['case_id']} - {result['status']}")
            report.append(f"Query: {result['query']}")
            report.append(f"Target: {result['target']}")
            report.append(f"Expected rank: {result['expected_rank']}")
            report.append(f"Actual: {', '.join(result['rank_info'])}")
            report.append(f"Comment: {result['comment']}")
            report.append(f"Top 5 results:")
            for i, (title, score) in enumerate(result['top_results'], 1):
                report.append(f"  {i}. {title} (score: {score})")
            report.append("")
    
    # 成功的用例（简要）
    passed_cases = [r for r in evaluation_results['results'] if r['status'] == "✓ PASS"]
    if passed_cases:
        report.append("## Passed Cases")
        report.append(f"Total passed: {len(passed_cases)}")
        report.append("")
    
    # 分类统计
    report.append("## Category Analysis")
    report.append("(Based on test case categories)")
    report.append("")
    
    report_text = "\n".join(report)
    print(report_text)
    
    # 保存文本报告
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
            # 写入详细的JSON结果
            f.write("\n\n" + "=" * 80 + "\n")
            f.write("DETAILED RESULTS (JSON)\n")
            f.write("=" * 80 + "\n")
            f.write(json.dumps(evaluation_results, ensure_ascii=False, indent=2))
        print(f"\nText report saved to: {output_file}")
    
    # 生成HTML报告
    if html_output:
        generate_html_report(evaluation_results, html_output)


def generate_html_report(evaluation_results: Dict, output_file: str):
    """
    生成HTML格式的评估报告
    
    Args:
        evaluation_results: 评估结果
        output_file: HTML输出文件路径
    """
    import os
    
    # 读取HTML模板
    template_path = os.path.join(os.path.dirname(__file__), 'report_template.html')
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            template = f.read()
    except FileNotFoundError:
        print(f"Error: Template file not found at {template_path}")
        return
    
    # 生成表格行
    table_rows = []
    for result in evaluation_results['results']:
        status = result['status']
        status_class = 'pass' if '✓' in status else 'fail'
        
        # 处理target显示（可能是字符串或列表）
        target = result['target']
        if isinstance(target, list):
            target_display = ', '.join(target)
        else:
            target_display = target
        
        # 处理actual result
        rank_info = result['rank_info']
        if isinstance(rank_info, list):
            if any('Not found' in info for info in rank_info):
                actual_result = f'<span class="status-fail">Not found in top 10</span>'
            elif len(rank_info) > 1:
                # 多个目标
                actual_result = '<br>'.join(rank_info)
            else:
                actual_result = rank_info[0]
        else:
            actual_result = str(rank_info)
        
        # 提取分数信息
        if 'Rank' in str(rank_info) and 'score:' in str(rank_info):
            actual_result = actual_result.replace('score:', '<span class="score">')
            if actual_result.count('(') > 0:
                actual_result = actual_result.replace(')', ')</span>')
        
        # 生成Top 5结果列表
        top_results_html = '<ol>'
        for title, score in result['top_results']:
            top_results_html += f'<li>{title} ({score})</li>'
        top_results_html += '</ol>'
        
        # 生成行HTML
        row_html = f'''
            <tr data-status="{status_class}">
                <td>{result['case_id']}</td>
                <td class="status-{status_class}">{status}</td>
                <td class="query-cell">{result['query']}</td>
                <td class="target-cell">{target_display}</td>
                <td>{result['expected_rank']}</td>
                <td>{actual_result}</td>
                <td class="top-results">{top_results_html}</td>
                <td class="comment-cell">{result['comment']}</td>
            </tr>'''
        table_rows.append(row_html)
    
    # 替换模板中的占位符
    html_content = template.replace('{{TOTAL_CASES}}', str(evaluation_results['total']))
    html_content = html_content.replace('{{PASSED_CASES}}', str(evaluation_results['passed']))
    html_content = html_content.replace('{{FAILED_CASES}}', str(evaluation_results['failed']))
    html_content = html_content.replace('{{PASS_RATE}}', f"{evaluation_results['pass_rate']:.2f}")
    html_content = html_content.replace('{{TABLE_ROWS}}', '\n'.join(table_rows))
    
    # 保存HTML文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML report saved to: {output_file}")

def main():
    """主函数"""
    # 文件路径
    titles_file = "./title_search/titles.json"
    test_cases_file = "./title_search/test_cases.json"
    report_file = "./title_search/evaluation_report.txt"
    html_report_file = "./title_search/evaluation_report.html"  # 新增
    
    # 加载数据
    print("Loading data...")
    titles = load_titles(titles_file)
    test_cases = load_test_cases(test_cases_file)
    print(f"Loaded {len(titles)} titles and {len(test_cases)} test cases")
    
    # 初始化搜索器
    print("\nInitializing semantic search...")
    searcher = SemanticTitleSearch()
    searcher.build_index(titles)
    
    # 运行评估
    print("\nRunning evaluation...")
    evaluation_results = evaluate_test_cases(searcher, test_cases)
    
    # 生成报告（包括文本和HTML）
    generate_report(evaluation_results, report_file, html_report_file)

if __name__ == "__main__":
    main()

