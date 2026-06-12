import os
import json
import httpx
import asyncio

# Configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
OUTPUTS_DIR = os.path.join("data", "outputs")


def get_latest_run_dir():
    if not os.path.exists(OUTPUTS_DIR):
        print(f"❌ Outputs directory not found: {OUTPUTS_DIR}")
        return None
    runs = [
        d
        for d in os.listdir(OUTPUTS_DIR)
        if os.path.isdir(os.path.join(OUTPUTS_DIR, d))
    ]
    if not runs:
        print("❌ No evaluation runs found.")
        return None
    runs.sort()
    return os.path.join(OUTPUTS_DIR, runs[-1])


async def analyze_results_with_llm(
    model_name: str, summary_data: dict, full_data: list
) -> str:
    print(f"🤖 Requesting LLM Analysis from Ollama ({model_name})...")

    # Create a compact text representation of the scores for the LLM to analyze
    compact_scores = []
    for item in full_data[
        :12
    ]:  # Prevent context window limits by packing key test cases
        case_id = item.get("id")
        fws = item.get("frameworks", {})
        row_summary = f"Case [{case_id}]:\n"
        for fw, metrics in fws.items():
            row_summary += (
                f"  - {fw.upper()}: "
                + ", ".join([f"{m}: {v:.2f}" for m, v in metrics.items()])
                + "\n"
            )
        compact_scores.append(row_summary)

    scores_text = "\n".join(compact_scores)
    macro_text = json.dumps(summary_data.get("macro_averages", {}), indent=2)

    prompt = f"""
You are an expert AI system evaluator and data analyst. Analyze these multi-framework RAG evaluation results.
The evaluation ran across three frameworks: RAGAS, Promptfoo, and LangSmith on the same dataset.

### MACRO AVERAGES:
{macro_text}

### SAMPLE INDIVIDUAL SCORES (TOP CASES):
{scores_text}

### ANALYSIS REQUIREMENTS:
Please write a comprehensive, rigorous evaluation analysis covering the following three sections in clean Markdown format:

1. **Cross-Framework Consistency & Operationalization**: 
    Analyze whether metrics (Faithfulness, Answer Relevance, Context Recall, Answer Correctness) produce consistent scores across frameworks. Note any severe deviations (e.g., RAGAS returning 0.0 while others score high due to parsing or structural strictness) and explain why these deviations occur based on how they operationalize constructs.

2. **Retrieval Pipeline Failure Detection**:
    Assess how reliably these automated metrics can detect subtle pipeline errors, such as context chunks that are topically related but factually misleading or lack grounding. Which framework or scoring logic appears most robust against these subtle errors?

3. **Statistical Stability & Guidance for Tuning**:
    Evaluate whether this prompt set size and corpus diversity are statistically stable enough to confidently guide production RAG pipeline tuning and prompt modifications. Provide actionable advice for the engineering team.

Write directly in Markdown. Be technical, objective, and specific.
"""

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2},
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(OLLAMA_URL, json=payload)
            if response.status_code == 200:
                return response.json().get(
                    "response", "Analysis could not be generated."
                )
            else:
                return f"Error from Ollama API: Status code {response.status_code}"
    except Exception as e:
        return f"Failed to communicate with local Ollama judge for analysis: {str(e)}"


def generate_html_report(
    run_dir: str, summary_data: dict, full_data: list, llm_analysis_md: str
):
    print("🎨 Generating Static HTML Report...")

    macro_averages = summary_data.get("macro_averages", {})
    metadata = summary_data.get("metadata", {})

    # Build Table Rows
    table_rows_html = ""
    for idx, item in enumerate(full_data):
        case_id = item.get("id", f"case_{idx}")
        question = item.get("user_input", "")
        fws = item.get("frameworks", {})

        short_q = question if len(question) <= 90 else question[:87] + "..."

        table_rows_html += f"""
        <tr class="border-b hover:bg-gray-50 text-sm">
            <td class="px-4 py-3 font-mono font-semibold text-indigo-600">{case_id}</td>
            <td class="px-4 py-3 text-gray-700" title="{question}">{short_q}</td>
        """

        for fw in ["ragas", "promptfoo", "langsmith"]:
            if fw in fws:
                m = fws[fw]
                scores_list = [v for v in m.values() if v is not None]
                avg_score = sum(scores_list) / len(scores_list) if scores_list else 0.0
                verdict_class = (
                    "bg-green-100 text-green-800"
                    if avg_score >= 0.75
                    else "bg-amber-100 text-amber-800"
                    if avg_score >= 0.4
                    else "bg-red-100 text-red-800"
                )
                verdict_text = (
                    "PASS"
                    if avg_score >= 0.75
                    else "WARN"
                    if avg_score >= 0.4
                    else "FAIL"
                )

                metrics_details = "".join(
                    [
                        f'<div class="flex justify-between text-xs border-b border-gray-100 py-0.5"><span class="text-gray-500">{k.title()}:</span> <span class="font-mono font-bold text-gray-800">{v:.2f}</span></div>'
                        for k, v in m.items()
                    ]
                )

                table_rows_html += f"""
                <td class="px-4 py-3 border-l border-gray-200">
                    <div class="flex items-center justify-between mb-1.5">
                        <span class="px-2 py-0.5 text-xs font-bold rounded-full {verdict_class}">{verdict_text}</span>
                        <span class="text-xs font-mono text-gray-400">avg: {avg_score:.2f}</span>
                    </div>
                    {metrics_details}
                </td>
                """
            else:
                table_rows_html += '<td class="px-4 py-3 border-l border-gray-200 text-gray-400 text-center italic text-xs">N/A</td>'

        table_rows_html += "</tr>"

    # Build Macro Cards
    macro_cards_html = ""
    colors = {"ragas": "indigo", "promptfoo": "emerald", "langsmith": "sky"}
    for fw, metrics in macro_averages.items():
        c = colors.get(fw, "gray")
        macro_cards_html += f"""
        <div class="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
            <div class="bg-{c}-600 px-4 py-3 text-white font-bold tracking-wide flex justify-between items-center">
                <span>{fw.upper()} Dashboard</span>
                <span class="text-xs bg-white bg-opacity-20 px-2 py-0.5 rounded">Active</span>
            </div>
            <div class="p-5 space-y-3">
        """
        for m, val in metrics.items():
            percentage = val * 100
            macro_cards_html += f"""
                <div>
                    <div class="flex justify-between text-sm font-medium text-gray-700 mb-1">
                        <span>{m.replace("_", " ").title()}</span>
                        <span class="font-mono font-bold text-gray-900">{val:.2f}</span>
                    </div>
                    <div class="w-full bg-gray-100 rounded-full h-2">
                        <div class="bg-{c}-500 h-2 rounded-full" style="width: {percentage}%"></div>
                    </div>
                </div>
            """
        macro_cards_html += "</div></div>"

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Framework RAG Evaluation Analytics</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        .prose h1 {{ font-size: 1.4rem; font-weight: 700; color: #1e293b; margin-top: 1.5rem; margin-bottom: 0.5rem; border-bottom: 2px solid #f1f5f9; padding-bottom: 0.25rem; }}
        .prose h2 {{ font-size: 1.15rem; font-weight: 600; color: #334155; margin-top: 1.25rem; margin-bottom: 0.5rem; }}
        .prose p {{ margin-bottom: 1rem; color: #475569; line-height: 1.6; }}
        .prose ul {{ list-style-type: disc; margin-left: 1.5rem; margin-bottom: 1rem; color: #475569; }}
        .prose li {{ margin-bottom: 0.25rem; }}
        .prose strong {{ color: #0f172a; font-weight: 600; }}
    </style>
</head>
<body class="bg-gray-50 min-h-screen text-gray-800 antialiased">

    <header class="bg-slate-900 text-white shadow-md">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex flex-col sm:flex-row justify-between items-center gap-4">
            <div class="flex items-center gap-3">
                <div class="p-2 bg-indigo-600 rounded-lg font-black tracking-wider text-xl">RAG</div>
                <div>
                    <h1 class="text-xl font-bold tracking-tight">Cross-Framework Comparative Analytics</h1>
                    <p class="text-xs text-slate-400">Automated multi-metric benchmarking framework matrix</p>
                </div>
            </div>
            <div class="text-right text-xs text-slate-400 space-y-0.5">
                <div><strong>Execution Timestamp:</strong> <span class="font-mono text-slate-200">{metadata.get("timestamp", "N/A")}</span></div>
                <div><strong>Target LLM Model:</strong> <span class="font-mono bg-slate-800 px-1.5 py-0.5 rounded text-indigo-300 font-semibold">{metadata.get("model", "N/A")}</span></div>
                <div><strong>Dataset Sample Size:</strong> <span class="font-mono text-slate-200">{metadata.get("total_samples", "0")} Rows</span></div>
            </div>
        </div>
    </header>

    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        
        <section>
            <div class="mb-4">
                <h2 class="text-lg font-bold text-gray-900">📊 Framework Macro-Level Aggregates</h2>
                <p class="text-sm text-gray-500">Aggregated dimension scores averaged across the entire test evaluation suite corpus.</p>
            </div>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                {macro_cards_html}
            </div>
        </section>

        <section class="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
            <div class="bg-slate-800 px-6 py-4 text-white border-b border-gray-200 flex items-center gap-2">
                <span class="text-xl">🧠</span>
                <h2 class="text-md font-bold tracking-wide">LLM Framework Diagnostic Report & Synthesis</h2>
            </div>
            <div class="p-6 md:p-8">
                <div id="llm-analysis-content" class="prose max-w-none">
                    <div class="animate-pulse space-y-4">
                        <div class="h-4 bg-gray-200 rounded w-1/4"></div>
                        <div class="h-4 bg-gray-200 rounded w-full"></div>
                        <div class="h-4 bg-gray-200 rounded w-5/6"></div>
                    </div>
                </div>
            </div>
        </section>

        <section class="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
            <div class="px-6 py-4 bg-gray-50 border-b border-gray-200">
                <h2 class="text-md font-bold text-gray-900">🎯 Per-Prompt Granular Cross-Matrix View</h2>
                <p class="text-xs text-gray-500 mt-0.5">Granular look at pass/fail thresholds and comparative metrics across individual evaluation rows.</p>
            </div>
            <div class="overflow-x-auto">
                <table class="w-full text-left border-collapse">
                    <thead>
                        <tr class="bg-gray-100 border-b border-gray-200 text-xs font-bold text-gray-600 tracking-wider uppercase">
                            <th class="px-4 py-3 w-24">Sample ID</th>
                            <th class="px-4 py-3">User Query / Context Dimension</th>
                            <th class="px-4 py-3 border-l border-gray-200 w-56">RAGAS Framework</th>
                            <th class="px-4 py-3 border-l border-gray-200 w-56">Promptfoo Framework</th>
                            <th class="px-4 py-3 border-l border-gray-200 w-56">LangSmith Framework</th>
                        </tr>
                    </thead>
                    <tbody>
                        {table_rows_html}
                    </tbody>
                </table>
            </div>
        </section>

    </main>

    <footer class="bg-white border-t border-gray-200 mt-16 py-6 text-center text-xs text-gray-400 font-medium">
        Local Multi-Framework Evaluation Stack Engine &bull; Generated Automatically
    </footer>

    <script id="raw-markdown-data" type="text/plain">{llm_analysis_md}</script>

    <script>
        document.addEventListener("DOMContentLoaded", function() {{
            const rawMd = document.getElementById("raw-markdown-data").textContent;
            document.getElementById("llm-analysis-content").innerHTML = marked.parse(rawMd);
        }});
    </script>
</body>
</html>
"""

    report_path = os.path.join(run_dir, "report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_template)
    print(f"✅ Analytics report compiled successfully! File saved at: {report_path}")


async def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate deep analytics HTML report from RAG evaluations."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3",
        help="Model to use for the analytical breakdown.",
    )
    args = parser.parse_args()

    run_dir = get_latest_run_dir()
    if not run_dir:
        return

    print(f"📂 Analyzing data found in: {run_dir}")

    summary_path = os.path.join(run_dir, "summary_report.json")
    all_path = os.path.join(run_dir, "all.json")

    if not os.path.exists(summary_path) or not os.path.exists(all_path):
        print(
            "❌ Essential execution logs (summary_report.json or all.json) are missing."
        )
        return

    with open(summary_path, "r", encoding="utf-8") as sf:
        summary_data = json.load(sf)
    with open(all_path, "r", encoding="utf-8") as af:
        full_data = json.load(af)

    llm_analysis_md = await analyze_results_with_llm(
        args.model, summary_data, full_data
    )
    generate_html_report(run_dir, summary_data, full_data, llm_analysis_md)


if __name__ == "__main__":
    asyncio.run(main())
