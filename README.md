Step 0: Initialise Miniconda with: export PATH=~/miniconda3/bin:$PATH

Step 1: uvicorn app:app --reload --host 0.0.0.0

Sample Input:
{
  "selected_files": ["20241230_cis_reg_no_17_0001_1735556104.pdf"],
  "selected_page_ranges": { "20241230_cis_reg_no_17_0001_1735556104.pdf": [1, 6] },
  "prompt": "Summarize key allegations from these pages",
  "top_k": 5,
  "last_messages": ["Previously we discussed the notice date"],
  "web_search": true,
  "llm_model": "Claude 3.7 Sonnet",
  "draft_mode": false,
  "analyse_mode": true
}