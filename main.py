import os
from typing import TypedDict, List
from dotenv import load_dotenv
from pypdf import PdfReader

from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI

# Local fallback splitter to avoid dependency/import issues
def simple_split_text(text: str, chunk_size: int = 2000, chunk_overlap: int = 200):
    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        # move start forward by chunk_size - overlap
        start += chunk_size - chunk_overlap

    return chunks


# ----------------------------
# Load Environment Variables
# ----------------------------
load_dotenv()


# ----------------------------
# Define Graph State
# ----------------------------
class MindmapState(TypedDict):
    input_path: str
    document_text: str
    chunks: List[str]
    structured_content: str
    markdown: str
    output_html: str


# ----------------------------
# Node 1: Load Document
# ----------------------------
def load_document(state: dict):
    reader = PdfReader(state["input_path"])
    text = ""

    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"

    return {"document_text": text}


# ----------------------------
# Node 2: Chunk Document
# ----------------------------
def chunk_document(state: dict):
    # Use local splitter to avoid relying on langchain's text_splitter
    chunks = simple_split_text(state["document_text"], chunk_size=2000, chunk_overlap=200)
    return {"chunks": chunks}


# ----------------------------
# Node 3: Structure via LLM
# ----------------------------
#llm = ChatOpenAI(
#    model="gpt-4o-mini",
#    temperature=0
#)

from langchain_groq import ChatGroq
# Using a currently supported Groq model
llm = ChatGroq(model="llama-3.1-8b-instant")

# def structure_content(state: dict):

#     full_text = "\n".join(state["chunks"])

#     prompt = f"""
#     Convert the following document into structured Markdown hierarchy.

#     RULES:
#     - Output ONLY valid Markdown.
#     - Use:
#         # Root Title
#         ## Section
#         ### Sub-section
#     - Maximum 3 levels deep.
#     - Keep concise.
#     - No explanations outside markdown.

#     Document:
#     {full_text}
#     """

#     try:
#         response = llm.invoke(prompt)
#         content = getattr(response, "content", None) or str(response)
#         return {"structured_content": content}
#     except Exception:
#         # Fallback when LLM invocation fails (no API key or network)
#         fallback = "# Document Mindmap\n\n"
#         fallback += "## Summary\n\n"
#         fallback += "- (LLM unavailable) Generated a simple summary of the document.\n\n"
#         # include a short excerpt from the document
#         excerpt = full_text[:800].strip()
#         if excerpt:
#             fallback += "### Excerpt\n\n" + excerpt.replace("\n", " ") + "\n"

#         return {"structured_content": fallback}

def structure_content(state: dict):

    full_text = "\n".join(state["chunks"])

    prompt = f"""
Create a clean, visually balanced mindmap from the document.

STYLE REQUIREMENTS:
- Output ONLY Markdown.
- Exactly 4 levels:
    # Root
    ## 3–5 Main Themes
    ### 2–3 Subtopics per theme
    #### 1–2 Supporting details per subtopic (only if truly necessary)
- Use simple, natural English.
- Keep each node 2–4 words.
- No sentences.
- No corporate buzzwords.
- No excessive detail.
- Avoid repetition.
- Keep only structurally important ideas.
- Preserve key concepts like:
    - Coverage types
    - Documentation requirements
    - Cost thresholds
    - Fraud indicators
    - Escalation rules
- Do NOT list individual part names.
- Do NOT list every rule.
- Keep branches visually balanced.

DEPTH CONTROL:
- Level 4 should only clarify important constraints (e.g., time limits, cost limits).
- Do not overuse Level 4.
- If a subtopic does not need detail, stop at Level 3.

QUALITY CHECK BEFORE OUTPUT:
- Are there more than 5 main themes? Reduce.
- Does any branch look overloaded? Simplify.
- Would this look clean in a visual mindmap? If not, simplify further.

Think like you are organizing structured notes for a smart reader.

Document:
{full_text}
"""

    response = llm.invoke(prompt)

    return {"structured_content": response.content}


# ----------------------------
# Node 4: Format Markdown
# ----------------------------
def format_markdown(state: dict):

    markdown = state["structured_content"]

    if not markdown.strip().startswith("#"):
        markdown = "# Document Mindmap\n\n" + markdown

    return {"markdown": markdown}


# ----------------------------
# Node 5: Generate Interactive Markmap
# ----------------------------
def generate_markmap(state: dict):

    html_template = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Interactive Mindmap</title>
<script src="https://cdn.jsdelivr.net/npm/markmap-autoloader"></script>
<style>
body {{
  margin: 0;
  padding: 0;
  font-family: Arial, sans-serif;
}}
.markmap {{
  width: 100vw;
  height: 100vh;
}}
</style>
</head>
<body>
<div class="markmap">
{state["markdown"]}
</div>
</body>
</html>
"""

    output_file = "generated_mindmap.html"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_template)

    return {"output_html": output_file}


# ----------------------------
# Build LangGraph
# ----------------------------
def build_graph():

    builder = StateGraph(MindmapState)

    builder.add_node("loader", load_document)
    builder.add_node("chunker", chunk_document)
    builder.add_node("structurer", structure_content)
    builder.add_node("formatter", format_markdown)
    builder.add_node("markmap", generate_markmap)

    builder.set_entry_point("loader")

    builder.add_edge("loader", "chunker")
    builder.add_edge("chunker", "structurer")
    builder.add_edge("structurer", "formatter")
    builder.add_edge("formatter", "markmap")

    return builder.compile()


# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    graph = build_graph()

    input_path = "AutoDrive_Warranty_Policy_2025.pdf"

    if not os.path.exists(input_path):
        print(f"❌ Input file not found: {input_path}")
        print("Please provide a valid PDF path or place the file in the project directory.")
        raise SystemExit(1)

    result = graph.invoke({
        "input_path": input_path
    })

    print("✅ Interactive Mindmap Generated:")
    print(result["output_html"])