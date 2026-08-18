"""
Function calling + PDF reading against the Ollama server on DGX Spark.

Ollama accepts ONLY text and images as input -- there is no native PDF input.
So a PDF must be turned into text (route A) or into page images (route B)
on the client side, before it ever reaches the model.

Setup:
    pip install openai pymupdf
    export OLLAMA_API_KEY="<your token>"

Usage:
    python pdf_tools_demo.py paper.pdf "What is the main contribution?"
    python pdf_tools_demo.py paper.pdf "Describe figure 2" --vision --page 4
    python pdf_tools_demo.py paper.pdf "Summarise it into summary.pdf"
"""

import argparse
import base64
import json
import os
import sys

try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz  # PyMuPDF legacy import name
    except ImportError as error:
        raise ImportError(
            "PyMuPDF is required for PDF tools. Install it with "
            "`python -m pip install PyMuPDF`. If a package named `fitz` is "
            "installed and causes a `frontend` import error, uninstall it with "
            "`python -m pip uninstall fitz`."
        ) from error
from openai import OpenAI

BASE_URL = os.environ.get(
    "OLLAMA_BASE_URL", "https://spark-da32.tail67be05.ts.net:8443/v1"
)
MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.6:35b")


def create_client():
    """Create the API client only when a model request is actually made."""

    api_key = os.environ.get("OLLAMA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OLLAMA_API_KEY is not set. Export it before sending a model request."
        )
    return OpenAI(base_url=BASE_URL, api_key=api_key)


# --------------------------------------------------------------------------
# PDF helpers -- this is the part that makes a PDF readable by the model
# --------------------------------------------------------------------------

def pdf_info(path):
    """Page count + whether the PDF carries a real text layer."""
    with fitz.open(path) as doc:
        n = doc.page_count
        sample = "".join(doc[i].get_text() for i in range(min(3, n)))
    return {
        "page_count": n,
        # A scanned PDF has (almost) no extractable text -> use route B.
        "has_text_layer": len(sample.strip()) > 50,
    }


def pdf_text(path, first_page=1, last_page=None, max_chars=40000):
    """Extract text from a page range. 1-indexed, inclusive."""
    with fitz.open(path) as doc:
        last_page = last_page or doc.page_count
        first_page = max(1, first_page)
        last_page = min(doc.page_count, last_page)
        chunks = [
            f"--- page {i + 1} ---\n{doc[i].get_text()}"
            for i in range(first_page - 1, last_page)
        ]
    text = "\n".join(chunks)
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n[truncated at {max_chars} chars]"
    return text


def pdf_page_png(path, page, dpi=150):
    """Render one page to PNG bytes -- for the vision route."""
    with fitz.open(path) as doc:
        pix = doc[page - 1].get_pixmap(dpi=dpi)
        return pix.tobytes("png")


# --------------------------------------------------------------------------
# PDF output. The model cannot emit a PDF -- it emits Markdown, we render it.
# fitz.Story lays out HTML into a PDF, so no dependency beyond PyMuPDF.
# --------------------------------------------------------------------------

PAGE_CSS = """
body { font-family: sans-serif; font-size: 10pt; line-height: 1.5; }
h1 { font-size: 17pt; margin: 0 0 12pt 0; }
h2 { font-size: 13pt; margin: 14pt 0 6pt 0; }
h3 { font-size: 11pt; margin: 10pt 0 4pt 0; }
p, li { margin: 0 0 6pt 0; }
code { font-family: monospace; font-size: 9pt; }
"""


def markdown_to_pdf(md_text, output_path, title=None):
    """Render Markdown to a paginated PDF."""
    try:
        import markdown as md_lib
        body = md_lib.markdown(md_text, extensions=["tables", "fenced_code"])
    except ImportError:
        # Minimal fallback so the demo runs without `pip install markdown`.
        body = "".join(
            f"<p>{line}</p>" if line.strip() else "<br/>"
            for line in md_text.splitlines()
        )

    heading = f"<h1>{title}</h1>" if title else ""
    story = fitz.Story(html=f"<body>{heading}{body}</body>", user_css=PAGE_CSS)
    writer = fitz.DocumentWriter(output_path)

    mediabox = fitz.paper_rect("a4")
    textbox = mediabox + (72, 72, -72, -72)  # 1 inch margins

    more = True
    while more:
        dev = writer.begin_page(mediabox)
        more, _ = story.place(textbox)
        story.draw(dev)
        writer.end_page()
    writer.close()

    return output_path


def write_pdf(content, output_path, title=None):
    """Tool wrapper -- returns a string the model can read as confirmation."""
    markdown_to_pdf(content, output_path, title=title)
    with fitz.open(output_path) as doc:
        pages = doc.page_count
    return f"Wrote {output_path} ({pages} pages)."


# --------------------------------------------------------------------------
# Route A: expose the PDF helpers as tools the model can call itself
# --------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "pdf_info",
            "description": "Get the page count of a PDF and whether it has a "
                           "machine-readable text layer. Call this first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the PDF file"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pdf_text",
            "description": "Extract the plain text of a page range from a PDF. "
                           "Read a few pages at a time rather than the whole file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the PDF file"},
                    "first_page": {"type": "integer", "description": "1-indexed, inclusive"},
                    "last_page": {"type": "integer", "description": "1-indexed, inclusive"},
                },
                "required": ["path", "first_page", "last_page"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_pdf",
            "description": "Render Markdown into a PDF file. Use this when the "
                           "user asks for a report, summary or document as PDF.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string",
                                "description": "The document body, in Markdown"},
                    "output_path": {"type": "string",
                                    "description": "Where to save, e.g. 'report.pdf'"},
                    "title": {"type": "string", "description": "Document title"},
                },
                "required": ["content", "output_path"],
            },
        },
    },
]

DISPATCH = {"pdf_info": pdf_info, "pdf_text": pdf_text, "write_pdf": write_pdf}


def chat_with_tools(question, pdf_path, max_rounds=8):
    """The tool loop. The model never runs code -- it asks, we execute, we reply."""
    client = create_client()
    messages = [
        {
            "role": "system",
            "content": "You can read PDF files and write PDF files through the "
                       "provided tools. Answer only from what the tools return.",
        },
        {"role": "user", "content": f"The PDF is at '{pdf_path}'.\n\n{question}"},
    ]

    for round_no in range(max_rounds):
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            stream=False,  # tool calls need a non-streaming response
        )
        msg = resp.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            return msg.content

        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
                result = DISPATCH[name](**args)
            except Exception as exc:  # feed the error back, let the model retry
                result = f"ERROR calling {name}: {exc}"
            print(f"  [round {round_no}] {name}({tc.function.arguments}) -> "
                  f"{str(result)[:80]}...", file=sys.stderr)

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result if isinstance(result, str) else json.dumps(result),
            })

    return "[stopped: too many tool rounds]"


# --------------------------------------------------------------------------
# Route B: render a page and send it to the VLM as an image
# --------------------------------------------------------------------------

def ask_about_page_image(pdf_path, page, question, dpi=150):
    client = create_client()
    png = pdf_page_png(pdf_path, page, dpi=dpi)
    b64 = base64.b64encode(png).decode()

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                # The data URI prefix matters -- a bare base64 string is the
                # usual cause of "Failed to load image or audio file".
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        }],
    )
    return resp.choices[0].message.content


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("question")
    ap.add_argument("--vision", action="store_true",
                    help="send a rendered page image instead of extracted text")
    ap.add_argument("--page", type=int, default=1, help="page for --vision")
    args = ap.parse_args()

    info = pdf_info(args.pdf)
    print(f"{args.pdf}: {info['page_count']} pages, "
          f"text layer: {info['has_text_layer']}", file=sys.stderr)

    if args.vision:
        print(ask_about_page_image(args.pdf, args.page, args.question))
    else:
        if not info["has_text_layer"]:
            print("WARNING: no text layer -- this looks like a scanned PDF. "
                  "Use --vision instead.", file=sys.stderr)
        print(chat_with_tools(args.question, args.pdf))


if __name__ == "__main__":
    main()
