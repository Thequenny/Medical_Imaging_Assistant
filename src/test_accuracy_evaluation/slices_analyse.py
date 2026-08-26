from pathlib import Path
import base64
from datetime import datetime
from html import escape
import json
import nibabel as nib
import numpy as np
from PIL import Image
from textwrap import wrap

from reportlab.pdfgen import canvas

try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz  # PyMuPDF legacy import name
    except ImportError as error:
        raise ImportError(
            "PyMuPDF is required for PDF export. Install it with "
            "`python -m pip install PyMuPDF`. If a package named `fitz` is "
            "installed and causes a `frontend` import error, uninstall it with "
            "`python -m pip uninstall fitz`."
        ) from error
import argparse
import base64
import markdown as md_lib

try:
    from .api_config import QWEN_MODEL, create_qwen_client
except ImportError:
    from api_config import QWEN_MODEL, create_qwen_client


DEFAULT_SLICE_ANALYSIS_PROMPT = """
Tu recois une serie ordonnee de coupes d'imagerie medicale.
Ton objectif n'est pas d'analyser chaque coupe individuellement, mais de produire
un resume global de l'ensemble de la serie.

Redige une synthese courte en anglais avec:

1. Vue d'ensemble de ce que montrent les coupes
2. Structures ou region anatomique probablement visibles, seulement si identifiable
3. Tendance generale observee entre les coupes
4. Qualite globale des images et artefacts evidents
5. Points d'attention visibles a verifier, sans diagnostic certain


Ne decris pas les coupes une par une. Ne liste des noms de fichiers que si une
coupe semble vraiment importante a revoir. Resume tout en un ou plusieurs paragraphe, pas en 
""".strip()
DEFAULT_SLICE_ANALYSIS_BATCH_SIZE = 12
DEFAULT_SLICE_ANALYSIS_STRIDE = 8


def convert_to_png(slice_type, nifti_path, root_dir="."):
    # 0 = sagittal, 1 = coronal, 2 = axial
    if slice_type == "sagittal slice":
        axis = 0
    elif slice_type == "coronal slice":
        axis = 1
    elif slice_type == "axial slice":
        axis = 2
    else:
        raise ValueError("Unknown slice type.")
    
    image = nib.load(nifti_path)
    data = image.get_fdata(dtype=np.float32)

    if data.ndim == 4:
        data = data[:, :, :, 0]

    output_dir = Path(root_dir) / "data" / "slices"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reset old slices
    for old_image in output_dir.glob("*.png"):
        old_image.unlink()

    for z in range(data.shape[axis]):
        slice_2d = np.take(data, z, axis=axis)

        # Simple normalization between 0 and 255
        min_value = np.min(slice_2d)
        max_value = np.max(slice_2d)

        slice_2d = slice_2d - min_value
        if max_value - min_value != 0:
            slice_2d = slice_2d / (max_value - min_value)

        slice_2d = (slice_2d * 255).astype(np.uint8)
        slice_image = Image.fromarray(slice_2d)
        slice_image.save(output_dir / f"slice_{z:03d}.png")

    return output_dir, data.shape[axis]


def generate_slice_html_report(
    image_paths,
    output_path=Path("data/dataset_analysis/slice_vlm_report.html"),
    series_name=None,
    slice_type=None,
    prompt=None,
    batch_size=DEFAULT_SLICE_ANALYSIS_BATCH_SIZE,
    batch_stride=DEFAULT_SLICE_ANALYSIS_STRIDE,
    progress_callback=None,
):
    image_paths = sorted(Path(image_path) for image_path in image_paths)

    if not image_paths:
        raise ValueError("No slice images to summarize.")

    prompt = prompt or DEFAULT_SLICE_ANALYSIS_PROMPT

    if batch_size is None or batch_size <= 0:
        batch_size = len(image_paths)

    batches = _build_spaced_batches(image_paths, batch_size, batch_stride)
    batch_reports = []

    for batch_number, batch_paths in enumerate(batches, start=1):
        if progress_callback is not None:
            progress_callback(f"Summarizing batch {batch_number}/{len(batches)}...")

        batch_prompt = _build_batch_analysis_prompt(
            prompt=prompt,
            batch_paths=batch_paths,
            batch_number=batch_number,
            total_batches=len(batches),
            batch_stride=batch_stride,
        )
        batch_answer, _messages = chat_qwen(batch_prompt, messages=[], image_paths=batch_paths)
        batch_reports.append(
            {
                "batch_number": batch_number,
                "image_paths": batch_paths,
                "answer": batch_answer,
            }
        )

    if len(batch_reports) == 1:
        answer = batch_reports[0]["answer"]
    else:
        if progress_callback is not None:
            progress_callback("Building global summary...")

        synthesis_prompt = _build_synthesis_prompt(prompt, batch_reports)
        answer, _messages = chat_qwen(synthesis_prompt, messages=[], image_paths=[])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if progress_callback is not None:
        progress_callback("Writing HTML report...")

    report_html = _build_slice_html_report(
        answer=answer,
        image_paths=image_paths,
        output_path=output_path,
        series_name=series_name,
        slice_type=slice_type,
        prompt=prompt,
        batch_reports=batch_reports,
        batch_stride=batch_stride,
    )
    output_path.write_text(report_html, encoding="utf-8")

    return output_path, answer


def _build_spaced_batches(image_paths, batch_size, batch_stride):
    if batch_stride is None or batch_stride <= 1:
        return _chunk_sequence(image_paths, batch_size)

    batches = []
    remaining_paths = []

    for offset in range(min(batch_stride, len(image_paths))):
        spaced_paths = image_paths[offset::batch_stride]
        full_batch_count = len(spaced_paths) // batch_size
        full_batch_end = full_batch_count * batch_size

        for index in range(0, full_batch_end, batch_size):
            batches.append(spaced_paths[index:index + batch_size])

        remaining_paths.extend(spaced_paths[full_batch_end:])

    batches.extend(_chunk_sequence(remaining_paths, batch_size))

    return batches


def _chunk_sequence(items, chunk_size):
    return [
        items[index:index + chunk_size]
        for index in range(0, len(items), chunk_size)
    ]


def _build_batch_analysis_prompt(
    prompt,
    batch_paths,
    batch_number,
    total_batches,
    batch_stride=None,
):
    slice_names = ", ".join(path.name for path in batch_paths)
    spacing_note = ""

    if batch_stride is not None and batch_stride > 1:
        spacing_note = (
            "\nLes images de ce lot sont volontairement espacees d'environ "
            f"{batch_stride} indices pour couvrir des niveaux differents de la serie."
        )

    return f"""{prompt}

Ce lot correspond a un echantillon ordonne de la serie complete: lot {batch_number}/{total_batches}.
Les images de ce lot sont fournies dans l'ordre suivant:
{slice_names}
{spacing_note}

Resume ce lot en quelques lignes seulement. Ne decris pas les coupes une par une.
Mentionne uniquement les tendances globales, la qualite generale et les points
d'attention evidents."""


def _build_synthesis_prompt(prompt, batch_reports):
    reports_text = "\n\n".join(
        _format_batch_report_for_synthesis(batch_report)
        for batch_report in batch_reports
    )

    return f"""A partir des resumes par lot ci-dessous, redige un resume final
coherent en francais pour l'ensemble de la serie.

Respecte cette consigne generale:
{prompt}

Ne repete pas les lots. Ne fais pas une analyse coupe par coupe. Fais une
synthese globale courte, puis mentionne seulement les points d'attention les
plus importants s'ils existent.

Resumes par lot:
{reports_text}"""


def _format_batch_report_for_synthesis(batch_report):
    slice_names = ", ".join(path.name for path in batch_report["image_paths"])

    return (
        f"Lot {batch_report['batch_number']}\n"
        f"Coupes: {slice_names}\n"
        f"Resume:\n{batch_report['answer']}"
    )


def _build_slice_html_report(
    answer,
    image_paths,
    output_path,
    series_name,
    slice_type,
    prompt,
    batch_reports=None,
    batch_stride=None,
):
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    escaped_series_name = escape(series_name or "Unknown series")
    escaped_slice_type = escape(slice_type or "Unknown slice type")
    escaped_prompt = escape(prompt)
    escaped_answer = escape(answer)
    escaped_batch_stride = escape(str(batch_stride or "contiguous"))

    gallery_items = "\n".join(
        _slice_gallery_item(image_path, output_path)
        for image_path in image_paths
    )
    batch_reports_html = _batch_reports_html(batch_reports or [])

    return f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <title>VLM slice summary report</title>
  <style>
    body {{
      margin: 0;
      background: #f5f7fa;
      color: #1f2933;
      font-family: Arial, sans-serif;
      line-height: 1.5;
    }}
    main {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 32px 24px;
    }}
    h1, h2 {{
      margin: 0 0 12px;
      color: #102a43;
    }}
    h1 {{
      font-size: 28px;
    }}
    h2 {{
      margin-top: 28px;
      font-size: 20px;
    }}
    .meta {{
      background: #ffffff;
      border: 1px solid #d9e2ec;
      border-radius: 6px;
      padding: 16px;
      margin: 16px 0;
    }}
    pre {{
      white-space: pre-wrap;
      word-wrap: break-word;
      background: #ffffff;
      border: 1px solid #d9e2ec;
      border-radius: 6px;
      padding: 16px;
      font-family: Arial, sans-serif;
      font-size: 14px;
    }}
    .gallery {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
      gap: 12px;
      margin-top: 12px;
    }}
    .slice {{
      display: block;
      color: #1f2933;
      text-decoration: none;
      background: #ffffff;
      border: 1px solid #d9e2ec;
      border-radius: 6px;
      padding: 8px;
    }}
    .slice img {{
      width: 100%;
      aspect-ratio: 1;
      object-fit: contain;
      background: #000000;
    }}
    .slice span {{
      display: block;
      margin-top: 6px;
      font-size: 12px;
      text-align: center;
      word-break: break-word;
    }}
    details {{
      background: #ffffff;
      border: 1px solid #d9e2ec;
      border-radius: 6px;
      margin: 10px 0;
      padding: 12px;
    }}
    summary {{
      cursor: pointer;
      font-weight: bold;
    }}
  </style>
</head>
<body>
  <main>
    <h1>VLM slice summary report</h1>
    <section class="meta">
      <div><strong>Series:</strong> {escaped_series_name}</div>
      <div><strong>Slice type:</strong> {escaped_slice_type}</div>
      <div><strong>Images used:</strong> {len(image_paths)}</div>
      <div><strong>Batch spacing:</strong> {escaped_batch_stride}</div>
      <div><strong>Generated:</strong> {escape(generated_at)}</div>
    </section>
    <h2>Resume global</h2>
    <pre>{escaped_answer}</pre>
    <h2>Prompt utilise</h2>
    <pre>{escaped_prompt}</pre>
    {batch_reports_html}
    <h2>Coupes utilisees</h2>
    <div class="gallery">
      {gallery_items}
    </div>
  </main>
</body>
</html>
"""


def _batch_reports_html(batch_reports):
    if len(batch_reports) <= 1:
        return ""

    sections = []
    for batch_report in batch_reports:
        batch_number = batch_report["batch_number"]
        image_paths = batch_report["image_paths"]
        slice_names = ", ".join(escape(path.name) for path in image_paths)
        escaped_answer = escape(batch_report["answer"])

        sections.append(
            f"<details>"
            f"<summary>Lot {batch_number} - {len(image_paths)} coupes</summary>"
            f"<p><strong>Coupes utilisees:</strong> {slice_names}</p>"
            f"<pre>{escaped_answer}</pre>"
            f"</details>"
        )

    return "<h2>Resumes par lot</h2>\n" + "\n".join(sections)


def _slice_gallery_item(image_path, output_path):
    image_path = Path(image_path)
    output_path = Path(output_path)

    try:
        href = image_path.resolve().relative_to(output_path.parent.resolve()).as_posix()
    except ValueError:
        href = image_path.resolve().as_uri()

    escaped_href = escape(href, quote=True)
    escaped_name = escape(image_path.stem)

    return (
        f'<a class="slice" href="{escaped_href}">'
        f'<img src="{escaped_href}" alt="{escaped_name}">'
        f"<span>{escaped_name}</span>"
        "</a>"
    )
# pdf output for the chat 

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
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        
        body = md_lib.markdown(md_text, extensions=["tables", "fenced_code"])
    except ImportError:
        # Minimal fallback so the demo runs without `pip install markdown`.
        body = "".join(
            f"<p>{line}</p>" if line.strip() else "<br/>"
            for line in md_text.splitlines()
        )

    heading = f"<h1>{title}</h1>" if title else ""
    story = fitz.Story(html=f"<body>{heading}{body}</body>", user_css=PAGE_CSS)
    writer = fitz.DocumentWriter(str(output_path))

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
    with fitz.open(str(output_path)) as doc:
        pages = doc.page_count
    return f"Wrote {output_path} ({pages} pages)."

TOOLS=[{
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
                                    "description": "Where to save, e.g. 'data/chat_report.pdf'. "
                                                   "Use this default if the user does not give a path."},
                    "title": {"type": "string", "description": "Document title"},
                },
                "required": ["content", "output_path"],
            },
        },
    },]


DISPATCH = {"write_pdf": write_pdf}


def _message_to_dict(message):
    if hasattr(message, "model_dump"):
        return message.model_dump(exclude_none=True)

    return message


def _resolve_tool_output_path(output_path, root_dir):
    output_path = Path(output_path)

    if output_path.is_absolute():
        return output_path

    return Path(root_dir) / output_path


def chat_qwen(
    prompt,
    root_dir=".",
    messages=None,
    use_slices=False,
    image_paths=None,
    max_tool_rounds=4,
):
    client = create_qwen_client()

    if messages is None:
        messages = []

    task = [{"type": "text", "text": prompt}]

    if image_paths is not None:
        image_paths = [Path(image_path) for image_path in image_paths]

    elif use_slices:
        slices_dir = Path(root_dir) / "data" / "slices"
        image_paths = sorted(slices_dir.glob("slice_*.png"))

    else:
        image_paths = []

    for image_path in image_paths:
        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")

        task.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_base64}"}
        })

    messages.append({"role": "user", "content": task})

    for _round_number in range(max_tool_rounds):
        response = client.chat.completions.create(
            model=QWEN_MODEL,
            messages=messages,
            tools=TOOLS,
            stream=False
        )

        answer = response.choices[0].message
        messages.append(_message_to_dict(answer))

        if not answer.tool_calls:
            return answer.content or "", messages

        for tool_call in answer.tool_calls:
            tool_name = tool_call.function.name

            try:
                tool_args = json.loads(tool_call.function.arguments or "{}")

                if tool_name not in DISPATCH:
                    raise ValueError(f"Unknown tool: {tool_name}")

                if tool_name == "write_pdf":
                    tool_args["output_path"] = _resolve_tool_output_path(
                        tool_args["output_path"],
                        root_dir,
                    )

                tool_result = DISPATCH[tool_name](**tool_args)
            except Exception as error:
                tool_result = f"ERROR calling {tool_name}: {error}"

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(tool_result),
            })

    return "[stopped: too many tool rounds]", messages

def generate_pdf(
    text,
    output_path=Path("data/dataset_analysis/slice_report.pdf"),
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    c = canvas.Canvas(str(output_path))
    y = 800

    for line in text.splitlines():
        for part in wrap(line, width=90) or [""]:
            c.drawString(50, y, part)
            y -= 15

            if y < 50:
                c.showPage()
                y = 800

    c.save()
    return output_path
