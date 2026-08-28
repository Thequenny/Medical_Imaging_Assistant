import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from tkinter import ttk
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
try:
    from .test_accuracy_evaluation.slices_analyse import (
        chat_qwen,
        convert_to_png,
        generate_pdf,
    )
except ImportError:
    from test_accuracy_evaluation.slices_analyse import (
        chat_qwen,
        convert_to_png,
        generate_pdf,
    )
from PIL import Image, ImageTk


try:
    from tkinterweb import HtmlFrame
except ImportError:
    HtmlFrame = None


APP_COLORS = {
    "background": "#F7F7F8",
    "surface": "#FFFFFF",
    "surface_muted": "#F0F2F5",
    "border": "#D9D9E3",
    "text": "#202123",
    "muted": "#6E6E80",
    "accent": "#10A37F",
    "accent_hover": "#0D8F70",
    "accent_soft": "#E7F5EF",
    "error": "#B42318",
}


def linux_monitor_geometries():
    """Return active X11 monitor rectangles without adding a dependency."""
    if not sys.platform.startswith("linux"):
        return []

    try:
        result = subprocess.run(
            ["xrandr", "--listmonitors"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=1,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return []

    monitor_pattern = re.compile(
        r"^\s*\d+:\s+\S+\s+(\d+)/\d+x(\d+)/\d+([+-]\d+)([+-]\d+)"
    )
    monitors = []
    for line in result.stdout.splitlines():
        match = monitor_pattern.match(line)
        if match:
            width, height, x_position, y_position = map(int, match.groups())
            monitors.append((x_position, y_position, width, height))
    return monitors


def windows_monitor_geometry(pointer_x, pointer_y):
    """Return the Windows work area containing the pointer, when available."""
    if not sys.platform.startswith("win"):
        return None

    try:
        import ctypes

        class Point(ctypes.Structure):
            _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

        class Rect(ctypes.Structure):
            _fields_ = [
                ("left", ctypes.c_long),
                ("top", ctypes.c_long),
                ("right", ctypes.c_long),
                ("bottom", ctypes.c_long),
            ]

        class MonitorInfo(ctypes.Structure):
            _fields_ = [
                ("cbSize", ctypes.c_ulong),
                ("rcMonitor", Rect),
                ("rcWork", Rect),
                ("dwFlags", ctypes.c_ulong),
            ]

        user32 = ctypes.windll.user32
        user32.MonitorFromPoint.argtypes = [Point, ctypes.c_ulong]
        user32.MonitorFromPoint.restype = ctypes.c_void_p
        user32.GetMonitorInfoW.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(MonitorInfo),
        ]
        user32.GetMonitorInfoW.restype = ctypes.c_int
        monitor = user32.MonitorFromPoint(Point(pointer_x, pointer_y), 2)
        monitor_info = MonitorInfo()
        monitor_info.cbSize = ctypes.sizeof(MonitorInfo)
        if not user32.GetMonitorInfoW(monitor, ctypes.byref(monitor_info)):
            return None

        work_area = monitor_info.rcWork
        return (
            work_area.left,
            work_area.top,
            work_area.right - work_area.left,
            work_area.bottom - work_area.top,
        )
    except (AttributeError, OSError):
        return None


def monitor_geometry(window):
    """Find the monitor under the pointer, falling back to Tk's screen."""
    try:
        pointer_x, pointer_y = window.winfo_pointerxy()
    except tk.TclError:
        pointer_x = window.winfo_screenwidth() // 2
        pointer_y = window.winfo_screenheight() // 2

    windows_monitor = windows_monitor_geometry(pointer_x, pointer_y)
    if windows_monitor is not None:
        return windows_monitor

    monitors = linux_monitor_geometries()
    if monitors:
        for x_position, y_position, width, height in monitors:
            if (
                x_position <= pointer_x < x_position + width
                and y_position <= pointer_y < y_position + height
            ):
                return x_position, y_position, width, height

        return min(
            monitors,
            key=lambda monitor: (
                pointer_x - (monitor[0] + monitor[2] / 2)
            ) ** 2
            + (
                pointer_y - (monitor[1] + monitor[3] / 2)
            ) ** 2,
        )

    try:
        return (
            window.winfo_vrootx(),
            window.winfo_vrooty(),
            window.winfo_vrootwidth(),
            window.winfo_vrootheight(),
        )
    except tk.TclError:
        return 0, 0, window.winfo_screenwidth(), window.winfo_screenheight()


def monitor_scale(monitor_width, monitor_height):
    """Scale large layouts while preserving their base size on small screens."""
    relative_scale = min(monitor_width / 1920, monitor_height / 1080)
    return min(max(relative_scale, 1.0), 1.75)


def configure_window(
    window,
    width,
    height,
    min_width=None,
    min_height=None,
    adaptive=True,
    target_monitor=None,
):
    """Size and center a window inside the monitor where it is opened."""
    if target_monitor is None:
        target_monitor = monitor_geometry(window)
    monitor_x, monitor_y, monitor_width, monitor_height = target_monitor

    margin = max(int(min(monitor_width, monitor_height) * 0.04), 24)
    available_width = max(monitor_width - (2 * margin), 1)
    available_height = max(monitor_height - (2 * margin), 1)
    scale = monitor_scale(monitor_width, monitor_height) if adaptive else 1.0
    window_width = min(max(int(width * scale), 1), available_width)
    window_height = min(max(int(height * scale), 1), available_height)
    x_position = monitor_x + max((monitor_width - window_width) // 2, 0)
    y_position = monitor_y + max((monitor_height - window_height) // 2, 0)

    window.geometry(
        f"{window_width}x{window_height}{x_position:+d}{y_position:+d}"
    )

    if min_width is not None and min_height is not None:
        window.minsize(
            min(min_width, available_width),
            min(min_height, available_height),
        )

    return window_width, window_height


windows = tk.Tk()
windows.title("Medical Dataset Workspace")
windows.configure(background=APP_COLORS["background"])
configure_window(windows, width=820, height=560, min_width=700, min_height=500)


def configure_ttk_styles():
    style = ttk.Style(windows)
    if "clam" in style.theme_names():
        style.theme_use("clam")

    style.configure(
        "App.TCombobox",
        fieldbackground=APP_COLORS["surface"],
        background=APP_COLORS["surface"],
        foreground=APP_COLORS["text"],
        bordercolor=APP_COLORS["border"],
        lightcolor=APP_COLORS["border"],
        darkcolor=APP_COLORS["border"],
        padding=7,
        arrowsize=14,
    )
    style.map(
        "App.TCombobox",
        fieldbackground=[("readonly", APP_COLORS["surface"])],
        selectbackground=[("readonly", APP_COLORS["surface"])],
        selectforeground=[("readonly", APP_COLORS["text"])],
    )
    style.configure(
        "App.TNotebook",
        background=APP_COLORS["background"],
        borderwidth=0,
        tabmargins=(0, 0, 0, 0),
    )
    style.configure(
        "App.TNotebook.Tab",
        background=APP_COLORS["surface_muted"],
        foreground=APP_COLORS["muted"],
        padding=(16, 9),
        borderwidth=0,
    )
    style.map(
        "App.TNotebook.Tab",
        background=[("selected", APP_COLORS["surface"])],
        foreground=[("selected", APP_COLORS["text"])],
    )
    style.configure(
        "App.TCheckbutton",
        background=APP_COLORS["surface"],
        foreground=APP_COLORS["text"],
        padding=6,
    )
    style.map(
        "App.TCheckbutton",
        background=[("active", APP_COLORS["surface"])],
    )
    style.configure(
        "App.Vertical.TScrollbar",
        background=APP_COLORS["border"],
        troughcolor=APP_COLORS["background"],
        bordercolor=APP_COLORS["background"],
        arrowcolor=APP_COLORS["muted"],
    )


configure_ttk_styles()


def make_button(parent, text, command, variant="primary", **options):
    if variant == "primary":
        background = APP_COLORS["accent"]
        foreground = "#FFFFFF"
        active_background = APP_COLORS["accent_hover"]
        relief = "flat"
        borderwidth = 0
    elif variant == "ghost":
        background = str(parent.cget("background"))
        foreground = APP_COLORS["text"]
        active_background = APP_COLORS["surface_muted"]
        relief = "flat"
        borderwidth = 0
    else:
        background = APP_COLORS["surface"]
        foreground = APP_COLORS["text"]
        active_background = APP_COLORS["surface_muted"]
        relief = "solid"
        borderwidth = 1

    button_options = {
        "text": text,
        "command": command,
        "background": background,
        "foreground": foreground,
        "activebackground": active_background,
        "activeforeground": foreground,
        "disabledforeground": APP_COLORS["muted"],
        "relief": relief,
        "borderwidth": borderwidth,
        "highlightthickness": 0,
        "cursor": "hand2",
        "font": ("TkDefaultFont", 10, "bold"),
        "padx": 16,
        "pady": 9,
    }
    button_options.update(options)
    button = tk.Button(parent, **button_options)
    return button


def make_card(parent, **options):
    return tk.Frame(
        parent,
        background=APP_COLORS["surface"],
        highlightbackground=APP_COLORS["border"],
        highlightthickness=1,
        **options,
    )


def add_page_heading(parent, title, subtitle=None, back_command=None):
    heading = tk.Frame(parent, background=APP_COLORS["background"])
    heading.pack(fill="x", padx=24, pady=(20, 14))

    if back_command is not None:
        make_button(
            heading,
            "← Back",
            back_command,
            variant="ghost",
            padx=10,
            pady=6,
        ).pack(side="left", padx=(0, 12))

    text_frame = tk.Frame(heading, background=APP_COLORS["background"])
    text_frame.pack(side="left", fill="x", expand=True)
    tk.Label(
        text_frame,
        text=title,
        background=APP_COLORS["background"],
        foreground=APP_COLORS["text"],
        font=("TkDefaultFont", 17, "bold"),
    ).pack(anchor="w")
    if subtitle:
        tk.Label(
            text_frame,
            text=subtitle,
            background=APP_COLORS["background"],
            foreground=APP_COLORS["muted"],
        ).pack(anchor="w", pady=(3, 0))

    return heading


project_dir = Path(__file__).resolve().parents[1]
report_dir = project_dir / "data" / "dataset_analysis"
json_report = report_dir / "analyse_dataset.json"
html_report = report_dir / "report.html"
dataset_analyzer_module = "src.dataset_analyzer"
report_module = "src.report"
launch_monitor = monitor_geometry(windows)
launch_monitor_scale = min(
    max(monitor_scale(launch_monitor[2], launch_monitor[3]), 0.85),
    1.45,
)
thumbnail_edge = min(max(int(220 * launch_monitor_scale), 180), 320)
THUMBNAIL_SIZE = (thumbnail_edge, thumbnail_edge)
SLICE_CARD_MIN_WIDTH = THUMBNAIL_SIZE[0] + 30
SELECTED_SLICE_BACKGROUND = APP_COLORS["accent_soft"]
DEFAULT_SLICE_BACKGROUND = APP_COLORS["surface"]
selected_slice_paths = set()
open_report_tabs = []


def widget_exists(widget):
    try:
        return widget.winfo_exists()
    except tk.TclError:
        return False


def run_in_background(work, on_success, on_error):
    result_queue = queue.Queue()

    def worker():
        try:
            result_queue.put(("success", work()))
        except Exception as error:
            result_queue.put(("error", error))

    def poll_result():
        try:
            status, payload = result_queue.get_nowait()
        except queue.Empty:
            if widget_exists(windows):
                windows.after(100, poll_result)
            return

        if status == "success":
            on_success(payload)
        else:
            on_error(payload)

    threading.Thread(target=worker, daemon=True).start()
    windows.after(100, poll_result)



# chat interface
def open_chat_window(image_paths=None):
    attached_image_paths = tuple(
        sorted(
            Path(path)
            for path in (
                image_paths
                if image_paths is not None
                else selected_slice_paths
            )
        )
    )

    colors = {
        **APP_COLORS,
        "user_bubble": "#E7F5EF",
        "assistant_bubble": "#FFFFFF",
        "error_bubble": "#FDECEC",
    }

    chat_window = tk.Toplevel(windows)
    chat_window.title("Qwen — Slice analysis")
    chat_window.configure(background=colors["background"])
    chat_window.grid_columnconfigure(0, weight=1)
    chat_window.grid_rowconfigure(1, weight=1)
    configure_window(
        chat_window,
        width=1100,
        height=800,
        min_width=850,
        min_height=650,
    )

    messages = []
    transcript = []

    header = tk.Frame(
        chat_window,
        background=colors["surface"],
        highlightbackground=colors["border"],
        highlightthickness=1,
    )
    header.grid(row=0, column=0, sticky="ew")

    header_text = tk.Frame(header, background=colors["surface"])
    header_text.pack(side="left", fill="x", expand=True, padx=24, pady=15)

    tk.Label(
        header_text,
        text="Qwen Slice Assistant",
        background=colors["surface"],
        foreground=colors["text"],
        font=("TkDefaultFont", 17, "bold"),
    ).pack(anchor="w")
    tk.Label(
        header_text,
        text="Discuss the selected medical images in a continuous conversation",
        background=colors["surface"],
        foreground=colors["muted"],
    ).pack(anchor="w", pady=(3, 0))

    header_actions = tk.Frame(header, background=colors["surface"])
    header_actions.pack(side="right", padx=24, pady=15)

    status_label = tk.Label(
        header_actions,
        text="Ready",
        background=colors["surface"],
        foreground=colors["accent"],
        font=("TkDefaultFont", 10, "bold"),
    )
    status_label.pack(side="right", padx=(14, 0))

    conversation_container = tk.Frame(
        chat_window,
        background=colors["background"],
    )
    conversation_container.grid(row=1, column=0, sticky="nsew")
    conversation_container.grid_columnconfigure(0, weight=1)
    conversation_container.grid_rowconfigure(0, weight=1)

    conversation_canvas = tk.Canvas(
        conversation_container,
        background=colors["background"],
        borderwidth=0,
        highlightthickness=0,
    )
    conversation_scrollbar = ttk.Scrollbar(
        conversation_container,
        orient="vertical",
        command=conversation_canvas.yview,
    )
    conversation_frame = tk.Frame(
        conversation_canvas,
        background=colors["background"],
    )
    conversation_window = conversation_canvas.create_window(
        (0, 0),
        window=conversation_frame,
        anchor="nw",
    )
    conversation_canvas.configure(yscrollcommand=conversation_scrollbar.set)
    conversation_canvas.grid(row=0, column=0, sticky="nsew")
    conversation_scrollbar.grid(row=0, column=1, sticky="ns")

    def update_conversation_scrollregion(event=None):
        conversation_canvas.configure(
            scrollregion=conversation_canvas.bbox("all")
        )

    def resize_conversation(event):
        conversation_canvas.itemconfigure(
            conversation_window,
            width=event.width,
        )

    conversation_frame.bind("<Configure>", update_conversation_scrollregion)
    conversation_canvas.bind("<Configure>", resize_conversation)
    chat_window.bind(
        "<MouseWheel>",
        lambda event: conversation_canvas.yview_scroll(
            int(-event.delta / 120),
            "units",
        ),
    )
    chat_window.bind(
        "<Button-4>",
        lambda event: conversation_canvas.yview_scroll(-1, "units"),
    )
    chat_window.bind(
        "<Button-5>",
        lambda event: conversation_canvas.yview_scroll(1, "units"),
    )

    def scroll_to_latest_message():
        if not widget_exists(conversation_canvas):
            return
        conversation_canvas.update_idletasks()
        conversation_canvas.yview_moveto(1.0)

    def add_message(role, text, detail=None, error=False):
        row = tk.Frame(
            conversation_frame,
            background=colors["background"],
        )
        row.pack(fill="x", padx=26, pady=8)

        is_user = role == "user"
        message_column = tk.Frame(
            row,
            background=colors["background"],
        )
        message_column.pack(
            side="right" if is_user else "left",
            anchor="e" if is_user else "w",
            padx=(180, 0) if is_user else (0, 180),
        )

        tk.Label(
            message_column,
            text="You" if is_user else "Qwen",
            background=colors["background"],
            foreground=colors["muted"],
            font=("TkDefaultFont", 9, "bold"),
        ).pack(anchor="e" if is_user else "w", pady=(0, 4))

        bubble_color = (
            colors["user_bubble"]
            if is_user
            else colors["error_bubble"] if error else colors["assistant_bubble"]
        )
        bubble = tk.Frame(
            message_column,
            background=bubble_color,
            highlightbackground=colors["border"],
            highlightthickness=0 if is_user else 1,
            padx=15,
            pady=11,
        )
        bubble.pack(anchor="e" if is_user else "w")

        tk.Label(
            bubble,
            text=text,
            background=bubble_color,
            foreground=colors["text"],
            justify="left",
            anchor="w",
            wraplength=680,
        ).pack(anchor="w")

        if detail:
            tk.Label(
                bubble,
                text=detail,
                background=bubble_color,
                foreground=colors["muted"],
                justify="left",
                anchor="w",
                wraplength=680,
                font=("TkDefaultFont", 9),
            ).pack(anchor="w", pady=(7, 0))

        chat_window.after_idle(scroll_to_latest_message)
        return row

    if attached_image_paths:
        image_names = ", ".join(
            path.name for path in attached_image_paths[:3]
        )
        if len(attached_image_paths) > 3:
            image_names += f" and {len(attached_image_paths) - 3} more"
        attachment_text = (
            f"{len(attached_image_paths)} slice(s) attached — {image_names}"
        )
        welcome_text = (
            "The selected slices are ready. Ask a question about their "
            "content, quality, anatomy, or visible abnormalities."
        )
    else:
        attachment_text = "No slices attached"
        welcome_text = (
            "No slices are attached yet. You can still ask a text question, "
            "or return to the slice gallery and select images first."
        )

    add_message("assistant", welcome_text, attachment_text)

    composer = tk.Frame(
        chat_window,
        background=colors["surface"],
        highlightbackground=colors["border"],
        highlightthickness=1,
    )
    composer.grid(row=2, column=0, sticky="ew")

    composer_content = tk.Frame(composer, background=colors["surface"])
    composer_content.pack(fill="x", padx=24, pady=(12, 8))
    composer_content.grid_columnconfigure(0, weight=1)

    tk.Label(
        composer_content,
        text=attachment_text,
        background=colors["surface"],
        foreground=colors["muted"],
        anchor="w",
    ).grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 7))

    prompt_border = tk.Frame(
        composer_content,
        background=colors["border"],
        padx=1,
        pady=1,
    )
    prompt_border.grid(row=1, column=0, sticky="ew", padx=(0, 10))

    prompt_entry = tk.Text(
        prompt_border,
        height=4,
        wrap="word",
        relief="flat",
        borderwidth=0,
        background=colors["surface"],
        foreground=colors["text"],
        insertbackground=colors["text"],
        padx=12,
        pady=10,
        undo=True,
    )
    prompt_entry.pack(fill="both", expand=True)

    def set_composer_enabled(enabled):
        prompt_entry.config(state="normal" if enabled else "disabled")
        send_button.config(state="normal" if enabled else "disabled")
        if enabled:
            prompt_entry.focus_set()

    def send_prompt():
        nonlocal messages

        prompt = prompt_entry.get("1.0", "end-1c").strip()
        if not prompt or str(send_button.cget("state")) == "disabled":
            return

        prompt_entry.delete("1.0", tk.END)
        attachment_detail = (
            f"{len(attached_image_paths)} attached slice(s)"
            if attached_image_paths
            else None
        )
        add_message("user", prompt, attachment_detail)
        transcript.append(("You", prompt))
        pending_row = add_message("assistant", "Qwen is thinking…")
        status_label.config(text="Thinking…", foreground=colors["muted"])
        set_composer_enabled(False)

        request_messages = list(messages)

        def work():
            return chat_qwen(
                prompt,
                root_dir=project_dir,
                messages=request_messages,
                image_paths=attached_image_paths,
            )

        def on_success(result):
            nonlocal messages
            if not widget_exists(chat_window):
                return

            answer, messages = result
            if widget_exists(pending_row):
                pending_row.destroy()
            add_message("assistant", answer)
            transcript.append(("Qwen", answer))
            status_label.config(text="Ready", foreground=colors["accent"])
            set_composer_enabled(True)

        def on_error(error):
            if not widget_exists(chat_window):
                return

            if widget_exists(pending_row):
                pending_row.destroy()
            add_message(
                "assistant",
                "The request could not be completed. You can edit your "
                "message and try again.",
                detail=str(error),
                error=True,
            )
            status_label.config(text="Request failed", foreground="#B42318")
            set_composer_enabled(True)

        run_in_background(work, on_success, on_error)

    def send_from_keyboard(event):
        if event.state & 0x0001:
            return None
        send_prompt()
        return "break"

    prompt_entry.bind("<Return>", send_from_keyboard)

    send_button = tk.Button(
        composer_content,
        text="Send ↑",
        command=send_prompt,
        background=colors["accent"],
        foreground="#FFFFFF",
        activebackground=colors["accent_hover"],
        activeforeground="#FFFFFF",
        relief="flat",
        borderwidth=0,
        padx=20,
        pady=12,
        cursor="hand2",
    )
    send_button.grid(row=1, column=1, sticky="se")

    tk.Label(
        composer_content,
        text="Enter to send  •  Shift+Enter for a new line",
        background=colors["surface"],
        foreground=colors["muted"],
        anchor="w",
        font=("TkDefaultFont", 9),
    ).grid(row=2, column=0, columnspan=2, sticky="ew", pady=(7, 0))

    def export_pdf():
        if not transcript:
            messagebox.showerror("PDF error", "No chat content to export.")
            return

        text = "\n\n".join(
            f"{speaker}\n{content}"
            for speaker, content in transcript
        )
        export_button.config(state="disabled")
        status_label.config(text="Exporting PDF…", foreground=colors["muted"])

        def work():
            return generate_pdf(text, report_dir / "slice_report.pdf")

        def on_success(pdf_path):
            if not widget_exists(chat_window):
                return
            export_button.config(state="normal")
            status_label.config(text="Ready", foreground=colors["accent"])
            messagebox.showinfo("PDF generated", f"PDF saved in:\n{pdf_path}")

        def on_error(error):
            if not widget_exists(chat_window):
                return
            export_button.config(state="normal")
            status_label.config(text="Export failed", foreground="#B42318")
            messagebox.showerror("PDF error", str(error))

        run_in_background(work, on_success, on_error)

    export_button = tk.Button(
        header_actions,
        text="Export PDF",
        command=export_pdf,
        background=colors["surface"],
        foreground=colors["text"],
        activebackground=colors["background"],
        relief="solid",
        borderwidth=1,
        padx=12,
        pady=6,
        cursor="hand2",
    )
    export_button.pack(side="right")

    prompt_entry.focus_set()


def open_large_image(image_path):
    image_path = Path(image_path)
    with Image.open(image_path) as source_image:
        original_image = source_image.copy()
        original_size = source_image.size

    image_window = tk.Toplevel(windows)
    image_window.title(image_path.name)
    image_window.configure(background=APP_COLORS["background"])

    target_monitor = monitor_geometry(image_window)
    monitor_width = target_monitor[2]
    monitor_height = target_monitor[3]
    max_width = max(int(monitor_width * 0.82) - 40, 1)
    max_height = max(int(monitor_height * 0.82) - 90, 1)
    scale = min(
        max_width / max(original_image.width, 1),
        max_height / max(original_image.height, 1),
        3.0,
    )
    display_size = (
        max(int(original_image.width * scale), 1),
        max(int(original_image.height * scale), 1),
    )

    configure_window(
        image_window,
        width=display_size[0] + 40,
        height=display_size[1] + 90,
        min_width=min(display_size[0] + 40, 420),
        min_height=min(display_size[1] + 90, 320),
        adaptive=False,
        target_monitor=target_monitor,
    )

    image_header = tk.Frame(
        image_window,
        background=APP_COLORS["surface"],
        highlightbackground=APP_COLORS["border"],
        highlightthickness=1,
    )
    image_header.pack(fill="x")
    tk.Label(
        image_header,
        text=(
            f"{image_path.name} — original size: "
            f"{original_size[0]}×{original_size[1]}"
        ),
        background=APP_COLORS["surface"],
        foreground=APP_COLORS["text"],
        font=("TkDefaultFont", 10, "bold"),
    ).pack(anchor="w", padx=16, pady=11)

    image_canvas = tk.Canvas(
        image_window,
        background="#111111",
        borderwidth=0,
        highlightthickness=0,
    )
    image_canvas.pack(fill="both", expand=True, padx=10, pady=(0, 10))
    canvas_image = image_canvas.create_image(0, 0, anchor="center")
    resize_job = None

    def render_image():
        nonlocal resize_job
        resize_job = None
        if not widget_exists(image_canvas):
            return

        canvas_width = max(image_canvas.winfo_width() - 20, 1)
        canvas_height = max(image_canvas.winfo_height() - 20, 1)
        current_scale = min(
            canvas_width / max(original_image.width, 1),
            canvas_height / max(original_image.height, 1),
            3.0,
        )
        current_size = (
            max(int(original_image.width * current_scale), 1),
            max(int(original_image.height * current_scale), 1),
        )
        if current_size == original_image.size:
            rendered_image = original_image
        else:
            rendered_image = original_image.resize(
                current_size,
                Image.Resampling.LANCZOS,
            )

        photo = ImageTk.PhotoImage(rendered_image)
        image_canvas.image = photo
        image_canvas.itemconfigure(canvas_image, image=photo)
        image_canvas.coords(
            canvas_image,
            image_canvas.winfo_width() // 2,
            image_canvas.winfo_height() // 2,
        )

    def schedule_image_render(event=None):
        nonlocal resize_job
        if resize_job is not None:
            image_window.after_cancel(resize_job)
        resize_job = image_window.after(60, render_image)

    image_canvas.bind("<Configure>", schedule_image_render)
    image_window.after_idle(render_image)


def clear_frame(frame):
    for widget in frame.winfo_children():
        widget.destroy()


def selected_dataset_folder():
    folder = path_folder.cget("text")

    if folder == "No selected folder":
        return None

    return Path(folder)


def show_html_report(frame):
    clear_frame(frame)
    frame.configure(background=APP_COLORS["surface"])

    report_header = tk.Frame(
        frame,
        background=APP_COLORS["surface"],
        highlightbackground=APP_COLORS["border"],
        highlightthickness=1,
    )
    report_header.pack(fill="x")
    tk.Label(
        report_header,
        text="Analysis report",
        background=APP_COLORS["surface"],
        foreground=APP_COLORS["text"],
        font=("TkDefaultFont", 12, "bold"),
    ).pack(side="left", padx=16, pady=10)
    tk.Label(
        report_header,
        text=str(html_report),
        background=APP_COLORS["surface"],
        foreground=APP_COLORS["muted"],
    ).pack(side="right", padx=16, pady=10)

    if HtmlFrame is None:
        tk.Label(
            frame,
            text="Install tkinterweb to display the HTML report.",
            background=APP_COLORS["surface"],
            foreground=APP_COLORS["muted"],
        ).pack(pady=30)
        return

    html_frame = HtmlFrame(
        frame,
        fontscale=1.0,
        horizontal_scrollbar="auto",
        textwrap=True,
    )
    html_frame.pack(fill="both", expand=True)
    html_frame.load_file(str(html_report), force=True)


def refresh_open_reports():
    active_report_tabs = []

    for report_tab in open_report_tabs:
        if not widget_exists(report_tab):
            continue
        show_html_report(report_tab)
        active_report_tabs.append(report_tab)

    open_report_tabs[:] = active_report_tabs


def run_project_module(module_name, arguments):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            module_name,
            *(str(argument) for argument in arguments),
        ],
        cwd=project_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        details = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(details or "The script failed without an error message.")


def update_dataset_report(selected_folder, selected_split):
    report_dir.mkdir(parents=True, exist_ok=True)
    run_project_module(
        dataset_analyzer_module,
        [
            selected_folder,
            "--split",
            selected_split,
            "--output",
            json_report,
        ]
    )
    run_project_module(
        report_module,
        [
            "--input",
            json_report,
            "--html",
            html_report,
        ]
    )


def display_current_report():
    refresh_open_reports()
    if not open_report_tabs:
        show_report()
        return

    report_window = open_report_tabs[0].winfo_toplevel()
    report_window.deiconify()
    report_window.lift()


def set_analysis_controls_enabled(enabled):
    button_state = "normal" if enabled else "disabled"
    split_state = "readonly" if enabled else "disabled"
    select_analyze_button.config(state=button_state)
    open_report_button.config(state=button_state)
    split_menu.config(state=split_state)


def open_html_report_window(report_path, title="HTML report"):
    report_window = tk.Toplevel(windows)
    report_window.title(title)
    report_window.configure(background=APP_COLORS["background"])
    configure_window(
        report_window,
        width=1300,
        height=850,
        min_width=900,
        min_height=650,
    )

    report_header = tk.Frame(
        report_window,
        background=APP_COLORS["surface"],
        highlightbackground=APP_COLORS["border"],
        highlightthickness=1,
    )
    report_header.pack(fill="x")
    tk.Label(
        report_header,
        text=title,
        background=APP_COLORS["surface"],
        foreground=APP_COLORS["text"],
        font=("TkDefaultFont", 12, "bold"),
    ).pack(side="left", padx=16, pady=10)
    tk.Label(
        report_header,
        text=str(report_path),
        background=APP_COLORS["surface"],
        foreground=APP_COLORS["muted"],
    ).pack(side="right", padx=16, pady=10)

    if HtmlFrame is None:
        tk.Label(
            report_window,
            text="Install tkinterweb to display the HTML report.",
            background=APP_COLORS["background"],
            foreground=APP_COLORS["muted"],
        ).pack(pady=30)
        return

    html_frame = HtmlFrame(
        report_window,
        fontscale=1.0,
        horizontal_scrollbar="auto",
        textwrap=True,
    )
    html_frame.pack(fill="both", expand=True)
    html_frame.load_file(str(report_path))


def choose_slice(frame, back_command=None):
    clear_frame(frame)
    frame.configure(background=APP_COLORS["background"])

    if isinstance(frame.master, ttk.Notebook):
        frame.master.tab(frame, text="Slices")

    if back_command is None:
        back_command = lambda: show_html_report(frame)

    add_page_heading(
        frame,
        "Browse medical slices",
        "Choose a NIfTI series to inspect.",
        back_command,
    )

    dataset_folder = selected_dataset_folder()
    if dataset_folder is None:
        empty_card = make_card(frame)
        empty_card.pack(fill="x", padx=80, pady=30)
        tk.Label(
            empty_card,
            text="No dataset selected",
            background=APP_COLORS["surface"],
            foreground=APP_COLORS["text"],
            font=("TkDefaultFont", 13, "bold"),
        ).pack(pady=(24, 5))
        tk.Label(
            empty_card,
            text="Import a dataset folder before browsing its series.",
            background=APP_COLORS["surface"],
            foreground=APP_COLORS["muted"],
        ).pack(pady=(0, 24))
        return

    nifti_files = sorted(
        list(dataset_folder.rglob("*.nii")) + list(dataset_folder.rglob("*.nii.gz"))
    )

    if not nifti_files:
        empty_card = make_card(frame)
        empty_card.pack(fill="x", padx=80, pady=30)
        tk.Label(
            empty_card,
            text="No NIfTI series found",
            background=APP_COLORS["surface"],
            foreground=APP_COLORS["text"],
            font=("TkDefaultFont", 13, "bold"),
        ).pack(pady=(24, 5))
        tk.Label(
            empty_card,
            text=str(dataset_folder),
            background=APP_COLORS["surface"],
            foreground=APP_COLORS["muted"],
            wraplength=650,
        ).pack(pady=(0, 24))
        return

    center_frame = make_card(frame)
    center_frame.pack(fill="x", padx=80, pady=24)

    tk.Label(
        center_frame,
        text=f"{len(nifti_files)} series available",
        background=APP_COLORS["surface"],
        foreground=APP_COLORS["text"],
        font=("TkDefaultFont", 13, "bold"),
    ).pack(anchor="w", padx=24, pady=(22, 5))
    tk.Label(
        center_frame,
        text="Select a series to choose its anatomical plane.",
        background=APP_COLORS["surface"],
        foreground=APP_COLORS["muted"],
    ).pack(anchor="w", padx=24)

    series_names = [str(path.relative_to(dataset_folder)) for path in nifti_files]
    series_by_name = dict(zip(series_names, nifti_files))

    series_menu = ttk.Combobox(
        center_frame,
        values=series_names,
        state="readonly",
        width=62,
        style="App.TCombobox",
    )
    series_menu.pack(fill="x", padx=24, pady=(16, 24))

    def select_series(event=None):
        selected_name = series_menu.get()
        if selected_name:
            choose_slice_type(frame, series_by_name[selected_name], back_command)

    series_menu.bind("<<ComboboxSelected>>", select_series)


def choose_slice_type(frame, nifti_path, back_command):
    clear_frame(frame)
    frame.configure(background=APP_COLORS["background"])

    if isinstance(frame.master, ttk.Notebook):
        frame.master.tab(frame, text=nifti_path.name)

    add_page_heading(
        frame,
        "Choose an anatomical plane",
        nifti_path.name,
        lambda: choose_slice(frame, back_command),
    )

    center_frame = make_card(frame)
    center_frame.pack(fill="x", padx=100, pady=30)

    tk.Label(
        center_frame,
        text="Slice orientation",
        background=APP_COLORS["surface"],
        foreground=APP_COLORS["text"],
        font=("TkDefaultFont", 14, "bold"),
    ).pack(anchor="w", padx=24, pady=(22, 5))
    tk.Label(
        center_frame,
        text="The volume will be converted into ordered PNG slices.",
        background=APP_COLORS["surface"],
        foreground=APP_COLORS["muted"],
    ).pack(anchor="w", padx=24, pady=(0, 14))

    for slice_type in ["sagittal slice", "coronal slice", "axial slice"]:
        make_button(
            center_frame,
            slice_type.title(),
            lambda selected_type=slice_type: display_slices(
                frame,
                nifti_path,
                selected_type,
                back_command,
            ),
            variant="secondary",
            anchor="w",
        ).pack(fill="x", padx=24, pady=6)

    tk.Frame(
        center_frame,
        height=12,
        background=APP_COLORS["surface"],
    ).pack()


def display_slices(frame, nifti_path, slice_type, back_command):
    global selected_slice_paths

    clear_frame(frame)
    frame.configure(background=APP_COLORS["background"])
    selected_slice_paths = set()

    add_page_heading(
        frame,
        "Preparing slices",
        f"{nifti_path.name} · {slice_type}",
        lambda: choose_slice_type(frame, nifti_path, back_command),
    )
    progress_card = make_card(frame)
    progress_card.pack(fill="x", padx=100, pady=40)
    tk.Label(
        progress_card,
        text="Converting the volume…",
        background=APP_COLORS["surface"],
        foreground=APP_COLORS["text"],
        font=("TkDefaultFont", 13, "bold"),
    ).pack(pady=(28, 6))
    tk.Label(
        progress_card,
        text="The gallery will appear as soon as the PNG slices are ready.",
        background=APP_COLORS["surface"],
        foreground=APP_COLORS["muted"],
    ).pack(pady=(0, 28))

    def work():
        return convert_to_png(slice_type, nifti_path, root_dir=project_dir)

    def on_success(result):
        global selected_slice_paths

        if not widget_exists(frame):
            return

        slices_dir, number_of_slices = result
        selected_slice_paths = set()

        clear_frame(frame)
        frame.configure(background=APP_COLORS["background"])

        page_title = f"{number_of_slices} {slice_type}s"
        add_page_heading(
            frame,
            page_title,
            nifti_path.name,
            lambda: choose_slice_type(frame, nifti_path, back_command),
        )

        controls_frame = make_card(frame)
        controls_frame.pack(fill="x", padx=24, pady=(0, 14))
        controls_content = tk.Frame(
            controls_frame,
            background=APP_COLORS["surface"],
        )
        controls_content.pack(fill="x", padx=14, pady=9)

        selection_label = tk.Label(
            controls_content,
            text="0 slices selected",
            background=APP_COLORS["surface"],
            foreground=APP_COLORS["muted"],
            font=("TkDefaultFont", 9, "bold"),
        )
        selection_label.pack(side="right", padx=8)

        image_files = sorted(slices_dir.glob("slice_*.png"))
        slice_frames_by_path = {}
        select_all_var = tk.BooleanVar(value=False)

        def update_selection_label():
            selection_label.config(
                text=f"{len(selected_slice_paths)} slices selected"
            )
            select_all_var.set(
                bool(image_files)
                and len(selected_slice_paths) == len(image_files)
            )

        def update_slice_frame_state(path, container):
            if path in selected_slice_paths:
                container.config(
                    background=DEFAULT_SLICE_BACKGROUND,
                    highlightbackground=APP_COLORS["accent"],
                    highlightthickness=3,
                )
            else:
                container.config(
                    background=DEFAULT_SLICE_BACKGROUND,
                    highlightbackground=APP_COLORS["border"],
                    highlightthickness=1,
                )

        def refresh_loaded_slices():
            for path, container in slice_frames_by_path.items():
                if widget_exists(container):
                    update_slice_frame_state(path, container)

        def toggle_all_slices():
            if select_all_var.get():
                selected_slice_paths.clear()
                selected_slice_paths.update(image_files)
            else:
                selected_slice_paths.clear()
            refresh_loaded_slices()
            update_selection_label()

        ttk.Checkbutton(
            controls_content,
            text="Select all slices",
            variable=select_all_var,
            command=toggle_all_slices,
            state="normal" if image_files else "disabled",
            style="App.TCheckbutton",
        ).pack(side="left", padx=5)

        if isinstance(frame.master, ttk.Notebook):
            frame.master.tab(frame, text=f"{nifti_path.stem} - {slice_type}")

        gallery = tk.Frame(frame, background=APP_COLORS["background"])
        gallery.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        canvas = tk.Canvas(
            gallery,
            background=APP_COLORS["background"],
            borderwidth=0,
            highlightthickness=0,
        )
        scrollbar = ttk.Scrollbar(
            gallery,
            orient="vertical",
            command=canvas.yview,
            style="App.Vertical.TScrollbar",
        )
        images_frame = tk.Frame(canvas, background=APP_COLORS["background"])
        grid_column_count = 1

        images_frame.bind(
            "<Configure>",
            lambda event: canvas.configure(scrollregion=canvas.bbox("all")),
        )

        canvas_window = canvas.create_window(
            (0, 0),
            window=images_frame,
            anchor="nw",
        )
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        images_frame.images = []

        def layout_slice_frames(available_width=None):
            nonlocal grid_column_count

            if available_width is None:
                available_width = canvas.winfo_width()

            new_column_count = max(
                1,
                int(available_width) // SLICE_CARD_MIN_WIDTH,
            )

            for column in range(max(grid_column_count, new_column_count)):
                images_frame.grid_columnconfigure(column, weight=0)
            for column in range(new_column_count):
                images_frame.grid_columnconfigure(column, weight=1)

            grid_column_count = new_column_count
            for index, container in enumerate(slice_frames_by_path.values()):
                container.grid(
                    row=index // grid_column_count,
                    column=index % grid_column_count,
                    padx=8,
                    pady=8,
                    sticky="n",
                )

        def resize_images_frame(event):
            canvas.itemconfigure(canvas_window, width=event.width)
            layout_slice_frames(event.width)

        canvas.bind("<Configure>", resize_images_frame)

        def add_images(start=0, batch_size=12):
            if not widget_exists(images_frame):
                return

            end = min(start + batch_size, len(image_files))

            for index in range(start, end):
                image_path = image_files[index]
                with Image.open(image_path) as source_image:
                    image = source_image.copy()
                image.thumbnail(THUMBNAIL_SIZE, Image.Resampling.LANCZOS)

                photo = ImageTk.PhotoImage(image)
                images_frame.images.append(photo)

                slice_frame = tk.Frame(
                    images_frame,
                    background=DEFAULT_SLICE_BACKGROUND,
                    highlightbackground=APP_COLORS["border"],
                    highlightthickness=1,
                    cursor="hand2",
                )
                slice_frames_by_path[image_path] = slice_frame
                update_slice_frame_state(image_path, slice_frame)

                image_label = tk.Label(
                    slice_frame,
                    image=photo,
                    background="#111111",
                    cursor="hand2",
                )
                image_label.pack(padx=8, pady=(8, 0))

                name_label = tk.Label(
                    slice_frame,
                    text=image_path.stem,
                    wraplength=THUMBNAIL_SIZE[0],
                    background=APP_COLORS["surface"],
                    foreground=APP_COLORS["muted"],
                    cursor="hand2",
                )
                name_label.pack(fill="x", padx=8, pady=8)

                def toggle_slice(event=None, path=image_path, container=slice_frame):
                    if path in selected_slice_paths:
                        selected_slice_paths.remove(path)
                    else:
                        selected_slice_paths.add(path)

                    update_slice_frame_state(path, container)
                    update_selection_label()

                def show_slice_context_menu(event, path=image_path):
                    context_menu = tk.Menu(
                        frame,
                        tearoff=0,
                        background=APP_COLORS["surface"],
                        foreground=APP_COLORS["text"],
                        activebackground=APP_COLORS["accent_soft"],
                        activeforeground=APP_COLORS["text"],
                        relief="solid",
                        borderwidth=1,
                    )
                    context_menu.add_command(
                        label="View larger",
                        command=lambda: open_large_image(path),
                    )
                    context_menu.add_command(
                        label="Send to chat",
                        command=lambda: open_chat_window([path]),
                    )
                    try:
                        context_menu.tk_popup(event.x_root, event.y_root)
                    finally:
                        context_menu.grab_release()

                slice_frame.bind("<Button-1>", toggle_slice)
                image_label.bind("<Button-1>", toggle_slice)
                name_label.bind("<Button-1>", toggle_slice)
                slice_frame.bind("<Button-3>", show_slice_context_menu)
                image_label.bind("<Button-3>", show_slice_context_menu)
                name_label.bind("<Button-3>", show_slice_context_menu)

            layout_slice_frames()

            if end < len(image_files):
                frame.after(1, lambda: add_images(end, batch_size))

        add_images()

    def on_error(error):
        if widget_exists(frame):
            messagebox.showerror("Slices error", str(error))

    run_in_background(work, on_success, on_error)


def open_slices_tab(notebook, report_tab):
    for tab_id in notebook.tabs():
        slice_tab = notebook.nametowidget(tab_id)
        if getattr(slice_tab, "is_slices_tab", False):
            notebook.select(slice_tab)
            choose_slice(slice_tab, back_command=lambda: notebook.select(report_tab))
            return

    slice_tab = tk.Frame(notebook, background=APP_COLORS["background"])
    slice_tab.is_slices_tab = True
    notebook.add(slice_tab, text="Slices")
    notebook.select(slice_tab)
    choose_slice(slice_tab, back_command=lambda: notebook.select(report_tab))


def show_report():
    report_window = tk.Toplevel(windows)
    report_window.title("Dataset workspace")
    report_window.configure(background=APP_COLORS["background"])
    configure_window(
        report_window,
        width=1300,
        height=850,
        min_width=900,
        min_height=650,
    )

    # Keep the workspace navigation in proportion with the report window.  A
    # fixed width made the labels clip on scaled displays and took too much
    # room away from the report when the window was made narrower.
    left_frame = make_card(report_window, width=245)
    left_frame.pack(side="left", fill="y", padx=(16, 8), pady=16)
    left_frame.pack_propagate(False)

    wrapped_sidebar_widgets = []

    sidebar_header = tk.Frame(left_frame, background=APP_COLORS["surface"])
    sidebar_header.pack(fill="x", padx=18, pady=(20, 22))
    workspace_label = tk.Label(
        sidebar_header,
        text="Workspace",
        background=APP_COLORS["surface"],
        foreground=APP_COLORS["text"],
        font=("TkDefaultFont", 16, "bold"),
        justify="left",
    )
    workspace_label.pack(anchor="w")
    workspace_subtitle = tk.Label(
        sidebar_header,
        text="Report and slice tools",
        background=APP_COLORS["surface"],
        foreground=APP_COLORS["muted"],
        justify="left",
    )
    workspace_subtitle.pack(anchor="w", pady=(3, 0))
    wrapped_sidebar_widgets.extend((workspace_label, workspace_subtitle))

    right_frame = tk.Frame(
        report_window,
        background=APP_COLORS["background"],
    )
    right_frame.pack(
        side="right",
        fill="both",
        expand=True,
        padx=(8, 16),
        pady=16,
    )

    notebook = ttk.Notebook(right_frame, style="App.TNotebook")
    notebook.pack(fill="both", expand=True)

    report_tab = tk.Frame(notebook, background=APP_COLORS["surface"])
    notebook.add(report_tab, text="Dataset analysis")
    open_report_tabs.append(report_tab)
    show_html_report(report_tab)

    dataset_analysis_frame = tk.Frame(
        left_frame,
        background=APP_COLORS["surface"],
    )
    dataset_analysis_frame.pack(fill="x", padx=14, pady=(0, 22))

    tk.Label(
        dataset_analysis_frame,
        text="DATASET",
        background=APP_COLORS["surface"],
        foreground=APP_COLORS["muted"],
        font=("TkDefaultFont", 9, "bold"),
    ).pack(anchor="w", padx=4, pady=(0, 7))
    analysis_report_button = make_button(
        dataset_analysis_frame,
        "Analysis report",
        lambda: notebook.select(report_tab),
        variant="secondary",
        anchor="w",
        justify="left",
    )
    analysis_report_button.pack(fill="x")
    wrapped_sidebar_widgets.append(analysis_report_button)

    qwen_analysis_frame = tk.Frame(
        left_frame,
        background=APP_COLORS["surface"],
    )
    qwen_analysis_frame.pack(fill="x", padx=14)

    tk.Label(
        qwen_analysis_frame,
        text="SLICE ANALYSIS",
        background=APP_COLORS["surface"],
        foreground=APP_COLORS["muted"],
        font=("TkDefaultFont", 9, "bold"),
    ).pack(anchor="w", padx=4, pady=(0, 7))
    browse_slices_button = make_button(
        qwen_analysis_frame,
        "Browse slices",
        lambda: open_slices_tab(notebook, report_tab),
        variant="secondary",
        anchor="w",
        justify="left",
    )
    browse_slices_button.pack(fill="x", pady=(0, 8))
    chat_qwen_button = make_button(
        qwen_analysis_frame,
        "Chat with Qwen",
        open_chat_window,
        anchor="w",
        justify="left",
    )
    chat_qwen_button.pack(fill="x")
    wrapped_sidebar_widgets.extend((browse_slices_button, chat_qwen_button))

    selected_folder = selected_dataset_folder()
    if selected_folder is not None:
        selected_folder_label = tk.Label(
            left_frame,
            text=selected_folder.name,
            background=APP_COLORS["surface_muted"],
            foreground=APP_COLORS["muted"],
            justify="left",
            padx=12,
            pady=9,
        )
        selected_folder_label.pack(side="bottom", fill="x", padx=14, pady=14)
        wrapped_sidebar_widgets.append(selected_folder_label)

    def resize_workspace_sidebar(event=None):
        if event is not None and event.widget is not report_window:
            return

        window_width = (
            event.width if event is not None else report_window.winfo_width()
        )
        content_width = max(window_width - 48, 1)
        preferred_width = round(content_width * 0.22)

        # Preserve a useful report area on small screens while allowing the
        # navigation to grow enough for larger fonts and long folder names.
        maximum_for_report = max(180, content_width - 560)
        sidebar_width = min(
            max(preferred_width, 210),
            340,
            maximum_for_report,
        )
        sidebar_width = max(sidebar_width, 180)
        left_frame.configure(width=sidebar_width)

        wraplength = max(sidebar_width - 64, 100)
        for widget in wrapped_sidebar_widgets:
            widget.configure(wraplength=wraplength)

    report_window.bind("<Configure>", resize_workspace_sidebar, add="+")
    report_window.after_idle(resize_workspace_sidebar)


def open_generated_report():
    if not html_report.exists():
        messagebox.showerror(
            "Missing report",
            "Generate data/dataset_analysis/report.html before opening the report.",
        )
        return

    status.config(text="Opening generated report...")
    status.config(foreground=APP_COLORS["muted"])
    display_current_report()
    status.config(text="Generated report opened")
    status.config(foreground=APP_COLORS["accent"])

#import of the dataset folder and report
def import_folder():
    folder = filedialog.askdirectory(title="Choose a folder")
    if folder:
        selected_folder = Path(folder).expanduser().resolve()
        selected_split = split_menu.get()
        path_folder.config(text=str(selected_folder))
        status.config(
            text=f"Analysis in progress ({selected_split} split)..."
        )
        status.config(foreground=APP_COLORS["muted"])
        set_analysis_controls_enabled(False)

        def work():
            update_dataset_report(selected_folder, selected_split)

        def on_success(result):
            status.config(
                text=f"Analysis finished ({selected_split} split)"
            )
            status.config(foreground=APP_COLORS["accent"])
            set_analysis_controls_enabled(True)
            refresh_open_reports()

        def on_error(error):
            status.config(text="Analysis error")
            status.config(foreground=APP_COLORS["error"])
            set_analysis_controls_enabled(True)
            messagebox.showerror("Analysis error", str(error))

        run_in_background(work, on_success, on_error)


main_frame = tk.Frame(windows, background=APP_COLORS["background"])
main_frame.pack(fill="both", expand=True, padx=46, pady=34)

app_header = tk.Frame(main_frame, background=APP_COLORS["background"])
app_header.pack(fill="x", pady=(0, 24))

tk.Label(
    app_header,
    text="MEDICAL IMAGING",
    background=APP_COLORS["accent_soft"],
    foreground=APP_COLORS["accent_hover"],
    font=("TkDefaultFont", 9, "bold"),
    padx=10,
    pady=5,
).pack(anchor="w")
tk.Label(
    app_header,
    text="Dataset workspace",
    background=APP_COLORS["background"],
    foreground=APP_COLORS["text"],
    font=("TkDefaultFont", 24, "bold"),
).pack(anchor="w", pady=(10, 4))
tk.Label(
    app_header,
    text=(
        "Inspect a NIfTI or DICOM dataset, review its quality report, "
        "and discuss selected slices with Qwen."
    ),
    background=APP_COLORS["background"],
    foreground=APP_COLORS["muted"],
    wraplength=700,
    justify="left",
).pack(anchor="w")

dataset_analysis_frame = make_card(main_frame)
dataset_analysis_frame.pack(fill="x")

card_content = tk.Frame(
    dataset_analysis_frame,
    background=APP_COLORS["surface"],
)
card_content.pack(fill="x", padx=24, pady=22)

tk.Label(
    card_content,
    text="Analyze a dataset",
    background=APP_COLORS["surface"],
    foreground=APP_COLORS["text"],
    font=("TkDefaultFont", 15, "bold"),
).pack(anchor="w")
tk.Label(
    card_content,
    text="Choose the split, then select the dataset folder to inspect.",
    background=APP_COLORS["surface"],
    foreground=APP_COLORS["muted"],
).pack(anchor="w", pady=(4, 16))

split_frame = tk.Frame(card_content, background=APP_COLORS["surface"])
split_frame.pack(fill="x", pady=(0, 14))

tk.Label(
    split_frame,
    text="Split",
    background=APP_COLORS["surface"],
    foreground=APP_COLORS["text"],
    font=("TkDefaultFont", 10, "bold"),
).pack(side="left", padx=(0, 10))

split_menu = ttk.Combobox(
    split_frame,
    values=("train", "test", "validation", "unknown", "all"),
    state="readonly",
    width=12,
    style="App.TCombobox",
)
split_menu.set("train")
split_menu.pack(side="left")

buttons_frame = tk.Frame(card_content, background=APP_COLORS["surface"])
buttons_frame.pack(fill="x", pady=(0, 16))

select_analyze_button = make_button(
    buttons_frame,
    "Select and analyze folder",
    import_folder,
)
select_analyze_button.pack(side="left", padx=(0, 10))

open_report_button = make_button(
    buttons_frame,
    "Open generated report",
    open_generated_report,
    variant="secondary",
)
open_report_button.pack(side="left")

selection_panel = tk.Frame(
    card_content,
    background=APP_COLORS["surface_muted"],
    padx=14,
    pady=11,
)
selection_panel.pack(fill="x")

tk.Label(
    selection_panel,
    text="Selected folder",
    background=APP_COLORS["surface_muted"],
    foreground=APP_COLORS["muted"],
    font=("TkDefaultFont", 9, "bold"),
).pack(anchor="w")

path_folder = tk.Label(
    selection_panel,
    text="No selected folder",
    background=APP_COLORS["surface_muted"],
    foreground=APP_COLORS["text"],
    wraplength=680,
    justify="left",
)
path_folder.pack(anchor="w", pady=(3, 0))

status = tk.Label(
    card_content,
    text="Ready",
    background=APP_COLORS["surface"],
    foreground=APP_COLORS["muted"],
    anchor="w",
    font=("TkDefaultFont", 9, "bold"),
)
status.pack(fill="x", pady=(12, 0))

tk.Label(
    main_frame,
    text="Analysis and report generation run in the background.",
    background=APP_COLORS["background"],
    foreground=APP_COLORS["muted"],
    font=("TkDefaultFont", 9),
).pack(anchor="w", pady=(14, 0))

windows.mainloop()
