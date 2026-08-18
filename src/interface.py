import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from tkinter.scrolledtext import ScrolledText
from tkinter import ttk
from pathlib import Path
import queue
import threading
from dataset_analyzer import analyze_dataset, save_dataset_analysis
from nifti_analyzer import DatasetNiftiPreparationError
from report import generate_reports
from slices_analyse import chat_qwen, convert_to_png, generate_pdf
from PIL import Image, ImageTk


try:
    from tkinterweb import HtmlFrame
except ImportError:
    HtmlFrame = None


windows = tk.Tk()
windows.title("Dataset analyse")
windows.geometry("600x300")

project_dir = Path(__file__).resolve().parents[1]
report_dir = project_dir / "data"
json_report = report_dir / "analyse_dataset.json"
html_report = report_dir / "report.html"
THUMBNAIL_SIZE = (120, 120)
SELECTED_SLICE_BACKGROUND = "#309AF0"
DEFAULT_SLICE_BACKGROUND = windows.cget("background")
selected_slice_paths = set()


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
def open_chat_window():
    chat_window = tk.Toplevel(windows)
    chat_window.title("Chat")
    chat_window.geometry("600x500")

    messages = []

    chat_area = ScrolledText(chat_window, wrap="word")
    chat_area.pack(fill="both", expand=True, padx=10, pady=10)

    prompt_entry = tk.Entry(chat_window)
    prompt_entry.pack(fill="x", padx=10, pady=5)

    def send_prompt():
        nonlocal messages

        prompt = prompt_entry.get()
        if not prompt:
            return

        image_paths = sorted(selected_slice_paths)

        prompt_entry.delete(0, tk.END)
        chat_area.insert(tk.END, f"You > {prompt}\n")
        chat_area.insert(tk.END, f"Qwen > Waiting... ({len(image_paths)} selected slices)\n")
        prompt_entry.config(state="disabled")
        send_button.config(state="disabled")

        def work():
            return chat_qwen(
                prompt,
                root_dir=project_dir,
                messages=messages,
                image_paths=image_paths,
               
            )

        def on_success(result):
            nonlocal messages
            if not widget_exists(chat_window):
                return

            answer, messages = result
            chat_area.delete("end-2l", "end-1l")
            chat_area.insert(tk.END, f"Qwen > {answer}\n\n")

            prompt_entry.config(state="normal")
            send_button.config(state="normal")

        def on_error(error):
            if not widget_exists(chat_window):
                return

            chat_area.delete("end-2l", "end-1l")
            prompt_entry.config(state="normal")
            send_button.config(state="normal")
            messagebox.showerror("Chat error", str(error))

        run_in_background(work, on_success, on_error)

    def export_pdf():
        text = chat_area.get("1.0", tk.END).strip()

        if not text:
            messagebox.showerror("PDF error", "No chat content to export.")
            return

        def work():
            return generate_pdf(text, report_dir / "slice_report.pdf")

        def on_success(pdf_path):
            messagebox.showinfo("PDF generated", f"PDF saved in:\n{pdf_path}")

        def on_error(error):
            messagebox.showerror("PDF error", str(error))

        run_in_background(work, on_success, on_error)

    buttons_frame = tk.Frame(chat_window)
    buttons_frame.pack(pady=10)

    send_button = tk.Button(chat_window, text="Send", command=send_prompt)
    send_button.pack(in_=buttons_frame, side="left", padx=5)

    tk.Button(buttons_frame, text="Generate PDF", command=export_pdf).pack(side="left", padx=5)

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

    tk.Label(frame, text=f"HTML report: {html_report}").pack(pady=5)

    if HtmlFrame is None:
        tk.Label(frame, text="Install tkinterweb to display the HTML report.").pack(pady=10)
        return

    html_frame = HtmlFrame(frame)
    html_frame.pack(fill="both", expand=True)
    html_frame.load_file(str(html_report))


def open_html_report_window(report_path, title="HTML report"):
    report_window = tk.Toplevel(windows)
    report_window.title(title)
    report_window.geometry("800x600")

    tk.Label(report_window, text=f"HTML report: {report_path}").pack(pady=5)

    if HtmlFrame is None:
        tk.Label(report_window, text="Install tkinterweb to display the HTML report.").pack(pady=10)
        return

    html_frame = HtmlFrame(report_window)
    html_frame.pack(fill="both", expand=True)
    html_frame.load_file(str(report_path))


def choose_slice(frame, back_command=None):
    clear_frame(frame)

    if isinstance(frame.master, ttk.Notebook):
        frame.master.tab(frame, text="Slices")

    if back_command is None:
        back_command = lambda: show_html_report(frame)

    tk.Button(frame, text="Back", command=back_command).pack(anchor="w")

    dataset_folder = selected_dataset_folder()
    if dataset_folder is None:
        tk.Label(frame, text="Import a dataset folder first.").pack(pady=10)
        return

    nifti_files = sorted(
        list(dataset_folder.rglob("*.nii")) + list(dataset_folder.rglob("*.nii.gz"))
    )

    if not nifti_files:
        tk.Label(frame, text="No NIfTI series found in this dataset.").pack(pady=10)
        return

    center_frame = tk.Frame(frame)
    center_frame.pack(expand=True)

    tk.Label(center_frame, text="Choose a series").pack(pady=10)

    series_names = [str(path.relative_to(dataset_folder)) for path in nifti_files]
    series_by_name = dict(zip(series_names, nifti_files))

    series_menu = ttk.Combobox(
        center_frame,
        values=series_names,
        state="readonly",
        width=50,
    )
    series_menu.pack(pady=10)

    def select_series(event=None):
        selected_name = series_menu.get()
        if selected_name:
            choose_slice_type(frame, series_by_name[selected_name], back_command)

    series_menu.bind("<<ComboboxSelected>>", select_series)


def choose_slice_type(frame, nifti_path, back_command):
    clear_frame(frame)

    if isinstance(frame.master, ttk.Notebook):
        frame.master.tab(frame, text=nifti_path.name)

    tk.Button(frame, text="Back", command=lambda: choose_slice(frame, back_command)).pack(anchor="w")

    center_frame = tk.Frame(frame)
    center_frame.pack(expand=True)

    tk.Label(center_frame, text=nifti_path.name).pack(pady=10)

    for slice_type in ["sagittal slice", "coronal slice", "axial slice"]:
        tk.Button(
            center_frame,
            text=slice_type,
            command=lambda selected_type=slice_type: display_slices(frame, nifti_path, selected_type, back_command),
        ).pack(pady=8)


def display_slices(frame, nifti_path, slice_type, back_command):
    global selected_slice_paths

    clear_frame(frame)
    selected_slice_paths = set()

    tk.Button(
        frame,
        text="Back",
        command=lambda: choose_slice_type(frame, nifti_path, back_command),
    ).pack(anchor="w")

    tk.Label(frame, text="Generating slices...").pack(pady=10)

    def work():
        return convert_to_png(slice_type, nifti_path, root_dir=project_dir)

    def on_success(result):
        global selected_slice_paths

        if not widget_exists(frame):
            return

        slices_dir, number_of_slices = result
        selected_slice_paths = set()

        clear_frame(frame)

        tk.Button(
            frame,
            text="Back",
            command=lambda: choose_slice_type(frame, nifti_path, back_command),
        ).pack(anchor="w")
        page_title = f"{nifti_path.name} - {number_of_slices} {slice_type}s"
        tk.Label(frame, text=page_title).pack(pady=5)
        selection_label = tk.Label(frame, text="Selected slices: 0")
        selection_label.pack(pady=3)

        image_files = sorted(slices_dir.glob("slice_*.png"))
        slice_frames_by_path = {}

        def update_selection_label():
            selection_label.config(text=f"Selected slices: {len(selected_slice_paths)}")

        def update_slice_frame_state(path, container):
            if path in selected_slice_paths:
                container.config(relief="solid", background=SELECTED_SLICE_BACKGROUND)
            else:
                container.config(relief="flat", background=DEFAULT_SLICE_BACKGROUND)

        def refresh_loaded_slices():
            for path, container in slice_frames_by_path.items():
                if widget_exists(container):
                    update_slice_frame_state(path, container)

        def select_all_slices():
            selected_slice_paths.clear()
            selected_slice_paths.update(image_files)
            refresh_loaded_slices()
            update_selection_label()

        controls_frame = tk.Frame(frame)
        controls_frame.pack(pady=3)

        tk.Button(
            controls_frame,
            text="Select all slices",
            command=select_all_slices,
            state="normal" if image_files else "disabled",
        ).pack(side="left", padx=5)

        if isinstance(frame.master, ttk.Notebook):
            frame.master.tab(frame, text=f"{nifti_path.stem} - {slice_type}")

        canvas = tk.Canvas(frame)
        scrollbar = tk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        images_frame = tk.Frame(canvas)

        images_frame.bind(
            "<Configure>",
            lambda event: canvas.configure(scrollregion=canvas.bbox("all")),
        )

        canvas.create_window((0, 0), window=images_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        images_frame.images = []

        def add_images(start=0, batch_size=12):
            if not widget_exists(images_frame):
                return

            end = min(start + batch_size, len(image_files))

            for index in range(start, end):
                image_path = image_files[index]
                image = Image.open(image_path)
                image.thumbnail(THUMBNAIL_SIZE)

                photo = ImageTk.PhotoImage(image)
                images_frame.images.append(photo)

                row = index // 3
                column = index % 3

                slice_frame = tk.Frame(images_frame, borderwidth=2, relief="flat")
                slice_frames_by_path[image_path] = slice_frame
                update_slice_frame_state(image_path, slice_frame)
                slice_frame.grid(row=row, column=column, padx=5, pady=5)

                image_label = tk.Label(slice_frame, image=photo)
                image_label.pack()

                name_label = tk.Label(slice_frame, text=image_path.stem)
                name_label.pack()

                def toggle_slice(event=None, path=image_path, container=slice_frame):
                    if path in selected_slice_paths:
                        selected_slice_paths.remove(path)
                    else:
                        selected_slice_paths.add(path)

                    update_slice_frame_state(path, container)
                    update_selection_label()

                slice_frame.bind("<Button-1>", toggle_slice)
                image_label.bind("<Button-1>", toggle_slice)
                name_label.bind("<Button-1>", toggle_slice)

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

    slice_tab = tk.Frame(notebook)
    slice_tab.is_slices_tab = True
    notebook.add(slice_tab, text="Slices")
    notebook.select(slice_tab)
    choose_slice(slice_tab, back_command=lambda: notebook.select(report_tab))


def show_report():

    report_window = tk.Toplevel(windows)
    report_window.title("Dataset report")
    report_window.geometry("800x600")

    left_frame = tk.Frame(report_window, width=150)
    left_frame.pack(side="left", fill="y", padx=10, pady=10)
    left_frame.pack_propagate(False)

    right_frame = tk.Frame(report_window)
    right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

    notebook = ttk.Notebook(right_frame)
    notebook.pack(fill="both", expand=True)

    report_tab = tk.Frame(notebook)
    notebook.add(report_tab, text="Report")
    show_html_report(report_tab)

    tk.Button(
        left_frame,
        text="Slices display",
        command=lambda: open_slices_tab(notebook, report_tab),
    ).pack(fill="x", pady=5)
    tk.Button(left_frame, text="Chat", command=open_chat_window).pack(fill="x", pady=5)


def generate_report():
    if not json_report.exists():
        messagebox.showerror("Missing analysis", "Import a dataset folder first.")
        return

    status.config(text="Generating report...")

    def work():
        generate_reports(json_report, html_report)

    def on_success(result):
        status.config(text="Report generated")
        show_report()

    def on_error(error):
        status.config(text="Report error")
        messagebox.showerror("Report error", str(error))

    run_in_background(work, on_success, on_error)

#import of the dataset folder and report
def import_folder():
    folder = filedialog.askdirectory(title="Choose a folder")
    if folder:
        path_folder.config(text=folder)
        status.config(text="Analysis in progress...")

        def work():
            report_dir.mkdir(exist_ok=True)

            analysis = analyze_dataset(folder, split=None)
            save_dataset_analysis(analysis, json_report)

        def on_success(result):
            status.config(text="Analysis finished")

        def on_error(error):
            if isinstance(error, DatasetNiftiPreparationError):
                status.config(text="DICOM conversion error")
                messagebox.showerror("DICOM conversion error", str(error))
            else:
                status.config(text="Analysis error")
                messagebox.showerror("Analysis error", str(error))

        run_in_background(work, on_success, on_error)


tk.Label(windows, text="DICOM or NIfTI file's datasets:").pack(pady=10)
tk.Button(windows, text="Import a folder", command=import_folder, 
          bg="#309AF0", fg="white").pack(pady=10)

tk.Button(windows, text="Generate a report", command=generate_report,
          bg="#309AF0", fg="white").pack(pady=10)

path_folder = tk.Label(windows, text="No selected folder")
path_folder.pack(pady=10)

status = tk.Label(windows, text="")
status.pack(pady=10)



windows.mainloop()
