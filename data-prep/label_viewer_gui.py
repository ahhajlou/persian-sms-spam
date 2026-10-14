import json
import re
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from spam_common.data import LABELED_FILE, write_jsonl
from spam_common.model_label import SCORE_FIELD, UNCERTAIN

SCORE_FILTERS = ["All", f"Uncertain ({UNCERTAIN[0]}-{UNCERTAIN[1]})", "Labelled by model", "From dataset"]
SORT_ORDERS = ["File order", "Most uncertain first"]

# Optional but highly recommended for proper Persian/Arabic RTL display
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    RTL_SUPPORT = True
except ImportError:
    RTL_SUPPORT = False

class SMSLabelerApp:
    def __init__(self, root, path=str(LABELED_FILE)):
        self.root = root
        self.default_path = path
        self.root.title("SMS Labeler GUI")
        self.root.geometry("900x700")
        
        # Default font that supports Persian/English well
        self.default_font = ("Tahoma", 11)
        self.root.option_add("*Font", self.default_font)
        
        self.data = []
        self.filtered_indices = []
        self.filepath = ""
        self.current_index = 0  # Track current position in filtered list

        self.setup_ui()

    def setup_ui(self):
        # --- Top Frame: File & Filters ---
        top_frame = ttk.LabelFrame(self.root, text="Controls & Filters", padding=10)
        top_frame.pack(fill="x", padx=10, pady=5)

        # File selection
        file_frame = ttk.Frame(top_frame)
        file_frame.pack(fill="x", pady=5)
        ttk.Label(file_frame, text="File:").pack(side="left")
        self.file_var = tk.StringVar(value=self.default_path)
        ttk.Entry(file_frame, textvariable=self.file_var, width=50).pack(side="left", padx=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_file).pack(side="left")
        ttk.Button(file_frame, text="Load", command=self.load_data).pack(side="left", padx=5)

        # Filters
        filter_frame = ttk.Frame(top_frame)
        filter_frame.pack(fill="x", pady=5)

        ttk.Label(filter_frame, text="Label:").pack(side="left", padx=(0, 5))
        self.label_var = tk.StringVar(value="All")
        label_combo = ttk.Combobox(filter_frame, textvariable=self.label_var, values=["All", "spam", "ham", "null"], state="readonly", width=10)
        label_combo.pack(side="left", padx=5)

        ttk.Label(filter_frame, text="Regex:").pack(side="left", padx=(15, 5))
        self.regex_var = tk.StringVar()
        ttk.Entry(filter_frame, textvariable=self.regex_var, width=25).pack(side="left", padx=5)

        ttk.Label(filter_frame, text="Field:").pack(side="left", padx=(0, 5))
        self.regex_field_var = tk.StringVar(value="text")
        field_combo = ttk.Combobox(filter_frame, textvariable=self.regex_field_var, values=["text", "address", "both"], state="readonly", width=10)
        field_combo.pack(side="left", padx=5)

        ttk.Button(filter_frame, text="Apply Filter", command=self.apply_filters).pack(side="left", padx=10)

        # Model score (written by `make model-label`): review uncertain messages first.
        score_frame = ttk.Frame(top_frame)
        score_frame.pack(fill="x", pady=5)
        ttk.Label(score_frame, text="Model score:").pack(side="left", padx=(0, 5))
        self.score_var = tk.StringVar(value=SCORE_FILTERS[0])
        ttk.Combobox(score_frame, textvariable=self.score_var, values=SCORE_FILTERS, state="readonly", width=22).pack(side="left", padx=5)
        ttk.Label(score_frame, text="Sort:").pack(side="left", padx=(15, 5))
        self.sort_var = tk.StringVar(value=SORT_ORDERS[0])
        ttk.Combobox(score_frame, textvariable=self.sort_var, values=SORT_ORDERS, state="readonly", width=20).pack(side="left", padx=5)

        # --- Middle Frame: Treeview ---
        middle_frame = ttk.Frame(self.root)
        middle_frame.pack(fill="both", expand=True, padx=10, pady=5)

        tree_scroll = ttk.Scrollbar(middle_frame)
        tree_scroll.pack(side="right", fill="y")

        self.tree = ttk.Treeview(middle_frame, columns=("ID", "Label", "Score", "Address", "Text"), show="headings", yscrollcommand=tree_scroll.set)
        self.tree.heading("ID", text="ID")
        self.tree.heading("Label", text="Label")
        self.tree.heading("Score", text="Spam score")
        self.tree.heading("Address", text="Address")
        self.tree.heading("Text", text="Text Preview")
        
        self.tree.column("ID", width=50, anchor="center")
        self.tree.column("Label", width=80, anchor="center")
        self.tree.column("Score", width=80, anchor="center")
        self.tree.column("Address", width=150)
        self.tree.column("Text", width=500)
        
        self.tree.pack(side="left", fill="both", expand=True)
        tree_scroll.config(command=self.tree.yview)

        self.tree.bind("<<TreeviewSelect>>", self.on_select)

        # --- Bottom Frame: Details & Editing ---
        bottom_frame = ttk.LabelFrame(self.root, text="Entry Details & Editing", padding=10)
        bottom_frame.pack(fill="x", padx=10, pady=5)

        detail_top = ttk.Frame(bottom_frame)
        detail_top.pack(fill="x", pady=5)

        ttk.Label(detail_top, text="Index:").pack(side="left")
        self.detail_id_var = tk.StringVar(value="-")
        ttk.Label(detail_top, textvariable=self.detail_id_var, foreground="blue").pack(side="left", padx=5)

        # Position indicator
        self.position_var = tk.StringVar(value="")
        ttk.Label(detail_top, textvariable=self.position_var, foreground="gray").pack(side="left", padx=10)

        ttk.Label(detail_top, text="New Label:").pack(side="left", padx=(20, 5))
        self.detail_label_var = tk.StringVar(value="ham")
        detail_label_combo = ttk.Combobox(detail_top, textvariable=self.detail_label_var, values=["spam", "ham"], state="readonly", width=10)
        detail_label_combo.pack(side="left", padx=5)

        ttk.Button(detail_top, text="Update Label", command=self.update_label).pack(side="left", padx=10)
        
        # Navigation buttons
        nav_frame = ttk.Frame(detail_top)
        nav_frame.pack(side="right")
        ttk.Button(nav_frame, text="◄ Previous", command=self.go_previous).pack(side="left", padx=2)
        ttk.Button(nav_frame, text="Next ►", command=self.go_next).pack(side="left", padx=2)

        ttk.Button(detail_top, text="💾 Save to File", command=self.save_data, style="Accent.TButton").pack(side="right", padx=10)

        # Text display area
        self.detail_text = tk.Text(bottom_frame, height=6, wrap="word", font=("Tahoma", 12))
        self.detail_text.pack(fill="x", pady=5)
        # Configure RTL alignment for Persian text
        self.detail_text.tag_configure("rtl", justify="right")
        self.detail_text.config(state="disabled")

        if not RTL_SUPPORT:
            self.detail_text.config(state="normal")
            self.detail_text.insert("end", "️ Note: For proper Persian (RTL) display, please run: pip install arabic-reshaper python-bidi")
            self.detail_text.config(state="disabled")

        # Bind keyboard shortcuts
        self.root.bind("<Left>", lambda e: self.go_previous())
        self.root.bind("<Right>", lambda e: self.go_next())
        self.root.bind("<Up>", lambda e: self.go_previous())
        self.root.bind("<Down>", lambda e: self.go_next())

    def browse_file(self):
        path = filedialog.askopenfilename(
            title="Select JSONL File",
            filetypes=[("JSONL files", "*.jsonl"), ("All files", "*.*")],
            initialfile=self.file_var.get()
        )
        if path:
            self.file_var.set(path)

    def load_data(self):
        path = self.file_var.get()
        try:
            self.data = []
            with open(path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self.data.append(json.loads(line))
                    except json.JSONDecodeError:
                        messagebox.showwarning("Parse Error", f"Invalid JSON on line {line_num}")
            
            self.filepath = path
            self.apply_filters()
            messagebox.showinfo("Success", f"Loaded {len(self.data)} entries.")
        except FileNotFoundError:
            messagebox.showerror("Error", "File not found.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {e}")

    def apply_filters(self):
        self.filtered_indices = []
        label_filter = self.label_var.get()
        regex_pattern = self.regex_var.get()
        regex_field = self.regex_field_var.get()
        score_filter = SCORE_FILTERS.index(self.score_var.get())

        try:
            regex = re.compile(regex_pattern, re.IGNORECASE) if regex_pattern else None
        except re.error:
            messagebox.showerror("Regex Error", "Invalid regular expression pattern.")
            return

        for i, item in enumerate(self.data):
            # 1. Label Filter
            # "null" matches rows whose label is missing, null or empty.
            label = item.get("label") or "null"
            if label_filter != "All" and label != label_filter:
                continue
            
            # 2. Model score filter
            score = item.get(SCORE_FIELD)
            if score_filter == 1 and not (score is not None and UNCERTAIN[0] <= score <= UNCERTAIN[1]):
                continue
            if score_filter == 2 and score is None:
                continue
            if score_filter == 3 and (SCORE_FIELD not in item or score is not None):
                continue

            # 3. Regex Filter
            if regex:
                match = False
                if regex_field in ['address', 'both'] and regex.search(str(item.get("address", ""))):
                    match = True
                if regex_field in ['text', 'both'] and regex.search(str(item.get("text", ""))):
                    match = True
                
                if not match:
                    continue
            
            self.filtered_indices.append(i)

        if self.sort_var.get() == SORT_ORDERS[1]:
            # Closest to 0.5 first; rows without a score last.
            def uncertainty(i):
                score = self.data[i].get(SCORE_FIELD)
                return abs(score - 0.5) if score is not None else 1.0
            self.filtered_indices.sort(key=uncertainty)

        self.current_index = 0  # Reset to first entry
        self.update_treeview()
        
        # Auto-select first entry if available
        if self.filtered_indices:
            self.select_entry_by_position(0)

    def update_treeview(self):
        for row in self.tree.get_children():
            self.tree.delete(row)
        
        for i in self.filtered_indices:
            item = self.data[i]
            text_preview = item.get("text", "")
            if len(text_preview) > 60:
                text_preview = text_preview[:60] + "..."
            
            self.tree.insert("", "end", iid=i, values=(
                i, 
                item.get("label") or "null",
                "—" if item.get(SCORE_FIELD) is None else f"{item[SCORE_FIELD]:.2f}",
                item.get("address", ""), 
                text_preview
            ))

    def on_select(self, event):
        selected = self.tree.selection()
        if not selected:
            return
        
        idx = int(selected[0])
        # Update current_index based on selection
        if idx in self.filtered_indices:
            self.current_index = self.filtered_indices.index(idx)
        
        self.display_entry(idx)

    def select_entry_by_position(self, position):
        """Select entry by position in filtered list"""
        if not self.filtered_indices or position < 0 or position >= len(self.filtered_indices):
            return
        
        self.current_index = position
        idx = self.filtered_indices[position]
        
        # Select in treeview
        self.tree.selection_set(idx)
        self.tree.focus(idx)
        self.tree.see(idx)
        
        self.display_entry(idx)

    def display_entry(self, idx):
        """Display entry details"""
        item = self.data[idx]
        
        self.detail_id_var.set(f"Index: {idx}")
        self.position_var.set(f"({self.current_index + 1}/{len(self.filtered_indices)})")
        self.detail_label_var.set(item.get("label") or "ham")
        
        # Update text display with RTL support
        raw_text = item.get("text", "")
        display_text = self.format_rtl(raw_text)
        
        self.detail_text.config(state="normal")
        self.detail_text.delete("1.0", "end")
        self.detail_text.insert("end", display_text)
        self.detail_text.config(state="disabled")

    def go_previous(self):
        """Navigate to previous entry"""
        if not self.filtered_indices:
            return
        
        if self.current_index > 0:
            self.select_entry_by_position(self.current_index - 1)

    def go_next(self):
        """Navigate to next entry"""
        if not self.filtered_indices:
            return
        
        if self.current_index < len(self.filtered_indices) - 1:
            self.select_entry_by_position(self.current_index + 1)

    def format_rtl(self, text):
        """Formats text for proper Right-to-Left display if libraries are available."""
        if RTL_SUPPORT and text:
            try:
                reshaped_text = arabic_reshaper.reshape(text)
                return get_display(reshaped_text)
            except Exception:
                pass
        return text

    def update_label(self):
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("Warning", "Please select an entry from the list first.")
            return
        
        idx = int(selected[0])
        new_label = self.detail_label_var.get()
        
        # Update data
        self.data[idx]["label"] = new_label
        
        # Refresh UI
        self.update_treeview()
        self.tree.selection_set(idx)
        self.tree.focus(idx)
        messagebox.showinfo("Success", f"Entry {idx} label updated to '{new_label}'")

    def save_data(self):
        if not self.filepath:
            messagebox.showwarning("Warning", "No file is currently loaded.")
            return
        
        try:
            write_jsonl(self.data, self.filepath)
            messagebox.showinfo("Success", "File saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    # Try to set a modern theme if available
    style = ttk.Style()
    if 'clam' in style.theme_names():
        style.theme_use('clam')
    
    app = SMSLabelerApp(root, *sys.argv[1:2])
    app.load_data()
    root.mainloop()
