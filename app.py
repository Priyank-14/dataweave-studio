import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import os
import warnings
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────────────────────
BG          = "#f4faf4"   # Main window background (light green-white)
SIDEBAR     = "#1a5c2a"   # Sidebar background (dark green)
BTN_GREEN   = "#2e7d32"   # Standard action button colour
BTN_HOVER   = "#1b5e20"   # Button hover state (darker green)
ACCENT      = "#66bb6a"   # Accent / highlight colour
TEXT_DARK   = "#1a1a1a"   # Primary text colour
TEXT_LIGHT  = "#ffffff"   # Light text (used on dark backgrounds)
TABLE_ALT   = "#e8f5e9"   # Alternating row colour in the data table
NOTE_BG     = "#fff8e1"   # Log panel background (warm yellow)
NOTE_BORDER = "#f9a825"   # Log panel border colour
BTN_NORM    = "#1565c0"   # Normalization button colour (blue)
BTN_NORM_H  = "#0d47a1"   # Normalization button hover (darker blue)

# ─────────────────────────────────────────────────────────────────────────────
# Null-like string values that should be treated as missing data.
# Any cell whose stripped value matches one of these will be replaced
# with a real NaN before any cleaning or analysis takes place.
# ─────────────────────────────────────────────────────────────────────────────
NULL_STRINGS = [
    "N.a", "N.A", "N/A", "n/a", "NA", "na",
    "null", "NULL", "Null", "none", "None", "NONE",
    "-", "--", "?", " ", ""
]


# ─────────────────────────────────────────────────────────────────────────────
# Main Application Class
# ─────────────────────────────────────────────────────────────────────────────

class DataCleaningApp(tk.Tk):
    """
    Root Tkinter window for the Data Cleaning Studio.

    State attributes
    ----------------
    df_original   : pd.DataFrame | None
        The raw DataFrame as loaded from the uploaded CSV.  Never mutated
        after upload so the user can always revert to the original view.

    df_cleaned    : pd.DataFrame | None
        The DataFrame produced after running the full cleaning pipeline.
        Populated by :meth:`_clean_data`.

    df_normalized : pd.DataFrame | None
        The DataFrame produced after Min-Max Normalization is applied to
        ``df_cleaned``.  Populated by :meth:`_apply_minmax`.
    """

    def __init__(self):
        super().__init__()
        self.title("Data Preprocessing  •  Step 1 – Data Cleaning")
        self.geometry("1280x780")
        self.configure(bg=BG)
        self.resizable(True, True)

        # Internal state — all None until the user loads a file
        self.df_original   = None
        self.df_cleaned    = None
        self.df_normalized = None
        self.df_pca        = None
        self.df_lda        = None
        self._reduction_canvas = None   # holds embedded matplotlib canvas

        self._build_ui()

    # ─────────────────────────────────────────────────────────────────────────
    # UI Construction
    # ─────────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        """
        Construct the top-level layout:
          • A fixed-height header bar with the application title
          • A body frame that holds the sidebar (left) and table area (right)
        """
        # Header bar
        header = tk.Frame(self, bg=SIDEBAR, height=60)
        header.pack(fill="x")
        header.pack_propagate(False)
        tk.Label(header, text="🧹  Data Cleaning Studio",
                 font=("Courier New", 18, "bold"),
                 bg=SIDEBAR, fg=TEXT_LIGHT).pack(side="left", padx=20, pady=10)
        tk.Label(header, text="Step 1 of 4 – Data Preprocessing",
                 font=("Courier New", 11), bg=SIDEBAR, fg=ACCENT).pack(side="right", padx=20)

        # Main body
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True)
        self._build_sidebar(body)
        self._build_table_area(body)

    def _build_sidebar(self, parent):
        """
        Build the left-hand control sidebar.

        The sidebar contains (top to bottom):
          1. Upload button  – always active
          2. Clean Data     – enabled after a file is loaded
          3. Show Original  – enabled after cleaning
          4. Show Cleaned   – enabled after cleaning
          5. Normalize (Min-Max) – enabled after cleaning (blue)
          6. Show Normalized     – enabled after normalization
          7. Save Cleaned CSV    – enabled after cleaning
          8. Data Integration placeholder (greyed out, Step 2)
          9. Stats label at the bottom

        Parameters
        ----------
        parent : tk.Frame
            The body frame that holds both sidebar and table area.
        """
        sidebar = tk.Frame(parent, bg=SIDEBAR, width=230)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        tk.Label(sidebar, text="Controls", font=("Courier New", 13, "bold"),
                 bg=SIDEBAR, fg=ACCENT).pack(pady=(30, 10))
        tk.Frame(sidebar, bg=ACCENT, height=1).pack(fill="x", padx=20, pady=4)

        # ── Primary action buttons ──
        self._sidebar_btn(sidebar, "📂  Upload Data",      self._upload_data)
        tk.Frame(sidebar, bg="#2d6e3a", height=1).pack(fill="x", padx=20, pady=10)
        self._sidebar_btn(sidebar, "✨  Clean Data",       self._clean_data,    state="disabled", ref="clean_btn")
        self._sidebar_btn(sidebar, "🔍  Show Original",    self._show_original, state="disabled", ref="orig_btn")
        self._sidebar_btn(sidebar, "✅  Show Cleaned",     self._show_cleaned,  state="disabled", ref="clean_view_btn")
        tk.Frame(sidebar, bg="#2d6e3a", height=1).pack(fill="x", padx=20, pady=10)

        # ── Min-Max Normalization button (styled in blue to distinguish it
        #    from the cleaning group above) ──────────────────────────────────
        self.norm_btn = tk.Button(
            sidebar,
            text="📐  Normalize (Min-Max)",
            font=("Courier New", 10, "bold"),
            bg=BTN_NORM, fg=TEXT_LIGHT,
            activebackground=BTN_NORM_H, activeforeground=TEXT_LIGHT,
            relief="flat", bd=0, cursor="hand2",
            padx=10, pady=10, state="disabled",
            command=self._apply_minmax
        )
        self.norm_btn.pack(fill="x", padx=14, pady=5)
        self.norm_btn.bind(
            "<Enter>",
            lambda e: self.norm_btn.config(bg=BTN_NORM_H)
            if str(self.norm_btn["state"]) == "normal" else None
        )
        self.norm_btn.bind(
            "<Leave>",
            lambda e: self.norm_btn.config(bg=BTN_NORM)
            if str(self.norm_btn["state"]) == "normal" else None
        )

        self._sidebar_btn(sidebar, "📊  Show Normalized",  self._show_normalized, state="disabled", ref="norm_view_btn")

        tk.Frame(sidebar, bg="#2d6e3a", height=1).pack(fill="x", padx=20, pady=10)
        self._sidebar_btn(sidebar, "💾  Save Cleaned CSV", self._save_csv,      state="disabled", ref="save_btn")
        tk.Frame(sidebar, bg="#2d6e3a", height=1).pack(fill="x", padx=20, pady=10)

        # Step 2 placeholder (not yet implemented)
        self._showcase_btn(sidebar, "🔗  Data Integration")

        tk.Frame(sidebar, bg="#2d6e3a", height=1).pack(fill="x", padx=20, pady=10)

        # ── Data Reduction dropdown ─────────────────────────────────────────
        self.reduction_var = tk.StringVar(value="📉  Data Reduction")
        self.reduction_menu_btn = tk.Menubutton(
            sidebar,
            textvariable=self.reduction_var,
            font=("Courier New", 10, "bold"),
            bg="#6a1b9a", fg=TEXT_LIGHT,
            activebackground="#4a148c", activeforeground=TEXT_LIGHT,
            relief="flat", bd=0, cursor="hand2",
            padx=10, pady=10,
            direction="below"
        )
        self.reduction_menu_btn.pack(fill="x", padx=14, pady=5)

        reduction_menu = tk.Menu(
            self.reduction_menu_btn, tearoff=0,
            bg="#4a148c", fg=TEXT_LIGHT,
            activebackground="#7b1fa2", activeforeground=TEXT_LIGHT,
            font=("Courier New", 10, "bold")
        )
        reduction_menu.add_command(label="PCA  –  Principal Component Analysis",
                                   command=lambda: self._apply_reduction("PCA"))
        reduction_menu.add_command(label="LDA  –  Linear Discriminant Analysis",
                                   command=lambda: self._apply_reduction("LDA"))
        self.reduction_menu_btn["menu"] = reduction_menu

        tk.Frame(sidebar, bg="#2d6e3a", height=1).pack(fill="x", padx=20, pady=10)

        # ── Visualization button ───────────────────────────────────────────────────────
        BTN_VIZ   = "#b5451b"
        BTN_VIZ_H = "#7c2d0e"
        self.viz_btn = tk.Button(
            sidebar,
            text="📈  Visualization",
            font=("Courier New", 10, "bold"),
            bg=BTN_VIZ, fg=TEXT_LIGHT,
            activebackground=BTN_VIZ_H, activeforeground=TEXT_LIGHT,
            relief="flat", bd=0, cursor="hand2",
            padx=10, pady=10, state="disabled",
            command=self._show_visualization
        )
        self.viz_btn.pack(fill="x", padx=14, pady=5)
        self.viz_btn.bind("<Enter>",
            lambda e: self.viz_btn.config(bg=BTN_VIZ_H)
            if str(self.viz_btn["state"]) == "normal" else None)
        self.viz_btn.bind("<Leave>",
            lambda e: self.viz_btn.config(bg=BTN_VIZ)
            if str(self.viz_btn["state"]) == "normal" else None)

        # Dataset stats label – updated after every upload
        self.stats_label = tk.Label(sidebar, text="", font=("Courier New", 9),
                                    bg=SIDEBAR, fg="#a5d6a7", wraplength=200, justify="left")
        self.stats_label.pack(padx=14, pady=(20, 0), anchor="w")

    def _sidebar_btn(self, parent, text, cmd, state="normal", ref=None):
        """
        Create and pack a standard (green) sidebar button.

        Parameters
        ----------
        parent : tk.Widget
            Container widget to pack the button into.
        text   : str
            Button label (may include emoji prefix).
        cmd    : callable
            Function to call when the button is clicked.
        state  : str, optional
            Initial Tkinter state: "normal" or "disabled". Default "normal".
        ref    : str | None, optional
            If provided, the button is stored as ``self.<ref>`` so other
            methods can enable / disable it programmatically.
        """
        btn = tk.Button(parent, text=text, font=("Courier New", 10, "bold"),
                        bg=BTN_GREEN, fg=TEXT_LIGHT,
                        activebackground=BTN_HOVER, activeforeground=TEXT_LIGHT,
                        relief="flat", bd=0, cursor="hand2",
                        padx=10, pady=10, state=state, command=cmd)
        btn.pack(fill="x", padx=14, pady=5)
        if ref:
            setattr(self, ref, btn)
        btn.bind("<Enter>", lambda e, b=btn: b.config(bg=BTN_HOVER) if str(b["state"]) == "normal" else None)
        btn.bind("<Leave>", lambda e, b=btn: b.config(bg=BTN_GREEN) if str(b["state"]) == "normal" else None)

    def _showcase_btn(self, parent, text):
        """
        Create a display-only placeholder button for an upcoming pipeline step.

        The button is intentionally styled grey and non-interactive to signal
        that this feature is not yet available, while still communicating the
        overall preprocessing roadmap to the user.

        Parameters
        ----------
        parent : tk.Widget
            Container widget.
        text   : str
            Label shown on the placeholder button.
        """
        btn = tk.Button(parent, text=text,
                        font=("Courier New", 10, "bold"),
                        bg="#4a4a4a", fg="#aaaaaa",
                        activebackground="#4a4a4a", activeforeground="#aaaaaa",
                        relief="flat", bd=0, cursor="arrow",
                        padx=10, pady=10,
                        command=lambda: None)
        btn.pack(fill="x", padx=14, pady=5)
        tk.Label(parent, text="  Step 2 – coming next",
                 font=("Courier New", 8, "italic"),
                 bg=SIDEBAR, fg="#555555").pack(anchor="w", padx=18)

    def _build_table_area(self, parent):
        """
        Build the right-hand panel with a vertical PanedWindow:
          top  – table + log panel (always visible)
          bottom – matplotlib graph (added when Data Reduction runs)
        """
        self._right_panel = tk.Frame(parent, bg=BG)
        self._right_panel.pack(side="left", fill="both", expand=True)

        # Status bar
        self.status_var = tk.StringVar(value="No file loaded  •  Upload a CSV to begin")
        tk.Label(self._right_panel, textvariable=self.status_var, font=("Courier New", 10),
                 bg="#dcedc8", fg="#33691e", anchor="w", padx=10, pady=6).pack(fill="x")

        # Vertical PanedWindow
        self._paned = tk.PanedWindow(self._right_panel, orient="vertical",
                                     bg=BG, sashwidth=6, sashrelief="flat")
        self._paned.pack(fill="both", expand=True)

        # Top pane: table + log
        top_pane = tk.Frame(self._paned, bg=BG)
        self._paned.add(top_pane, stretch="always")

        table_frame = tk.Frame(top_pane, bg=BG)
        table_frame.pack(fill="both", expand=True, padx=16, pady=(10, 4))
        self.table_label = tk.Label(table_frame, text="Data Preview",
                                    font=("Courier New", 12, "bold"), bg=BG, fg=BTN_GREEN)
        self.table_label.pack(anchor="w", pady=(0, 6))

        tree_container = tk.Frame(table_frame, bg=BG)
        tree_container.pack(fill="both", expand=True)

        self.tree = ttk.Treeview(tree_container, show="headings")
        vsb = ttk.Scrollbar(tree_container, orient="vertical",   command=self.tree.yview)
        hsb = ttk.Scrollbar(tree_container, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side="right",  fill="y")
        hsb.pack(side="bottom", fill="x")
        self.tree.pack(fill="both", expand=True)
        self._style_treeview()

        # Log panel inside top pane
        self.note_frame = tk.Frame(top_pane, bg=NOTE_BG,
                                   highlightbackground=NOTE_BORDER, highlightthickness=2)
        self.note_frame.pack(fill="x", padx=16, pady=(4, 12))
        self.note_text = tk.Text(self.note_frame, height=6, font=("Courier New", 10),
                                 bg=NOTE_BG, fg="#5d4037", relief="flat",
                                 state="disabled", wrap="word", padx=10, pady=8)
        self.note_text.pack(fill="x")
        self._note("📋  Cleaning log will appear here once you run Clean Data.")

        # Bottom pane: graph placeholder (added dynamically)
        self._graph_pane = tk.Frame(self._paned, bg=BG)
        self._graph_pane_added = False

    def _style_treeview(self):
        """
        Apply a custom visual theme to the ttk.Treeview widget.

        Uses the "clam" base theme (cross-platform) and overrides heading and
        row colours to match the application's green colour palette.
        """
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Treeview", background="white", foreground=TEXT_DARK,
                        rowheight=24, fieldbackground="white", font=("Courier New", 10))
        style.configure("Treeview.Heading", background=BTN_GREEN, foreground="white",
                        font=("Courier New", 10, "bold"), relief="flat")
        style.map("Treeview.Heading", background=[("active", BTN_HOVER)])
        style.map("Treeview",         background=[("selected", ACCENT)])

    # ─────────────────────────────────────────────────────────────────────────
    # Data Actions
    # ─────────────────────────────────────────────────────────────────────────

    def _upload_data(self):
        """
        Open a file-chooser dialog, load the selected CSV into a DataFrame,
        and display it in the table.

        Behaviour
        ---------
        • Accepts any CSV file via ``filedialog.askopenfilename``.
        • Uses ``NULL_STRINGS`` as additional ``na_values`` so common
          null-like tokens are automatically parsed as NaN on load.
        • Resets ``df_cleaned``, ``df_normalized``, and all button states so
          the UI is consistent for a fresh file.
        • Shows a ``messagebox`` error if the file cannot be read.
        """
        path = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not path:
            return  # User cancelled the dialog — do nothing

        try:
            df = pd.read_csv(path, na_values=NULL_STRINGS, keep_default_na=True)
            self.df_original   = df
            self.df_cleaned    = None
            self.df_normalized = None
            self._render_table(df)

            fname = os.path.basename(path)
            r, c  = df.shape
            self.status_var.set(f"📄  {fname}   •   {r} rows × {c} columns")
            self.table_label.config(text="Data Preview  (Original)")
            self.stats_label.config(text=f"Rows : {r}\nCols : {c}\nFile : {fname}")

            # Enable only the buttons that make sense after a fresh upload
            self.clean_btn.config(state="normal")
            self.orig_btn.config(state="disabled")
            self.clean_view_btn.config(state="disabled")
            self.save_btn.config(state="disabled")
            self.norm_btn.config(state="disabled")
            self.norm_view_btn.config(state="disabled")
            self.viz_btn.config(state="disabled")

            self._note("📋  Cleaning log will appear here once you run Clean Data.")
        except Exception as e:
            messagebox.showerror("Error", f"Could not read file:\n{e}")


    """ data cleaning"""
    def _clean_data(self):
        """
        Run the full 6-step cleaning pipeline on ``df_original`` and store
        the result in ``df_cleaned``.

        Pipeline steps
        --------------
        0. **Null-string replacement**
               Strip whitespace from every object column and replace any cell
               whose value matches ``NULL_STRINGS`` with ``np.nan``.  Also
               catches residual ``"nan"`` strings left by earlier processing.

        1. **Remove fully-empty rows**
               Drops rows where *every* cell is NaN.

        2. **Remove duplicate rows**
               Drops rows that are exact duplicates (all columns identical).

        3. **Auto-convert string columns to float**
               For each object column, attempts ``pd.to_numeric``.  If >= 50%
               of non-null values parse successfully the entire column is
               converted (partial failures become NaN).

        4. **Fill missing values**
               - Numeric columns  -> filled with the column median (robust
                 to remaining outliers).
               - Categorical columns -> filled with the column mode
                 (most frequent value), or ``"Unknown"`` if no mode exists.

        5. **Verify fill success**
               Logs any cells that could not be filled (e.g. a column that
               contained only NaN — no valid value exists to compute from).

        6. **Outlier removal (IQR method)**
               For each numeric column with >= 4 non-null values:
               Computes Q1, Q3, IQR = Q3 - Q1.
               Removes rows where the value falls outside
               [Q1 - 1.5 * IQR,  Q3 + 1.5 * IQR].
               Columns with IQR == 0 (constant values) are skipped.

        Side effects
        ------------
        • Populates ``self.df_cleaned``.
        • Resets ``self.df_normalized`` to ``None`` (a re-clean invalidates
          any previously computed normalization).
        • Updates the table view, status bar, and log panel.
        • Enables the normalization button; disables the clean button.
        """
        if self.df_original is None:
            return

        df  = self.df_original.copy()
        log = []
        initial_rows = len(df)

        # ── STEP 0: Replace ALL null-like strings with real NaN ────────────
        null_string_count = 0
        for col in df.columns:
            if df[col].dtype == object:
                cleaned_col = df[col].astype(str).str.strip()
                was_null    = cleaned_col.isin(NULL_STRINGS)
                null_string_count += was_null.sum()
                cleaned_col = cleaned_col.where(~was_null, other=np.nan)
                # Catch residual "nan" text that may appear after astype(str)
                cleaned_col = cleaned_col.where(cleaned_col.str.lower() != "nan", other=np.nan)
                df[col] = cleaned_col

        if null_string_count:
            log.append(f"🔄  'N.a' / null-like strings found     : {null_string_count}  → converted to NaN")

        # ── STEP 1: Remove fully-empty rows ────────────────────────────────
        empty_rows = int(df.isnull().all(axis=1).sum())
        df = df.dropna(how="all")
        if empty_rows:
            log.append(f"🗑️  Fully-empty rows removed            : {empty_rows}")
        else:
            log.append("✅  Fully-empty rows                    : 0 found")

        # ── STEP 2: Remove duplicate rows ──────────────────────────────────
        dup_count = int(df.duplicated().sum())
        df = df.drop_duplicates()
        if dup_count:
            log.append(f"🗑️  Duplicate rows removed              : {dup_count}")
        else:
            log.append("✅  Duplicate rows                      : 0 found")

        # ── STEP 3: Convert numeric-looking string columns to float ────────
        for col in df.columns:
            if df[col].dtype == object:
                conv     = pd.to_numeric(df[col], errors="coerce")
                non_null = int(df[col].notna().sum())
                ok       = int(conv.notna().sum())
                # Only convert if the majority of values parse successfully
                if non_null > 0 and (ok / non_null) >= 0.5:
                    df[col] = conv

        # ── STEP 4: Fill missing values (CoW-safe via df.assign) ───────────
        null_counts    = df.isnull().sum()
        total_nulls    = int(null_counts.sum())
        cols_with_null = null_counts[null_counts > 0].index.tolist()

        if cols_with_null:
            for col in cols_with_null:
                n = int(null_counts[col])
                if pd.api.types.is_numeric_dtype(df[col]):
                    non_null_vals = df[col].dropna()
                    if len(non_null_vals) > 0:
                        fill_val = float(non_null_vals.median())
                        df = df.assign(**{col: df[col].fillna(fill_val)})
                        log.append(f"🔧  '{col}' – {n} nulls filled (median={fill_val:.2f})")
                    else:
                        log.append(f"⚠️   '{col}' – {n} nulls, no data to compute median (skipped)")
                else:
                    modes = df[col].dropna().mode()
                    if len(modes) > 0:
                        fill_val = modes.iloc[0]
                        df = df.assign(**{col: df[col].fillna(fill_val)})
                        log.append(f"🔧  '{col}' – {n} nulls filled (mode='{fill_val}')")
                    else:
                        # Fallback: no mode could be computed (all values were NaN)
                        df = df.assign(**{col: df[col].fillna("Unknown")})
                        log.append(f"🔧  '{col}' – {n} nulls filled with 'Unknown'")
        else:
            log.append("✅  Missing / N.a values                : none found")

        # ── STEP 5: Verify fill worked ──────────────────────────────────────
        remaining_nulls = int(df.isnull().sum().sum())
        if remaining_nulls > 0:
            log.append(f"ℹ️   Note: {remaining_nulls} cells still empty "
                       f"(columns with NO valid data to compute a fill value)")

        # ── STEP 6: Outlier removal via IQR ────────────────────────────────
        numeric_cols   = df.select_dtypes(include=[np.number]).columns.tolist()
        total_outliers = 0

        for col in numeric_cols:
            non_null_col = df[col].dropna()
            if len(non_null_col) < 4:
                # Not enough data points to compute a meaningful IQR
                continue
            Q1, Q3 = non_null_col.quantile(0.25), non_null_col.quantile(0.75)
            IQR    = Q3 - Q1
            if IQR == 0:
                # Constant column — every value is the same, skip
                continue
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            mask  = (df[col] < lower) | (df[col] > upper)
            n_out = int(mask.sum())
            if n_out:
                df = df[~mask]   # Keep only the inlier rows
                total_outliers += n_out
                log.append(f"📉  '{col}' – {n_out} outliers removed "
                            f"(bounds: {lower:.2f} … {upper:.2f})")

        if total_outliers == 0:
            log.append("✅  Outliers (IQR method)               : none found")

        # ── Summary ─────────────────────────────────────────────────────────
        df = df.reset_index(drop=True)
        final_rows   = len(df)
        rows_removed = initial_rows - final_rows

        log.append("─" * 56)
        log.append(f"📊  Initial rows                        : {initial_rows}")
        log.append(f"📊  Final rows                          : {final_rows}")
        log.append(f"📊  Rows removed                        : {rows_removed}")
        log.append(f"📊  Null / N.a strings replaced         : {null_string_count}")
        log.append(f"📊  Null values handled (fill/impute)   : {total_nulls}")
        log.append(f"📊  Outliers removed                    : {total_outliers}")

        # ── Persist result and update UI ────────────────────────────────────
        self.df_cleaned    = df
        self.df_normalized = None   # Invalidate any previous normalization
        self._render_table(df)
        self.table_label.config(text="Data Preview  (Cleaned ✅)")
        r, c = df.shape
        self.status_var.set(f"✅  Cleaned data   •   {r} rows × {c} columns")
        self.orig_btn.config(state="normal")
        self.clean_view_btn.config(state="normal")
        self.save_btn.config(state="normal")
        self.clean_btn.config(state="disabled")
        self.norm_btn.config(state="normal")      # Unlocked after cleaning
        self.norm_view_btn.config(state="disabled")
        self.viz_btn.config(state="normal")         # Visualization unlocked after cleaning
        self._note("\n".join(log))


    """ data transformation"""



    def _apply_minmax(self):
        """
        Apply Min-Max Normalization to all numeric columns of ``df_cleaned``
        and store the result in ``df_normalized``.

        Method
        ------
        **Min-Max Normalization** (also called *feature scaling*) linearly
        transforms each numeric column so its values fall in [0.0, 1.0]:

            X_norm = (X - X_min) / (X_max - X_min)

        Where:
          • X_min  — minimum value of the column
          • X_max  — maximum value of the column

        Properties
        ----------
        • Preserves the relative distances between all values.
        • Sensitive to outliers (use only on cleaned data).
        • Non-numeric columns are left unchanged.
        • Constant columns (X_min == X_max) are set to 0.0 to avoid
          division by zero.

        Side effects
        ------------
        • Populates ``self.df_normalized``.
        • Updates the table to show the normalized data.
        • Updates the status bar and log panel with per-column details.
        • Disables the normalization button; enables "Show Normalized".
        """
        if self.df_cleaned is None:
            messagebox.showinfo("Info", "Please run Clean Data first.")
            return

        df  = self.df_cleaned.copy()
        log = []

        # Header block in the log panel
        log.append("━" * 56)
        log.append("  📐  Normalization Method : Min-Max Normalization")
        log.append("  Formula  :  X_norm = (X − X_min) / (X_max − X_min)")
        log.append("  Output range  :  [0.0 , 1.0]  for every numeric column")
        log.append("━" * 56)

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            messagebox.showwarning("No Numeric Columns",
                                   "No numeric columns found to normalize.")
            return

        for col in numeric_cols:
            col_min   = df[col].min()
            col_max   = df[col].max()
            col_range = col_max - col_min

            if col_range == 0:
                # All values are identical — normalization is undefined.
                # Set to 0.0 as a safe, deterministic fallback.
                df[col] = 0.0
                log.append(
                    f"⚠️   '{col}'  –  constant column (min=max={col_min:.4f})  → set to 0.0"
                )
            else:
                df[col] = (df[col] - col_min) / col_range
                log.append(
                    f"✅  '{col}'  –  min={col_min:.4f}  max={col_max:.4f}  "
                    f"→  scaled to [0.0 , 1.0]"
                )

        # Summary footer
        log.append("─" * 56)
        log.append(f"📊  Numeric columns normalized          : {len(numeric_cols)}")
        log.append(f"📊  Non-numeric columns (unchanged)     : {len(df.columns) - len(numeric_cols)}")
        log.append("ℹ️   All values are now in the range [0, 1].")

        self.df_normalized = df.reset_index(drop=True)
        self._render_table(self.df_normalized)
        self.table_label.config(text="Data Preview  (Min-Max Normalized 📐)")
        r, c = self.df_normalized.shape
        self.status_var.set(
            f"📐  Min-Max Normalized data   •   {r} rows × {c} columns   •   "
            f"numeric values scaled to [0, 1]"
        )
        self.norm_btn.config(state="disabled")     # Already normalized — no need to re-run
        self.norm_view_btn.config(state="normal")  # User can now switch back to this view
        self._note("\n".join(log))

    # ─────────────────────────────────────────────────────────────────────────
    # Data Reduction  (Step 4)
    # ─────────────────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Visualization  (Stats + Charts of cleaned data)
    # ─────────────────────────────────────────────────────────────────────────

    def _show_visualization(self):
        """Show basic stats (mean/median/mode) table + bar/box/hist charts of cleaned data."""
        if self.df_cleaned is None:
            messagebox.showinfo("Info", "Please run Clean Data first.")
            return

        df = self.df_cleaned.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            messagebox.showwarning("Warning", "No numeric columns found for visualization.")
            return

        def safe_mode(series):
            m = series.dropna().mode()
            return float(m.iloc[0]) if not m.empty else np.nan

        # ── Stats table ────────────────────────────────────────────────────
        stats_rows = []
        for col in numeric_cols:
            s = df[col].dropna()
            if s.empty:
                continue
            mv = safe_mode(s)
            stats_rows.append({
                "Column"  : col,
                "Mean"    : round(s.mean(),   3),
                "Median"  : round(s.median(), 3),
                "Mode"    : round(mv, 3) if not np.isnan(mv) else "-",
                "Std Dev" : round(s.std(),    3),
                "Min"     : round(s.min(),    3),
                "Max"     : round(s.max(),    3),
            })

        df_stats = pd.DataFrame(stats_rows)
        self._render_table(df_stats)
        self.table_label.config(text="📊  Statistics Summary  (Cleaned Data)")
        self.status_var.set(
            f"📈  Visualization   •   {len(stats_rows)} numeric columns   "
            f"•   {len(df)} rows"
        )

        log_lines = [
            "━" * 56,
            "  📈  VISUALIZATION – Basic Statistics (Cleaned Data)",
            "━" * 56,
        ]
        for row in stats_rows:
            mode_str = f"{row['Mode']:.3f}" if row["Mode"] != "-" else "  -   "
            log_lines.append(
                f"  {row['Column'][:14]:<14}  "
                f"mean={row['Mean']:.3f}  "
                f"median={row['Median']:.3f}  "
                f"mode={mode_str}  "
                f"std={row['Std Dev']:.3f}"
            )
        log_lines.append("─" * 56)
        log_lines.append(f"  Columns visualised : {len(stats_rows)}")
        log_lines.append(f"  Rows in dataset    : {len(df)}")
        self._note("\n".join(log_lines))

        # ── Clear and prepare graph pane ───────────────────────────────────
        for widget in self._graph_pane.winfo_children():
            widget.destroy()
        if self._reduction_canvas is not None:
            plt.close("all")
            self._reduction_canvas = None

        if not self._graph_pane_added:
            self._paned.add(self._graph_pane, stretch="never", minsize=320)
            self._graph_pane_added = True

        tk.Label(
            self._graph_pane,
            text="📈  Data Visualization  –  Cleaned Dataset",
            font=("Courier New", 11, "bold"),
            bg=BG, fg="#b5451b"
        ).pack(anchor="w", padx=16, pady=(8, 2))

        # Only plot cols with actual data; cap at 6 for readability
        plot_cols    = [c for c in numeric_cols if not df[c].dropna().empty][:6]
        n_cols       = len(plot_cols)
        short_labels = [c[:8] for c in plot_cols]
        means        = [df[c].mean()   for c in plot_cols]
        medians      = [df[c].median() for c in plot_cols]
        modes_vals   = [safe_mode(df[c]) for c in plot_cols]
        modes_plot   = [v if not np.isnan(v) else 0 for v in modes_vals]

        fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), dpi=90)
        fig.patch.set_facecolor("#f4faf4")
        x     = np.arange(n_cols)
        width = 0.26

        # Chart 1: Grouped bar – Mean / Median / Mode
        ax1 = axes[0]
        ax1.set_facecolor("#e8f5e9")
        ax1.bar(x - width, means,      width, label="Mean",   color="#2e7d32", alpha=0.85)
        ax1.bar(x,         medians,    width, label="Median", color="#1565c0", alpha=0.85)
        ax1.bar(x + width, modes_plot, width, label="Mode",   color="#b5451b", alpha=0.85)
        ax1.set_xticks(x)
        ax1.set_xticklabels(short_labels, rotation=40, ha="right", fontsize=7)
        ax1.set_title("Mean / Median / Mode", fontsize=9, fontweight="bold", color="#333")
        ax1.legend(fontsize=7)
        ax1.tick_params(labelsize=7)
        ax1.grid(axis="y", linestyle="--", alpha=0.4)

        # Chart 2: Box plot – distribution spread
        ax2 = axes[1]
        ax2.set_facecolor("#e8f5e9")
        data_to_box = [df[c].dropna().values for c in plot_cols]
        bp = ax2.boxplot(
            data_to_box, tick_labels=short_labels,
            patch_artist=True, notch=False,
            medianprops=dict(color="#b5451b", linewidth=2)
        )
        palette = ["#a5d6a7","#81c784","#66bb6a","#4caf50","#43a047","#388e3c"]
        for patch, clr in zip(bp["boxes"], palette[:n_cols]):
            patch.set_facecolor(clr)
            patch.set_alpha(0.78)
        ax2.set_title("Box Plot  (Distribution)", fontsize=9, fontweight="bold", color="#333")
        ax2.set_xticklabels(short_labels, rotation=40, ha="right", fontsize=7)
        ax2.tick_params(labelsize=7)
        ax2.grid(axis="y", linestyle="--", alpha=0.4)

        # Chart 3: Histogram of first column with mean/median lines
        ax3 = axes[2]
        ax3.set_facecolor("#e8f5e9")
        primary_col  = plot_cols[0]
        primary_data = df[primary_col].dropna()
        ax3.hist(primary_data, bins=min(12, max(5, len(primary_data))),
                 color="#1565c0", alpha=0.80, edgecolor="white", linewidth=0.6)
        ax3.axvline(primary_data.mean(),   color="#2e7d32", linestyle="--",
                    linewidth=1.8, label=f"Mean {primary_data.mean():.1f}")
        ax3.axvline(primary_data.median(), color="#b5451b", linestyle="-.",
                    linewidth=1.8, label=f"Median {primary_data.median():.1f}")
        ax3.set_title(f"Histogram  –  {primary_col[:12]}", fontsize=9,
                      fontweight="bold", color="#333")
        ax3.legend(fontsize=7)
        ax3.tick_params(labelsize=7)
        ax3.grid(axis="y", linestyle="--", alpha=0.4)

        fig.tight_layout(pad=1.4)

        canvas = FigureCanvasTkAgg(fig, master=self._graph_pane)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=16, pady=(0, 10))
        self._reduction_canvas = canvas

    def _apply_reduction(self, method: str):
        """Apply PCA or LDA to df_normalized and show the result table + graph."""
        if self.df_normalized is None:
            messagebox.showinfo(
                "Info",
                "Please run Normalize (Min-Max) first before applying Data Reduction."
            )
            return

        df  = self.df_normalized.copy()
        log = []

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        non_num_cols = [c for c in df.columns if c not in numeric_cols]

        if len(numeric_cols) < 2:
            messagebox.showwarning("Warning", "Need at least 2 numeric columns for reduction.")
            return

        X_raw = df[numeric_cols].values

        # ── Drop feature columns that still contain NaN after normalization ──
        valid_col_mask = ~np.isnan(X_raw).any(axis=0)
        X = X_raw[:, valid_col_mask]
        valid_numeric_cols = [c for c, ok in zip(numeric_cols, valid_col_mask) if ok]

        # Drop any remaining rows with NaN (shouldn't happen but safety net)
        row_mask = ~np.isnan(X).any(axis=1)
        X = X[row_mask]
        df_clean = df[row_mask].reset_index(drop=True)

        if X.shape[0] < 3 or X.shape[1] < 2:
            messagebox.showwarning("Warning", "Not enough valid data after cleaning for reduction.")
            return

        log.append("━" * 56)
        log.append(f"  Valid features used  : {X.shape[1]} / {len(numeric_cols)}")
        log.append(f"  Valid samples used   : {X.shape[0]}")

        if method == "PCA":
            n_components = min(2, X.shape[1])
            reducer = PCA(n_components=n_components)
            X_reduced = reducer.fit_transform(X)
            col_names = [f"PC{i+1}" for i in range(n_components)]
            ev = reducer.explained_variance_ratio_ * 100

            log.append("  🔵  Method : Principal Component Analysis (PCA)")
            log.append(f"  Output components    : {n_components}")
            for i, v in enumerate(ev):
                log.append(f"  PC{i+1} explained variance : {v:.2f}%")
            log.append(f"  Total variance kept  : {sum(ev):.2f}%")
            log.append("━" * 56)

            df_result = pd.DataFrame(X_reduced, columns=col_names)
            for c in non_num_cols:
                df_result[c] = df_clean[c].values
            self.df_pca = df_result

            self._render_table(df_result)
            self.table_label.config(text="Data Preview  (PCA – 2 Principal Components 🔵)")
            r, c = df_result.shape
            self.status_var.set(
                f"🔵  PCA applied   •   {r} rows × {c} cols   "
                f"•   Variance kept: {sum(ev):.1f}%"
            )
            self.reduction_var.set("📉  Data Reduction  ▸  PCA ✅")
            self._note("\n".join(log))
            self._draw_reduction_plot(
                X_reduced, method="PCA",
                xlabel=f"PC1  ({ev[0]:.1f}% variance)",
                ylabel=f"PC2  ({ev[1]:.1f}% variance)" if n_components > 1 else "—",
                labels=None
            )

        else:  # LDA
            n = X.shape[0]
            # Build class labels: prefer an existing categorical column,
            # but fall back to synthetic 3-class bins (safe for any dataset size)
            if non_num_cols:
                target_col = non_num_cols[0]
                le = LabelEncoder()
                y_raw = le.fit_transform(df_clean[target_col].astype(str))
                # If every row is its own class (n_classes == n_samples),
                # collapse to 3 bins so LDA can actually work
                if len(np.unique(y_raw)) >= n:
                    n_classes_use = 3
                    y = np.array([i * n_classes_use // n for i in range(n)])
                    class_labels = np.array([f"Group {v}" for v in y])
                    target_name  = f"{target_col} (binned→3 groups)"
                else:
                    y = y_raw
                    class_labels = le.inverse_transform(y)
                    target_name  = target_col
            else:
                n_classes_use = 3
                y = np.array([i * n_classes_use // n for i in range(n)])
                class_labels = np.array([f"Group {v}" for v in y])
                target_name  = "synthetic 3-class bins"

            n_classes  = len(np.unique(y))
            n_comp_lda = min(2, n_classes - 1, X.shape[1])

            if n_comp_lda < 1:
                messagebox.showwarning("LDA", "Not enough distinct classes for LDA.")
                return

            reducer = LDA(n_components=n_comp_lda)
            X_reduced = reducer.fit_transform(X, y)
            col_names = [f"LD{i+1}" for i in range(n_comp_lda)]
            ev_ratio  = getattr(reducer, "explained_variance_ratio_", np.zeros(n_comp_lda)) * 100

            log.append("  🟣  Method : Linear Discriminant Analysis (LDA)")
            log.append(f"  Target / grouping    : {target_name}")
            log.append(f"  Classes              : {n_classes}")
            log.append(f"  Output discriminants : {n_comp_lda}")
            for i, v in enumerate(ev_ratio):
                log.append(f"  LD{i+1} explained variance : {v:.2f}%")
            log.append("━" * 56)

            df_result = pd.DataFrame(X_reduced, columns=col_names)
            df_result["Class"] = class_labels
            self.df_lda = df_result

            self._render_table(df_result)
            self.table_label.config(text="Data Preview  (LDA – Linear Discriminants 🟣)")
            r, c = df_result.shape
            self.status_var.set(
                f"🟣  LDA applied   •   {r} rows × {c} cols   •   {n_classes} classes"
            )
            self.reduction_var.set("📉  Data Reduction  ▸  LDA ✅")
            self._note("\n".join(log))
            self._draw_reduction_plot(
                X_reduced, method="LDA",
                xlabel=f"LD1  ({ev_ratio[0]:.1f}% variance)" if ev_ratio[0] > 0 else "LD1",
                ylabel=f"LD2  ({ev_ratio[1]:.1f}% variance)" if (n_comp_lda > 1 and ev_ratio[1] > 0) else "LD2",
                labels=class_labels
            )

    def _draw_reduction_plot(self, X_reduced, method, xlabel, ylabel, labels=None):
        """
        Embed a matplotlib scatter plot into the bottom pane of the PanedWindow.
        Replaces any previously embedded canvas.
        """
        # Clear previous graph widgets
        for widget in self._graph_pane.winfo_children():
            widget.destroy()
        if self._reduction_canvas is not None:
            plt.close("all")
            self._reduction_canvas = None

        # Add graph pane to paned window if not already there
        if not self._graph_pane_added:
            self._paned.add(self._graph_pane, stretch="never", minsize=280)
            self._graph_pane_added = True

        # Header label inside graph pane
        color_hex = "#1565c0" if method == "PCA" else "#6a1b9a"
        icon      = "🔵" if method == "PCA" else "🟣"
        tk.Label(
            self._graph_pane,
            text=f"{icon}  {method} – 2D Scatter Plot",
            font=("Courier New", 11, "bold"),
            bg=BG, fg=color_hex
        ).pack(anchor="w", padx=16, pady=(8, 2))

        # Build figure
        fig, ax = plt.subplots(figsize=(7, 3.0), dpi=92)
        fig.patch.set_facecolor("#f4faf4")
        ax.set_facecolor("#e8f5e9")

        y2d = X_reduced[:, 1] if X_reduced.shape[1] > 1 else np.zeros(len(X_reduced))

        if labels is not None:
            unique_labels = np.unique(labels)
            palette = plt.cm.Set2(np.linspace(0, 0.8, len(unique_labels)))
            for lbl, col in zip(unique_labels, palette):
                mask = labels == lbl
                ax.scatter(
                    X_reduced[mask, 0], y2d[mask],
                    label=str(lbl), color=col,
                    alpha=0.80, edgecolors="#ffffff", linewidths=0.6, s=70
                )
            ax.legend(
                fontsize=8, title="Class", title_fontsize=8,
                framealpha=0.7, loc="best"
            )
        else:
            ax.scatter(
                X_reduced[:, 0], y2d,
                color="#2e7d32", alpha=0.75,
                edgecolors="#ffffff", linewidths=0.6, s=70
            )

        ax.set_title(f"{method}  –  2D Projection of Normalized Data",
                     fontsize=10, color=color_hex, fontweight="bold", pad=6)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, linestyle="--", alpha=0.35, color="#a5d6a7")
        fig.tight_layout(pad=1.2)

        canvas = FigureCanvasTkAgg(fig, master=self._graph_pane)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=16, pady=(0, 10))
        self._reduction_canvas = canvas

    # ─────────────────────────────────────────────────────────────────────────
    # View Toggle Methods
    # ─────────────────────────────────────────────────────────────────────────

    def _show_original(self):
        """
        Switch the table view to display ``df_original`` (the raw uploaded data).

        Does nothing if no file has been loaded yet.
        """
        if self.df_original is None:
            return
        self._render_table(self.df_original)
        self.table_label.config(text="Data Preview  (Original)")
        r, c = self.df_original.shape
        self.status_var.set(f"🔍  Original data   •   {r} rows × {c} columns")

    def _show_cleaned(self):
        """
        Switch the table view to display ``df_cleaned`` (post-cleaning data).

        Shows an info dialog if cleaning has not been run yet.
        """
        if self.df_cleaned is None:
            messagebox.showinfo("Info", "Please run Clean Data first.")
            return
        self._render_table(self.df_cleaned)
        self.table_label.config(text="Data Preview  (Cleaned ✅)")
        r, c = self.df_cleaned.shape
        self.status_var.set(f"✅  Cleaned data   •   {r} rows × {c} columns")

    def _show_normalized(self):
        """
        Switch the table view to display ``df_normalized`` (Min-Max scaled data).

        Shows an info dialog if normalization has not been run yet.
        """
        if self.df_normalized is None:
            messagebox.showinfo("Info", "Please run Normalize first.")
            return
        self._render_table(self.df_normalized)
        self.table_label.config(text="Data Preview  (Min-Max Normalized 📐)")
        r, c = self.df_normalized.shape
        self.status_var.set(f"📐  Min-Max Normalized data   •   {r} rows × {c} columns")

    # ─────────────────────────────────────────────────────────────────────────
    # File I/O
    # ─────────────────────────────────────────────────────────────────────────

    def _save_csv(self):
        """
        Open a save dialog and write ``df_cleaned`` to the chosen CSV path.

        Notes
        -----
        • Only ``df_cleaned`` is saved (not the normalized version).
        • Row index is excluded from the output file (``index=False``).
        • Shows a success message or an info dialog if no cleaned data exists.
        """
        if self.df_cleaned is None:
            messagebox.showinfo("Info", "No cleaned data to save.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            title="Save Cleaned CSV"
        )
        if path:
            self.df_cleaned.to_csv(path, index=False)
            messagebox.showinfo("Saved", f"Cleaned data saved to:\n{path}")

    # ─────────────────────────────────────────────────────────────────────────
    # Table Rendering Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _render_table(self, df):
        """
        Populate the Treeview widget with the contents of a DataFrame.

        Behaviour
        ---------
        • Clears any existing rows and column definitions first.
        • Column widths are set to ``max(100, len(column_name) x 10 + 20)``
          so wider column names are always fully visible.
        • Rows alternate between white and ``TABLE_ALT`` (light green) for
          easier visual scanning.
        • NaN / None values are displayed as empty strings.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame whose content should be rendered.
        """
        self.tree.delete(*self.tree.get_children())
        self.tree["columns"] = list(df.columns)
        for col in df.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=max(100, len(str(col)) * 10 + 20),
                             minwidth=60, anchor="center")
        for i, (_, row) in enumerate(df.iterrows()):
            tag    = "even" if i % 2 == 0 else "odd"
            values = []
            for v in row:
                try:
                    if pd.isna(v):
                        values.append("")
                        continue
                except Exception:
                    pass
                values.append(v)
            self.tree.insert("", "end", values=values, tags=(tag,))
        self.tree.tag_configure("even", background="white")
        self.tree.tag_configure("odd",  background=TABLE_ALT)

    def _note(self, text):
        """
        Replace the contents of the log / notes panel with ``text``.

        The Text widget is kept in ``DISABLED`` state between updates to
        prevent accidental user edits.  This method temporarily enables it,
        replaces the content, then disables it again.

        Parameters
        ----------
        text : str
            The message or multi-line log string to display.
        """
        self.note_text.config(state="normal")
        self.note_text.delete("1.0", "end")
        self.note_text.insert("end", text)
        self.note_text.config(state="disabled")


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = DataCleaningApp()
    app.mainloop()