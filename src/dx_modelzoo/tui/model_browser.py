from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from textual import events, on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import Footer, Header, Input, Static, Tree
from textual.widgets.tree import TreeNode

from dx_modelzoo.loader.discovery import ModelEntry, discover_models
from dx_modelzoo.loader.model_builder import ModelBuilder


class ModelInfoPanel(Static):
    """Right-side panel showing model details."""

    DEFAULT_CSS = """
    ModelInfoPanel {
        width: 1fr;
        height: 100%;
        padding: 1 2;
        border-left: solid $accent;
        overflow-y: auto;
    }
    """

    def update_model(self, entry: ModelEntry) -> None:
        try:
            builder = ModelBuilder(entry.yaml_path)
            cfg = builder.config

            lines = []
            lines.append(f"[bold cyan]📦 {cfg['name']}[/]")
            lines.append("")

            # Domain / Task
            lines.append(f"[bold]Domain:[/] {entry.domain}")
            lines.append(f"[bold]Task:[/]   {entry.task}")
            lines.append("")

            # Inputs
            inputs = cfg.get("inputs", [])
            if inputs:
                lines.append("[bold underline]Inputs[/]")
                if isinstance(inputs, list):
                    for inp in inputs:
                        name = inp.get("name", "?")
                        shape = inp.get("shape", "?")
                        dtype = inp.get("dtype", "float32")
                        layout = inp.get("layout", "")
                        lines.append(f"  {name}: {shape} ({dtype}) {layout}")
                elif isinstance(inputs, dict):
                    for inp_name, inp_cfg in inputs.items():
                        shape = inp_cfg.get("shape", "?")
                        dtype = inp_cfg.get("dtype", "float32")
                        layout = inp_cfg.get("layout", "")
                        lines.append(f"  {inp_name}: {shape} ({dtype}) {layout}")
                lines.append("")

            # Profiles
            profiles = cfg.get("profiles", {})
            if profiles:
                lines.append("[bold underline]Profiles[/]")
                for pname, pcfg in profiles.items():
                    target = pcfg.get("target", "?")
                    lines.append(f"  [green]▸[/] {pname} → {target}")
                lines.append("")

            # Artifacts
            artifacts = cfg.get("artifacts", {})
            if artifacts:
                lines.append("[bold underline]Artifacts[/]")
                art_path = artifacts.get("path", "")
                if art_path:
                    lines.append(f"  path: {art_path}")
                lines.append("")

            # Dataset
            dataset = cfg.get("dataset", {})
            if dataset:
                lines.append("[bold underline]Dataset[/]")
                ds_type = dataset.get("type", dataset.get("name", "?"))
                lines.append(f"  type: {ds_type}")
                for k, v in dataset.items():
                    if k not in ("type", "name"):
                        lines.append(f"  {k}: {v}")
                lines.append("")

            # Preprocessing
            preproc = cfg.get("preprocessing", [])
            if preproc:
                lines.append("[bold underline]Preprocessing[/]")
                if isinstance(preproc, list):
                    for step in preproc:
                        if isinstance(step, dict) and step:
                            op = step.get("type", list(step.keys())[0])
                            lines.append(f"  - {op}")
                        else:
                            lines.append(f"  - {step}")
                elif isinstance(preproc, dict):
                    for inp_name, steps in preproc.items():
                        lines.append(f"  {inp_name}:")
                        if isinstance(steps, list):
                            for step in steps:
                                if isinstance(step, dict) and step:
                                    op = step.get("type", list(step.keys())[0])
                                    lines.append(f"    - {op}")
                                else:
                                    lines.append(f"    - {step}")
                lines.append("")

            # YAML path
            lines.append(f"[dim]YAML: {entry.yaml_path}[/]")

            self.update("\n".join(lines))
        except Exception as e:
            self.update(f"[red]Error loading model: {e}[/]")

    def show_empty(self) -> None:
        self.update("[dim]← Select a model to view details[/]")


class ActionPanel(Static):
    """Bottom panel for action selection."""

    DEFAULT_CSS = """
    ActionPanel {
        height: 5;
        padding: 1 2;
        border-top: solid $accent;
        display: none;
    }
    """

    def show_actions(self, model_name: str, profiles: list[str]) -> None:
        lines = [f"[bold yellow]⚡ Actions for {model_name}[/]"]
        lines.append("  [bold]e[/] Eval   [bold]c[/] Compile   [bold]Esc[/] Cancel")
        self.update("\n".join(lines))
        self.display = True

    def hide(self) -> None:
        self.display = False
        self.update("")


class ProfileSelectPanel(Static):
    """Panel for profile selection."""

    DEFAULT_CSS = """
    ProfileSelectPanel {
        height: 7;
        padding: 1 2;
        border-top: solid $accent;
        display: none;
    }
    """

    def show_profiles(self, action: str, profiles: list[str]) -> None:
        lines = [f"[bold yellow]Select profile for {action}:[/]"]
        # content lines = height(7) - padding(2) - border(1) = 4
        # available for profiles = 4 - header(1) - esc(1) = 2 rows
        max_rows = 2
        if len(profiles) <= max_rows:
            for i, p in enumerate(profiles, 1):
                lines.append(f"  [bold]{i}[/] {p}")
        else:
            import math

            cols = math.ceil(len(profiles) / max_rows)
            for row in range(max_rows):
                parts = []
                for col in range(cols):
                    idx = col * max_rows + row
                    if idx < len(profiles):
                        padded = profiles[idx].ljust(16)
                        parts.append(f"  [bold]{idx + 1}[/] {padded}")
                lines.append("".join(parts))
        lines.append("  [bold]Esc[/] Cancel")
        self.update("\n".join(lines))
        self.display = True

    def hide(self) -> None:
        self.display = False
        self.update("")


class ModelTreeBrowser(App):
    """Interactive model browser TUI."""

    TITLE = "dx-modelzoo"

    CSS = """
    #main-area {
        height: 1fr;
    }
    #tree-panel {
        width: 2fr;
        height: 100%;
        overflow-y: auto;
    }
    #search-bar {
        dock: top;
        display: none;
        height: 3;
        padding: 0 1;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("slash", "start_search", "Search", show=True),
        Binding("right", "expand_node", "Expand", show=False),
        Binding("left", "collapse_node", "Collapse", show=False),
        Binding("e", "eval", "Eval", show=False),
        Binding("c", "compile", "Compile", show=False),
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("1", "profile_1", "Profile 1", show=False),
        Binding("2", "profile_2", "Profile 2", show=False),
        Binding("3", "profile_3", "Profile 3", show=False),
        Binding("4", "profile_4", "Profile 4", show=False),
    ]

    def __init__(
        self,
        entries: List[ModelEntry],
        domain: Optional[str] = None,
        task: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.entries = entries
        self.filter_domain = domain
        self.filter_task = task
        self._entry_map: Dict[str, ModelEntry] = {}
        self._selected_entry: Optional[ModelEntry] = None
        self._action_mode: Optional[str] = None
        self._profiles: list[str] = []
        self._searching: bool = False
        self._filtered: bool = False  # True when showing search results

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Input(placeholder="🔍 Search model name... (Esc to cancel)", id="search-bar")
        with Horizontal(id="main-area"):
            yield Tree("📂 Models", id="tree-panel")
            yield ModelInfoPanel(id="info-panel")
        yield ActionPanel(id="action-panel")
        yield ProfileSelectPanel(id="profile-panel")
        yield Footer()

    def on_mount(self) -> None:
        tree: Tree = self.query_one("#tree-panel", Tree)
        tree.root.expand()
        self._build_tree(tree)
        tree.focus()
        self.query_one("#info-panel", ModelInfoPanel).show_empty()

    def on_key(self, event: events.Key) -> None:
        if event.character and ord(event.character) > 127:
            self.notify(
                "⚠️ Please switch to English keyboard",
                severity="warning",
                timeout=2,
            )

    def _build_tree(self, tree: Tree) -> None:
        # Group: domain > task > family > model
        structure: Dict[str, Dict[str, Dict[str, List[ModelEntry]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )

        for entry in self.entries:
            rel = entry.yaml_path.relative_to(entry.yaml_path.parents[3])
            family = rel.parts[2] if len(rel.parts) >= 4 else "other"
            structure[entry.domain][entry.task][family].append(entry)

        for domain_name in sorted(structure):
            domain_node = tree.root.add(
                f"[bold blue]📁 {domain_name}[/]",
                expand=False,
            )
            for task_name in sorted(structure[domain_name]):
                task_entries = structure[domain_name][task_name]
                total = sum(len(v) for v in task_entries.values())
                task_node = domain_node.add(
                    f"[bold green]📂 {task_name}[/] [dim]({total})[/]",
                    expand=False,
                )
                for family_name in sorted(task_entries):
                    models = task_entries[family_name]
                    if len(models) == 1:
                        # Single model — add directly under task
                        entry = models[0]
                        node_id = f"{domain_name}/{task_name}/{entry.name}"
                        self._entry_map[node_id] = entry
                        task_node.add_leaf(
                            f"[white]🔹 {entry.name}[/]",
                            data=node_id,
                        )
                    else:
                        # Family folder
                        family_node = task_node.add(
                            f"[yellow]📦 {family_name}[/] [dim]({len(models)})[/]",
                            expand=False,
                        )
                        for entry in sorted(models, key=lambda e: e.name):
                            node_id = f"{domain_name}/{task_name}/{entry.name}"
                            self._entry_map[node_id] = entry
                            family_node.add_leaf(
                                f"[white]🔹 {entry.name}[/]",
                                data=node_id,
                            )

    @on(Tree.NodeHighlighted)
    def on_node_highlighted(self, event: Tree.NodeHighlighted) -> None:
        node = event.node
        # Reset sub-menus on any navigation
        self._action_mode = None
        self.query_one("#profile-panel", ProfileSelectPanel).hide()

        if node.data and node.data in self._entry_map:
            entry = self._entry_map[node.data]
            self._selected_entry = entry
            self.query_one("#info-panel", ModelInfoPanel).update_model(entry)
            # Auto-show action panel for model nodes
            try:
                builder = ModelBuilder(entry.yaml_path)
                self._profiles = list(builder.config.get("profiles", {}).keys())
            except Exception:
                self._profiles = []
            self.query_one("#action-panel", ActionPanel).show_actions(entry.name, self._profiles)
        else:
            self._selected_entry = None
            self.query_one("#action-panel", ActionPanel).hide()

    @on(Tree.NodeSelected)
    def on_node_selected(self, event: Tree.NodeSelected) -> None:
        pass  # Actions handled via highlight + keybindings

    def action_eval(self) -> None:
        if self._selected_entry and self._profiles:
            self._action_mode = "eval"
            self.query_one("#action-panel", ActionPanel).hide()
            self.query_one("#profile-panel", ProfileSelectPanel).show_profiles("eval", self._profiles)

    def action_compile(self) -> None:
        if self._selected_entry and self._profiles:
            compile_profiles = [p for p in self._profiles if p != "onnx"]
            if not compile_profiles:
                self.notify("No compile-compatible profiles (non-onnx)", severity="warning")
                return
            self._action_mode = "compile"
            self._compile_profiles = compile_profiles
            self.query_one("#action-panel", ActionPanel).hide()
            self.query_one("#profile-panel", ProfileSelectPanel).show_profiles("compile", compile_profiles)

    def _execute_action(self, profile_idx: int) -> None:
        if not self._selected_entry or not self._action_mode:
            return

        if self._action_mode == "compile":
            profiles = getattr(self, "_compile_profiles", self._profiles)
        else:
            profiles = self._profiles

        if profile_idx < 0 or profile_idx >= len(profiles):
            return

        profile = profiles[profile_idx]
        model_name = self._selected_entry.name
        action = self._action_mode

        self.exit(result={"action": action, "model": model_name, "profile": profile})

    def action_profile_1(self) -> None:
        if self._action_mode:
            self._execute_action(0)

    def action_profile_2(self) -> None:
        if self._action_mode:
            self._execute_action(1)

    def action_profile_3(self) -> None:
        if self._action_mode:
            self._execute_action(2)

    def action_profile_4(self) -> None:
        if self._action_mode:
            self._execute_action(3)

    def action_cancel(self) -> None:
        if self._searching:
            # In search → close search, restore full tree
            self._close_search(restore=True)
        elif self._action_mode:
            # In profile select → back to action panel
            self._action_mode = None
            self.query_one("#profile-panel", ProfileSelectPanel).hide()
            if self._selected_entry:
                self.query_one("#action-panel", ActionPanel).show_actions(self._selected_entry.name, self._profiles)
        elif self._filtered:
            # Viewing search results → restore full tree
            self._filtered = False
            self._restore_tree()
        else:
            # In action panel → dismiss all
            self.query_one("#action-panel", ActionPanel).hide()
            self.query_one("#profile-panel", ProfileSelectPanel).hide()

    def action_expand_node(self) -> None:
        tree: Tree = self.query_one("#tree-panel", Tree)
        node = tree.cursor_node
        if node and not node.is_expanded and node.children:
            node.expand()

    def action_collapse_node(self) -> None:
        tree: Tree = self.query_one("#tree-panel", Tree)
        node = tree.cursor_node
        if node and node.is_expanded:
            node.collapse()
        elif node and node.parent and node.parent != tree.root:
            tree.select_node(node.parent)
            node.parent.collapse()

    # --- Search ---

    def action_start_search(self) -> None:
        """Activate search bar."""
        if self._action_mode:
            return
        self._searching = True
        # Save current expand state before filtering
        self._save_expand_state()
        search_bar = self.query_one("#search-bar", Input)
        search_bar.display = True
        search_bar.value = ""
        search_bar.focus()

    def _close_search(self, restore: bool = True) -> None:
        """Deactivate search bar and optionally restore full tree."""
        self._searching = False
        self._filtered = False
        search_bar = self.query_one("#search-bar", Input)
        search_bar.display = False
        search_bar.value = ""
        if restore:
            self._restore_tree()
        tree: Tree = self.query_one("#tree-panel", Tree)
        tree.focus()

    def _save_expand_state(self) -> None:
        """Save expand/collapse state of all tree nodes."""
        tree: Tree = self.query_one("#tree-panel", Tree)
        self._saved_expand: Dict[str, bool] = {}
        self._walk_save(tree.root)

    def _walk_save(self, node: TreeNode) -> None:
        label = str(node.label)
        self._saved_expand[label] = node.is_expanded
        for child in node.children:
            self._walk_save(child)

    def _restore_tree(self) -> None:
        """Rebuild full tree and restore saved expand state."""
        tree: Tree = self.query_one("#tree-panel", Tree)
        tree.clear()
        self._entry_map.clear()
        self._build_tree(tree)
        # Restore expand state
        saved = getattr(self, "_saved_expand", {})
        if saved:
            self._walk_restore(tree.root, saved)

    def _walk_restore(self, node: TreeNode, saved: Dict[str, bool]) -> None:
        label = str(node.label)
        if label in saved:
            if saved[label]:
                node.expand()
            else:
                node.collapse()
        for child in node.children:
            self._walk_restore(child, saved)

    def _filter_tree(self, query: str) -> None:
        """Rebuild tree showing only models matching query (all expanded)."""
        tree: Tree = self.query_one("#tree-panel", Tree)
        tree.clear()
        # Use a temporary entry map for filtered view; keep original entries intact
        self._entry_map.clear()

        query_lower = query.lower()
        matched = [e for e in self.entries if query_lower in e.name.lower()]

        if not matched:
            tree.root.add_leaf(f"[dim]No results for '{query}'[/]")
            return

        structure: Dict[str, Dict[str, Dict[str, List[ModelEntry]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        for entry in matched:
            rel = entry.yaml_path.relative_to(entry.yaml_path.parents[3])
            family = rel.parts[2] if len(rel.parts) >= 4 else "other"
            structure[entry.domain][entry.task][family].append(entry)

        for domain_name in sorted(structure):
            domain_node = tree.root.add(
                f"[bold blue]📁 {domain_name}[/]",
                expand=True,
            )
            for task_name in sorted(structure[domain_name]):
                task_entries = structure[domain_name][task_name]
                total = sum(len(v) for v in task_entries.values())
                task_node = domain_node.add(
                    f"[bold green]📂 {task_name}[/] [dim]({total})[/]",
                    expand=True,
                )
                for family_name in sorted(task_entries):
                    models = task_entries[family_name]
                    if len(models) == 1:
                        entry = models[0]
                        node_id = f"{domain_name}/{task_name}/{entry.name}"
                        self._entry_map[node_id] = entry
                        task_node.add_leaf(
                            f"[white]🔹 {entry.name}[/]",
                            data=node_id,
                        )
                    else:
                        family_node = task_node.add(
                            f"[yellow]📦 {family_name}[/] [dim]({len(models)})[/]",
                            expand=True,
                        )
                        for entry in sorted(models, key=lambda e: e.name):
                            node_id = f"{domain_name}/{task_name}/{entry.name}"
                            self._entry_map[node_id] = entry
                            family_node.add_leaf(
                                f"[white]🔹 {entry.name}[/]",
                                data=node_id,
                            )

    @on(Input.Changed, "#search-bar")
    def on_search_changed(self, event: Input.Changed) -> None:
        """Live filter as user types."""
        query = event.value.strip()
        if query:
            self._filter_tree(query)
        else:
            self._restore_tree()

    @on(Input.Submitted, "#search-bar")
    def on_search_submitted(self, event: Input.Submitted) -> None:
        """Confirm search: hide search bar, keep filtered tree, focus first result."""
        self._searching = False
        self._filtered = bool(event.value.strip())
        search_bar = self.query_one("#search-bar", Input)
        search_bar.display = False
        tree: Tree = self.query_one("#tree-panel", Tree)
        tree.focus()
        self._select_first_leaf(tree.root)

    def _select_first_leaf(self, node: TreeNode) -> bool:
        """Select the first leaf node in the tree. Returns True if found."""
        if not node.children:
            if node.data:
                tree: Tree = self.query_one("#tree-panel", Tree)
                tree.select_node(node)
                return True
            return False
        for child in node.children:
            if self._select_first_leaf(child):
                return True
        return False


def run_interactive_list(
    models_dir: Path,
    domain: Optional[str] = None,
    task: Optional[str] = None,
) -> Optional[dict]:
    """Launch the interactive model browser. Returns action dict or None."""
    entries = discover_models(models_dir, domain=domain, task=task)
    if not entries:
        return None
    app = ModelTreeBrowser(entries, domain=domain, task=task)
    return app.run()
