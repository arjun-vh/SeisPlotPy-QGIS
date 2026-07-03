import os
import markdown
from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QSplitter, QListWidget, QTextBrowser, 
    QListWidgetItem, QMessageBox
)
from qgis.PyQt.QtCore import Qt, QUrl
from qgis.PyQt.QtGui import QDesktopServices

# Define the order and titles of the documentation pages
DOC_PAGES = [
    ("index.md", "Home"),
    ("getting_started.md", "Getting Started"),
    ("loading.md", "Loading Data"),
    ("project_persistence.md", "Project Persistence"),
    ("display.md", "Display Controls"),
    ("navigation.md", "Navigation & Map Link"),
    ("interpretation.md", "Interpretation"),
    ("processing.md", "Processing & Attributes"),
    ("headers.md", "Header Utilities"),
    ("exporting.md", "Exporting"),
    ("batch_loading.md", "Batch Loading"),
    ("reference.md", "Quick Reference")
]

# Custom CSS for the QTextBrowser to look like a modern e-book/GitHub doc
DOC_CSS = """
<style>
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
        font-size: 14px;
        line-height: 1.6;
        color: #24292e;
        background-color: #ffffff;
        padding: 20px;
    }
    h1, h2, h3, h4 {
        margin-top: 24px;
        margin-bottom: 16px;
        font-weight: 600;
        line-height: 1.25;
        border-bottom: 1px solid #eaecef;
        padding-bottom: 0.3em;
    }
    h1 { font-size: 2em; }
    h2 { font-size: 1.5em; }
    h3 { font-size: 1.25em; border-bottom: none; }
    p { margin-top: 0; margin-bottom: 16px; }
    a { color: #0366d6; text-decoration: none; }
    a:hover { text-decoration: underline; }
    code {
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
        padding: 0.2em 0.4em;
        margin: 0;
        font-size: 85%;
        background-color: rgba(27,31,35,0.05);
        border-radius: 3px;
    }
    pre {
        padding: 16px;
        overflow: auto;
        font-size: 85%;
        line-height: 1.45;
        background-color: #f6f8fa;
        border-radius: 3px;
    }
    pre code {
        background-color: transparent;
        padding: 0;
    }
    blockquote {
        padding: 0 1em;
        color: #6a737d;
        border-left: 0.25em solid #dfe2e5;
        margin: 0 0 16px 0;
    }
    table {
        border-spacing: 0;
        border-collapse: collapse;
        margin-bottom: 16px;
        width: 100%;
    }
    table th, table td {
        padding: 6px 13px;
        border: 1px solid #dfe2e5;
    }
    table tr { background-color: #fff; border-top: 1px solid #c6cbd1; }
    table tr:nth-child(2n) { background-color: #f6f8fa; }
</style>
"""

class HelpViewer(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SeisPlotPy Documentation")
        self.resize(1000, 700)
        
        # Determine the docs directory path (relative to this file)
        plugin_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.docs_dir = os.path.join(plugin_dir, "docs")
        
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        self.splitter = QSplitter(Qt.Horizontal)
        
        # Table of Contents (Left Pane)
        self.toc_list = QListWidget()
        self.toc_list.setMinimumWidth(200)
        self.toc_list.setMaximumWidth(300)
        
        # Content Browser (Right Pane)
        self.content_browser = QTextBrowser()
        self.content_browser.setOpenLinks(False) # We will handle links manually
        self.content_browser.anchorClicked.connect(self.handle_link_clicked)
        
        self.splitter.addWidget(self.toc_list)
        self.splitter.addWidget(self.content_browser)
        self.splitter.setSizes([250, 750])
        
        layout.addWidget(self.splitter)
        
        # Populate TOC
        self.populate_toc()
        
        # Connect signal
        self.toc_list.itemSelectionChanged.connect(self.load_selected_page)
        
        # Select first item
        if self.toc_list.count() > 0:
            self.toc_list.setCurrentRow(0)
            
    def populate_toc(self):
        """Populates the QListWidget with documentation topics."""
        for filename, title in DOC_PAGES:
            item = QListWidgetItem(title)
            item.setData(Qt.UserRole, filename)
            self.toc_list.addItem(item)
            
    def load_selected_page(self):
        """Reads the selected Markdown file, converts it, and displays it."""
        items = self.toc_list.selectedItems()
        if not items:
            return
            
        filename = items[0].data(Qt.UserRole)
        self.load_page(filename)
        
    def load_page(self, filename, anchor=None):
        """Loads a specific markdown file and optionally scrolls to an anchor."""
        filepath = os.path.join(self.docs_dir, filename)
        
        if not os.path.exists(filepath):
            self.content_browser.setHtml(f"<h2>Error</h2><p>File not found: {filename}</p>")
            return
            
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                md_text = f.read()
                
            # Convert Markdown to HTML
            # 'tables' extension for markdown tables
            # 'fenced_code' for code blocks
            html_content = markdown.markdown(md_text, extensions=['tables', 'fenced_code'])
            
            # Inject CSS
            full_html = f"<html><head>{DOC_CSS}</head><body>{html_content}</body></html>"
            
            self.content_browser.setHtml(full_html)
            
            if anchor:
                # Basic support for scrolling to anchors if needed.
                # In QTextBrowser, setting the source URL with an anchor sometimes handles it.
                pass 
                
        except Exception as e:
            self.content_browser.setHtml(f"<h2>Error</h2><p>Failed to load {filename}: {str(e)}</p>")

    def handle_link_clicked(self, url: QUrl):
        """Intercepts link clicks to handle internal routing or open external links."""
        url_str = url.toString()
        
        if url_str.startswith("http://") or url_str.startswith("https://"):
            # Open external links in default web browser
            QDesktopServices.openUrl(url)
        else:
            # Handle internal links (e.g., "loading.md" or "loading.md#section")
            filename = url_str
            anchor = None
            if "#" in url_str:
                filename, anchor = url_str.split("#", 1)
                
            # Find the file in our TOC and select it
            for i in range(self.toc_list.count()):
                item = self.toc_list.item(i)
                if item.data(Qt.UserRole) == filename:
                    self.toc_list.setCurrentItem(item)
                    # The selectionChanged signal will trigger load_selected_page
                    break
