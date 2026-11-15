"""Render Mermaid .mmd files to PNG for the Word report.

Requires Playwright, Chromium and network access for the pinned Mermaid library.
Install Playwright separately, then run `playwright install chromium`.
`CHROMIUM_EXECUTABLE_PATH` can select an existing Chromium installation.
"""
from pathlib import Path
import os

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent
DIAGRAMS = ("overview", "experiment_pipeline", "smo_pair_update", "architecture")


def main():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=os.environ.get("CHROMIUM_EXECUTABLE_PATH"))
        page = browser.new_page(viewport={"width": 2000, "height": 1100}, device_scale_factor=2)
        page.set_content('<!doctype html><meta charset="utf-8"><div id="figure" style="display:inline-block;background:white;padding:20px"></div>')
        page.evaluate("""async () => {
          window.mm = (await import('https://cdn.jsdelivr.net/npm/mermaid@11.17.2/dist/mermaid.esm.min.mjs')).default;
          window.mm.initialize({startOnLoad: false, theme: 'base',
            themeVariables: {fontFamily: 'Arial, sans-serif', fontSize: '18px',
              primaryColor: '#f4f6f8', primaryBorderColor: '#84929d',
              primaryTextColor: '#1e2933', lineColor: '#71818d',
              clusterBkg: '#ffffff', clusterBorder: '#cad3d9'},
            flowchart: {curve: 'linear', htmlLabels: true, useMaxWidth: false}});
          document.body.style = 'margin:0;background:white';
        }""")
        for name in DIAGRAMS:
            source = (ROOT / f"{name}.mmd").read_text(encoding="utf-8")
            page.evaluate("""async ([name, source]) => {
              const {svg} = await window.mm.render('chart-' + name, source);
              document.getElementById('figure').innerHTML = svg;
            }""", [name, source])
            page.locator("#figure").screenshot(path=str(ROOT / f"{name}.png"))
            print(f"Rendered {name}.png")
        browser.close()


if __name__ == "__main__":
    main()
