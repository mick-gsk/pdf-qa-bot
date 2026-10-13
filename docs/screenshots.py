"""Regenerate the images in docs/: screenshot.png, value-check.png and demo.gif.

    pip install playwright && playwright install chromium
    python docs/screenshots.py

Each image runs the app on a free port with an empty, temporary DATA_DIR and the two example
PDFs. screenshot.png and demo.gif use the extractive mode, so every answer in them is real
output of this repository. value-check.png uses a scripted model whose answer contains one wrong
amount, with GROUNDING=flag, to show how the check marks it. The GIF needs ffmpeg.
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import uvicorn
from playwright.sync_api import Page, sync_playwright

from pdfqa.api import create_app
from pdfqa.config import Settings

DOCS = Path(__file__).parent
EXAMPLES = DOCS.parent / "examples"
PDFS = [EXAMPLES / "employee-handbook.pdf", EXAMPLES / "espresso-manual.pdf"]
DESKTOP = {"viewport": {"width": 1280, "height": 800}, "device_scale_factor": 2}


class ScriptedLLM:
    """Answers every question with the same text: a correct amount and a wrong one."""

    name = "scripted demo model"
    text = ("Hotel costs are reimbursed up to 120 euros per night [1]. In Paris the limit is "
            "150 euros per night [1].")

    def complete(self, system: str, prompt: str) -> str:
        return self.text

    def stream(self, system: str, prompt: str) -> Iterator[str]:
        for word in self.text.split(" "):
            yield word + " "


@contextmanager
def server(settings: Settings, llm: ScriptedLLM | None = None) -> Iterator[str]:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with mock.patch("pdfqa.api.get_llm", return_value=llm):
        app = create_app(settings)
    srv = uvicorn.Server(uvicorn.Config(app, port=port, log_level="warning"))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    while not srv.started:
        time.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{port}/"
    finally:
        srv.should_exit = True
        thread.join()


def upload(page: Page) -> None:
    page.set_input_files("#file", [str(p) for p in PDFS])
    page.wait_for_selector(".file >> nth=1")


def ask(page: Page, question: str, typing_delay: float = 0) -> None:
    page.click("#question")
    page.keyboard.type(question, delay=typing_delay)
    page.keyboard.press("Enter")
    page.wait_for_function("!document.querySelector('.pending') && !document.querySelector('#send').disabled")


def hover_value(page: Page, selector: str = "mark") -> None:
    page.hover(f"{selector} >> nth=0")
    page.wait_for_selector("#tip:not([hidden])")


def screenshot(browser, data_dir: Path) -> None:
    with server(Settings(data_dir=data_dir)) as url:
        page = browser.new_page(**DESKTOP)
        page.goto(url)
        upload(page)
        ask(page, "How many vacation days do full-time employees get?")
        ask(page, "What is the hotel limit in Paris?")
        page.click(".turn:last-child details.src summary >> nth=0")
        page.wait_for_timeout(400)
        page.screenshot(path=DOCS / "screenshot.png")
        page.close()


def value_check(browser, data_dir: Path) -> None:
    settings = Settings(data_dir=data_dir, grounding="flag")
    with server(settings, ScriptedLLM()) as url:
        page = browser.new_page(**DESKTOP)
        page.goto(url)
        upload(page)
        ask(page, "What is the hotel limit in Paris?")
        page.click("details.src summary >> nth=0")
        hover_value(page, "mark.unsupported")
        page.screenshot(path=DOCS / "value-check.png")
        page.close()


def demo(browser, data_dir: Path) -> None:
    video_dir = Path(tempfile.mkdtemp())
    with server(Settings(data_dir=data_dir)) as url:
        context = browser.new_context(viewport={"width": 1280, "height": 760}, record_video_dir=video_dir,
                                      record_video_size={"width": 1280, "height": 760})
        page = context.new_page()
        page.goto(url)
        page.wait_for_timeout(1200)
        upload(page)
        page.wait_for_timeout(1500)
        page.click(".example >> nth=1")
        page.wait_for_function("!document.querySelector('.pending')")
        page.wait_for_timeout(800)
        hover_value(page)
        page.wait_for_timeout(2200)
        page.mouse.move(640, 700)
        ask(page, "What does error E03 mean?", typing_delay=45)
        page.wait_for_timeout(1500)
        ask(page, "Is there a company car policy?", typing_delay=45)
        page.wait_for_timeout(2500)
        video = page.video.path()
        context.close()
    palette = video_dir / "palette.png"
    scale = "fps=10,scale=960:-1:flags=lanczos"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-vf", f"{scale},palettegen=stats_mode=diff",
                    palette], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-i", palette, "-lavfi",
                    f"{scale}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle",
                    DOCS / "demo.gif"], check=True)
    shutil.rmtree(video_dir)


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for step in (screenshot, value_check, demo):
            with tempfile.TemporaryDirectory() as data_dir:
                step(browser, Path(data_dir))
            print(f"{step.__name__}: done")
        browser.close()


if __name__ == "__main__":
    main()
