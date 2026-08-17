#!/usr/bin/env python3
"""
run_colab_m2.py
Playwright automation: upload colab_m2_pretrain.ipynb to Google Colab,
connect T4 GPU runtime, and run all cells.

Uses the user's existing Chrome profile so Google sign-in is preserved.
"""
import os, sys, time, pathlib
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

NOTEBOOK = str(pathlib.Path(__file__).parent / "colab_m2_pretrain.ipynb")
# Chrome user-data dir — reuse existing profile so Google login is intact
CHROME_USER_DATA = os.path.expandvars(
    r"%LOCALAPPDATA%\Google\Chrome\User Data"
)

def wait_and_click(page, selector, timeout=30_000, desc=""):
    print(f"  -> waiting for: {desc or selector}")
    page.wait_for_selector(selector, timeout=timeout)
    page.click(selector)

def run():
    print(f"Notebook: {NOTEBOOK}")
    assert os.path.exists(NOTEBOOK), f"Notebook not found: {NOTEBOOK}"

    with sync_playwright() as pw:
        # Use a dedicated Playwright profile under AppData
        # (avoids conflict with the running Chrome instance)
        pw_profile = os.path.expandvars(
            r"%LOCALAPPDATA%\Playwright\colab_profile"
        )
        os.makedirs(pw_profile, exist_ok=True)
        print(f"Playwright profile: {pw_profile}")

        browser = pw.chromium.launch_persistent_context(
            user_data_dir=pw_profile,
            channel="chrome",
            headless=False,
            args=["--start-maximized", "--disable-sync"],
            no_viewport=True,
        )

        page = browser.new_page()

        # ── Step 1: open Colab ───────────────────────────────────────────────
        print("\n[1] Navigating to colab.research.google.com ...")
        page.goto("https://colab.research.google.com/", timeout=60_000)
        page.wait_for_load_state("networkidle", timeout=60_000)
        time.sleep(2)

        # ── Step 2: upload notebook via File > Upload notebook ───────────────
        print("[2] Uploading notebook ...")

        # Try clicking File menu
        try:
            # Colab has a "File" menu in the toolbar
            page.click("text=File", timeout=10_000)
            time.sleep(0.5)
            page.click("text=Upload notebook", timeout=10_000)
            time.sleep(1)
        except PWTimeout:
            print("  File menu not found, trying direct upload dialog ...")

        # Handle file chooser dialog
        print("  Waiting for file chooser ...")
        with page.expect_file_chooser(timeout=15_000) as fc_info:
            # If dialog already open, just set files; otherwise trigger it
            try:
                page.click("text=Upload notebook", timeout=5_000)
            except Exception:
                pass
        fc_info.value.set_files(NOTEBOOK)
        print(f"  Uploaded: {NOTEBOOK}")
        time.sleep(3)

        page.wait_for_load_state("networkidle", timeout=60_000)
        print(f"  URL: {page.url}")

        # ── Step 3: set runtime to T4 GPU ───────────────────────────────────
        print("[3] Setting runtime to T4 GPU ...")
        try:
            # Runtime > Change runtime type
            page.click("text=Runtime", timeout=15_000)
            time.sleep(0.5)
            page.click("text=Change runtime type", timeout=10_000)
            time.sleep(1)

            # Select T4 GPU
            # The dialog has a Hardware accelerator dropdown
            page.wait_for_selector("text=Hardware accelerator", timeout=15_000)
            # Click the dropdown
            try:
                page.select_option("select", "GPU", timeout=5_000)
            except Exception:
                # Try clicking the dropdown and selecting T4
                page.click("[aria-label*='Hardware accelerator']", timeout=5_000)
                time.sleep(0.5)
                page.click("text=T4 GPU", timeout=5_000)

            # Save
            page.click("text=Save", timeout=10_000)
            print("  Runtime set to T4 GPU")
            time.sleep(2)
        except Exception as e:
            print(f"  Runtime dialog issue: {e} -- continuing anyway")

        # ── Step 4: Run All ──────────────────────────────────────────────────
        print("[4] Running all cells (Runtime > Run all) ...")
        try:
            page.click("text=Runtime", timeout=15_000)
            time.sleep(0.5)
            page.click("text=Run all", timeout=10_000)
            time.sleep(1)
            # Confirm if there's a "Run anyway" dialog
            try:
                page.click("text=Run anyway", timeout=5_000)
                print("  Confirmed 'Run anyway'")
            except Exception:
                pass
            print("  All cells started!")
        except Exception as e:
            print(f"  Could not trigger Run All: {e}")

        # ── Step 5: Monitor ──────────────────────────────────────────────────
        print("\n[5] Training in progress. Browser will stay open.")
        print("    Check the Colab tab -- training takes ~7h on T4.")
        print("    serve.pt will be saved to Google Drive when done.")
        print("\nPress Ctrl+C to stop monitoring (browser stays open).\n")

        # Keep script alive, poll every 60s for status
        try:
            while True:
                time.sleep(60)
                try:
                    # Look for error indicators
                    content = page.content()
                    if "Runtime disconnected" in content:
                        print("[!] Runtime disconnected -- check Colab")
                    elif "error" in content.lower() and "cell" in content.lower():
                        print("[!] Possible cell error -- check Colab")
                    else:
                        print(f"[ok] {time.strftime('%H:%M:%S')} training running ...")
                except Exception:
                    pass
        except KeyboardInterrupt:
            print("\nMonitoring stopped. Browser stays open.")

        browser.close()

if __name__ == "__main__":
    run()
