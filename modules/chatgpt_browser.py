"""ChatGPT Web Browser Automation via Playwright for Cinematic Drama Dubbing."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Sequence

from . import (
    PipelineError,
    ensure_dir,
    log,
    read_json,
    write_json,
)

CHATGPT_URL = "https://chatgpt.com"


class ChatGPTBrowserController:
    """Automates ChatGPT Web UI via Playwright with persistent sessions and conversational memory."""

    def __init__(
        self,
        profile_dir: str = "storage/browser_profile",
        cdp_url: str = "http://localhost:9222",
        headless: bool = False,
        timeout_sec: int = 180,
    ):
        self.profile_dir = ensure_dir(profile_dir)
        self.cdp_url = cdp_url
        self.headless = headless
        self.timeout_sec = timeout_sec
        self.playwright = None
        self.browser_context = None
        self.page = None

    def start(self):
        """Connect via CDP or launch persistent Chrome browser context."""
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise PipelineError("Playwright is not installed. Run: pip install playwright") from exc

        self.playwright = sync_playwright().start()

        # 1. Try connecting to an already running Chrome with remote debugging port
        try:
            log.info("Checking for running Chrome on %s...", self.cdp_url)
            self.browser_context = self.playwright.chromium.connect_over_cdp(self.cdp_url)
            self.page = self.browser_context.pages[0] if self.browser_context.pages else self.browser_context.new_page()
            log.info("Connected to active Chrome via CDP!")
        except Exception:
            log.info("CDP port not active. Launching dedicated Chrome session (%s)...", self.profile_dir)
            self.browser_context = self.playwright.chromium.launch_persistent_context(
                user_data_dir=self.profile_dir,
                channel="chrome",
                headless=self.headless,
                args=["--no-sandbox", "--disable-blink-features=AutomationControlled"],
            )
            self.page = self.browser_context.pages[0] if self.browser_context.pages else self.browser_context.new_page()

        # Navigate to ChatGPT
        current_url = self.page.url
        if "chatgpt.com" not in current_url:
            log.info("Navigating to %s...", CHATGPT_URL)
            self.page.goto(CHATGPT_URL, wait_until="commit", timeout=60000)

        self._ensure_logged_in()

    def _ensure_logged_in(self):
        """Verify prompt input area is ready."""
        log.info("Verifying ChatGPT session readiness...")
        for attempt in range(self.timeout_sec):
            input_box = self.page.query_selector(
                "#prompt-textarea, #mobile-composer-prompt, textarea, div[contenteditable='true']"
            )
            if input_box:
                log.info("ChatGPT session is ready!")
                return

            # Check if login button is present
            login_btn = self.page.query_selector("button[data-testid='login-button'], a[href*='login']")
            if login_btn and attempt % 15 == 0:
                log.info("ChatGPT prompt ready (guest or persistent session).")

            time.sleep(1)

        raise PipelineError("Timeout waiting for ChatGPT web session to become ready.")

    def _send_message_and_wait(self, text: str) -> str:
        """Send message into ChatGPT prompt textarea and wait for full generation."""
        prev_assistants = len(self.page.query_selector_all("li[data-message-role='assistant'], div[data-message-author-role='assistant']"))

        # Find input box
        textarea = self.page.wait_for_selector(
            "#prompt-textarea, #mobile-composer-prompt, textarea, div[contenteditable='true']",
            timeout=30000,
        )
        if not textarea:
            raise PipelineError("Could not locate ChatGPT prompt input field.")

        # Focus and fill text
        textarea.click()
        time.sleep(0.2)
        textarea.fill(text)
        time.sleep(0.5)

        # Send via submit button or Enter
        send_btn = self.page.query_selector(
            "button[data-testid='send-button'], button[aria-label*='Send'], button[aria-label*='prompt']"
        )
        if send_btn and send_btn.is_enabled():
            send_btn.click()
        else:
            self.page.keyboard.press("Enter")

        log.info("Sent prompt to ChatGPT. Waiting for generation to stream...")

        # Wait for new assistant item to appear
        start_t = time.time()
        while time.time() - start_t < 30:
            curr_assistants = len(self.page.query_selector_all("li[data-message-role='assistant'], div[data-message-author-role='assistant']"))
            if curr_assistants > prev_assistants:
                break
            time.sleep(0.5)

        # Stream until output is stable
        last_text = ""
        stable_ticks = 0
        while time.time() - start_t < self.timeout_sec:
            # Dismiss any popup/modal (e.g. login reminder)
            try:
                for sel in ["button[aria-label='Close']", "button:has-text('Stay logged out')", "button:has-text('Dismiss')"]:
                    m_btn = self.page.query_selector(sel)
                    if m_btn and m_btn.is_visible():
                        m_btn.click()
                        time.sleep(0.3)
            except Exception:
                pass

            # Check if EXACT 'Continue generating' button appeared
            continue_btn = self.page.query_selector("button[data-testid='continue-generating-button']")
            if not continue_btn:
                for btn in self.page.query_selector_all("button"):
                    try:
                        if btn.inner_text().strip().lower() == "continue generating":
                            continue_btn = btn
                            break
                    except Exception:
                        pass

            if continue_btn and continue_btn.is_visible():
                log.info("Detected 'Continue generating' button. Clicking automatically...")
                continue_btn.click()
                time.sleep(2.0)
                continue

            stop_btn = self.page.query_selector(
                "button[data-testid='stop-button'], button[aria-label*='Stop']"
            )

            assistants = self.page.query_selector_all("li[data-message-role='assistant'], div[data-message-author-role='assistant']")
            if assistants:
                curr_text = assistants[-1].inner_text()
                if curr_text and curr_text == last_text:
                    if not (stop_btn and stop_btn.is_visible()):
                        stable_ticks += 1
                        if stable_ticks >= 3:
                            clean = curr_text.strip()
                            if clean.startswith("ChatGPT said:"):
                                clean = clean[len("ChatGPT said:"):].strip()
                            return clean
                else:
                    stable_ticks = 0
                    last_text = curr_text
            time.sleep(0.8)

        if last_text:
            clean = last_text.strip()
            if clean.startswith("ChatGPT said:"):
                clean = clean[len("ChatGPT said:"):].strip()
            return clean
        raise PipelineError("Timeout waiting for ChatGPT response to complete.")

    def close(self):
        """Cleanly disconnect without destroying user cookies."""
        try:
            if self.browser_context:
                self.browser_context.close()
            if self.playwright:
                self.playwright.stop()
        except Exception:
            pass


def _parse_markdown_table(table_text: str) -> list[dict[str, Any]]:
    """Parse Markdown or tab-separated table into list of dialogue dictionaries."""
    rows: list[dict[str, Any]] = []
    raw_lines = [line.strip() for line in table_text.splitlines() if line.strip()]

    # Determine delimiter
    pipe_count = sum(1 for l in raw_lines if "|" in l)
    tab_count = sum(1 for l in raw_lines if "\t" in l)
    is_tab = tab_count > pipe_count

    lines = [l for l in raw_lines if (("\t" in l) if is_tab else ("|" in l))]

    header_idx = -1
    for i, line in enumerate(lines):
        u = line.upper()
        if "ID" in u and ("TRANSLATION" in u or "DIALOGUE" in u or "SPEAKER" in u or "ROLE" in u):
            header_idx = i
            break

    if header_idx == -1:
        header_idx = 0

    for line in lines[header_idx + 1 :]:
        if "---" in line:
            continue
        if is_tab:
            parts = [p.strip() for p in line.split("\t") if p.strip()]
        else:
            parts = [p.strip() for p in line.strip("|").split("|")]

        if len(parts) >= 3:
            try:
                # Expecting: ID | SPEAKER | GENDER | EMOTION | DIALOGUE
                seg_id = int(re.sub(r"[^\d]", "", parts[0]))
                raw_spk = parts[1].strip().lower() if len(parts) >= 2 else "extra_male"
                spk_clean = re.sub(r"[^a-z_]", "", raw_spk)

                if "heroine" in spk_clean:
                    speaker = "heroine"
                elif "hero" in spk_clean:
                    speaker = "hero"
                elif "villain" in spk_clean:
                    speaker = "villain"
                elif "father" in spk_clean:
                    speaker = "father"
                elif "mother" in spk_clean:
                    speaker = "mother"
                elif "maid" in spk_clean:
                    speaker = "maid"
                elif "guard" in spk_clean or "soldier" in spk_clean or "general" in spk_clean:
                    speaker = "guard"
                elif "king" in spk_clean or "emperor" in spk_clean:
                    speaker = "king"
                elif "female" in spk_clean or "woman" in spk_clean:
                    speaker = "extra_female"
                elif "male" in spk_clean or "man" in spk_clean:
                    speaker = "extra_male"
                else:
                    speaker = spk_clean or "extra_male"

                raw_gender = parts[2].strip().lower() if len(parts) >= 3 else ""
                if speaker in ["heroine", "mother", "maid", "extra_female"]:
                    gender = "female"
                elif speaker in ["hero", "villain", "father", "guard", "king", "extra_male"]:
                    gender = "male"
                else:
                    gender = "female" if ("female" in raw_gender or "स्त्री" in raw_gender) else "male"

                emotion = parts[3].strip().lower() if len(parts) >= 4 else "neutral"
                text = parts[4].strip() if len(parts) >= 5 else parts[-1].strip()
                # Clean any quotes or accidental brackets
                text = re.sub(r"^[\"']|[\"']$", "", text).strip()
                rows.append(
                    {
                        "id": seg_id,
                        "speaker": speaker,
                        "gender": gender,
                        "emotion": emotion,
                        "recap_text": text,
                    }
                )
            except Exception:
                continue

    return rows


def generate_cinematic_script_via_chatgpt(
    segments: list[dict[str, Any]],
    output_script_path: str,
    *,
    target_language: str = "Hindi",
    batch_size: int = 45,
    overwrite: bool = False,
    profile_dir: str = "storage/browser_profile",
) -> list[dict[str, Any]]:
    """Run full story dossier extraction + conversational scene scripting with ChatGPT web automation."""
    if not overwrite and os.path.isfile(output_script_path):
        cached = read_json(output_script_path)
        if isinstance(cached, list) and len(cached) == len(segments):
            log.info("Loaded cached cinematic dub script (%d segments) from %s", len(cached), output_script_path)
            return cached

    controller = ChatGPTBrowserController(profile_dir=profile_dir)
    controller.start()

    try:
        log.info("🎬 [Phase 1/2] Analyzing Master Drama Dossier & Character Roster across %d dialogues...", len(segments))

        # Sample first 150 dialogue segments for master dossier analysis
        dossier_sample = segments[: min(150, len(segments))]
        sample_text = "\n".join(
            [f"[{s['id']}] ({s['speaker_cluster']}): {s['original_text']}" for s in dossier_sample]
        )

        dossier_prompt = f"""\
You are an expert anime and short-drama dubbing director in {target_language}.
Here is the opening dialogue transcript of our Chinese drama:
---
{sample_text}
---
Analyze the storyline, identify the characters, and establish who is speaking:
1. DRAMA SYNOPSIS: Summarize the core plot conflict, hero/heroine's goal, and main twists in 3 punchy sentences.
2. CHARACTER ROSTER: Identify characters from story semantics and dialogue relationships:
   - "hero" (Male Lead / Confident young protagonist - male)
   - "heroine" (Female Lead / Reborn warrior lady - female)
   - "maid" (Maid / Loyal female attendant - female)
   - "guard" (Soldiers / Guards / Generals - male)
   - "villain" (Main Antagonist / Sinister rival - male)
   - "father" (Elder Male Patriarch - male)
   - "mother" (Elder Female Matriarch - female)
   - "king" (Emperor / Monarch - male)
   - "extra_male" / "extra_female" (Bystanders)
3. Directing Tone: Modern, friendly, conversational, cinematic spoken {target_language}.

Confirm once you have understood the cast and storyline!
"""
        dossier_response = controller._send_message_and_wait(dossier_prompt)
        log.info("Master Drama Dossier Established!\n%s", dossier_response[:300])

        # ---------------------------------------------------------------------
        # Phase 2: Conversational Scene Scripting (40-50 segments per turn)
        # ---------------------------------------------------------------------
        log.info("🎭 [Phase 2/2] Scripting conversational dialogues in chunks of %d...", batch_size)
        batches = [segments[i : i + batch_size] for i in range(0, len(segments), batch_size)]

        scripted_dialogues: dict[int, dict[str, Any]] = {}

        for b_idx, batch in enumerate(batches, start=1):
            batch_payload = "\n".join(
                [
                    f"ID:{s['id']} | DUR:{s['duration']}s | MAX_CHARS:{max(18, int(float(s['duration']) * 14.0))} | SPK:{s['speaker_cluster']} | CHINESE:{s['original_text']}"
                    for s in batch
                ]
            )

            chunk_prompt = f"""\
Translate and adapt the next scene of dialogues (Batch {b_idx}/{len(batches)}, IDs {batch[0]['id']}-{batch[-1]['id']}) into natural spoken {target_language}.

CRITICAL RULES:
1. SEMANTIC SPEAKER ATTRIBUTION (DO NOT RELY BLINDLY ON SPK CLUSTER):
   - The raw audio tag (SPK: spk_X) can be imprecise. You MUST determine the true speaker primarily from the meaning and conversational context of the dialogue!
   - Examples:
     * When addressing someone as '小姐' (Miss/Lady) or serving her, SPEAKER = "maid" (GENDER = female).
     * When speaking of returning from 10 years disguised as a man ('你女扮男装替父从军十年'), that is the maid or companion speaking ("maid", female).
     * When someone says '我这是重生了' (I have been reborn), that is the female lead ("heroine", female).
     * When commanding soldiers or guarding gates, SPEAKER = "guard" or "hero".
     * When elder parents speak, SPEAKER = "father" (male) or "mother" (female).
   - Allowed SPEAKER values: hero | heroine | maid | guard | villain | father | mother | king | extra_male | extra_female.
2. SPEAKER CONTINUITY (DO NOT FLIP CHARACTERS MID-SPEECH):
   - When a character is speaking across consecutive dialogue turns or delivering a continuous monologue, KEEP THE SAME SPEAKER!
   - Never alternate between hero, villain, and extra_male within 1-2 seconds of the same dialogue flow.
3. CHARACTER BUDGET & SPOKEN PACING (MANDATORY):
   - Each dialogue line specifies MAX_CHARS (~14 characters/second).
   - The translated {target_language} text MUST NOT exceed MAX_CHARS! Keep words punchy, dramatic, and concise so the speech fits naturally without fast-forwarding or voice distortion.
4. NARRATIVE CONTINUITY: Review our ongoing conversation and the previous scene above! Seamlessly continue the flow, emotional momentum, and character dynamics from where we left off.
5. SPOKEN CASUAL DIALOGUE: Use authentic, engaging spoken language (no archaic Sanskritized or robotic textbook Hindi).
6. EMOTION TAG: Specify one emotion: angry | sarcastic | emotional | crying | laughing | shocked | neutral.
7. 100% PURE TARGET LANGUAGE: Translate completely into {target_language}. Do NOT leave any Chinese characters, names, or raw honorifics (e.g. adapt '小姐' into 'दीदी/मैडम', '陛下' into 'महाराज', '将军' into 'सेनापति/जनरल').

OUTPUT FORMAT:
| ID | SPEAKER | GENDER | EMOTION | DIALOGUE |

RAW DIALOGUES TO SCRIPT:
{batch_payload}
"""
            log.info("Sending Scene %d/%d (IDs %d-%d) to ChatGPT...", b_idx, len(batches), batch[0]["id"], batch[-1]["id"])
            chunk_response = controller._send_message_and_wait(chunk_prompt)
            parsed_rows = _parse_markdown_table(chunk_response)

            for row in parsed_rows:
                scripted_dialogues[row["id"]] = row

            log.info("  Scene %d scripted: parsed %d/%d dialogue lines.", b_idx, len(parsed_rows), len(batch))

        # Merge translations back into full segment array
        final_script: list[dict[str, Any]] = []
        for s in segments:
            sid = s["id"]
            row = scripted_dialogues.get(sid, {})
            merged = dict(s)
            merged_spk = row.get("speaker") or ("heroine" if s.get("speaker_cluster") in ["spk_1", "spk_3"] else "hero")
            merged["speaker"] = merged_spk
            merged["gender"] = row.get("gender") or ("female" if merged_spk in ["heroine", "maid", "mother", "extra_female"] else "male")
            merged["emotion"] = row.get("emotion") or "neutral"
            merged["recap_text"] = row.get("recap_text") or s["original_text"]
            final_script.append(merged)

        # Post-process: smooth out rogue 1-turn speaker flips within same continuous speech burst
        for i in range(1, len(final_script) - 1):
            prev_s = final_script[i - 1]
            curr_s = final_script[i]
            next_s = final_script[i + 1]
            gap_prev = curr_s["start"] - prev_s["end"]
            gap_next = next_s["start"] - curr_s["end"]
            if prev_s["speaker"] == next_s["speaker"] and curr_s["speaker"] != prev_s["speaker"]:
                if gap_prev < 1.2 and gap_next < 1.2 and curr_s["duration"] < 2.5:
                    log.info("Smoothing rogue speaker flip on seg %d: %s -> %s", curr_s["id"], curr_s["speaker"], prev_s["speaker"])
                    curr_s["speaker"] = prev_s["speaker"]
                    curr_s["gender"] = prev_s["gender"]

        log.info("Cinematic script complete: %d segments scripted.", len(final_script))
        write_json(output_script_path, final_script)
        return final_script

    finally:
        controller.close()
