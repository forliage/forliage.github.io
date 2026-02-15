---
title: "How to use TTS"
description: "Text2Speech is a indispensable part for scientific research, this blog introduce the basic principle and the usage of TTS"
pubDate: 2026-02-15
tags: ["TTS"]
heroImage: "/images/tts.jpg"
---

# I.Basic Principles and Training Process of TTS

## 1.1.Text Front-end

Objective: To transform the "raw text" into a representation that is easier for the model to pronounce.

Common steps:

* **Text normalization**: Convert "3.14", "2026/02/15", "$12.5" into readable language forms (e.g., "three point one four", "February 15, 2026", "twelve point five dollars").
* **Tokenization/Sentence splitting**: Avoid long sentences that cause prosody collapse.
* **G2P (Grapheme-to-Phoneme)**: Convert letters/characters to phonemes, especially important for English and mixed-language text.
* **Prosody hints**: Pauses, stress, speaking rate, etc. (often specified via SSML).

## 1.2.Acoustic Model

Objective: To predict **acoustic features** (e.g., mel-spectrogram) from "language representations (phonemes/text encoding + prosody)".

Typical route:

* Early stages: Tacotron / Tacotron2
* More stable later stages: Transformer TTS, FastSpeech / FastSpeech2 (non-autoregressive, faster, more controllable)

Training data is typically a large amount of **(text, speech)** aligned data; the model learns "how to read sentences like a human," including statistical patterns such as pauses, intonation, and linking.

## 1.3.Vocoder

Objective: To restore mel-spectrogram into playable waveform (wav).

Typical:

* WaveNet(Classic but slow)
* Parallel WaveGAN, HiFi-GAN(Fast, high quality, commonly used in engineering)

## 1.4.Post-processing and Engineering

* **Sample rate/bit rate conversion, loudness normalization**
* **Concatenation (multi-sentence, multi-paragraph synthesis)**
* **Caching and concurrency** (very important for video dubbing)
* **Controllability** (rate/pitch/volume, style, emotion, etc.: often specified via SSML or server parameters)

# II. How to use Microsoft Free TTS? Two Routes

### Route A: Edge Read Aloud(Common "free high-quality" method: edge-tts)

You are now using `edge_tts`, which is this route: it lets Python call **Microsoft Edge's online TTS/Read Aloud** capability. Installation is very light: `pip install edge-tts`.
And it supports selecting voice, adjusting rate/volume/pitch, etc. in the command line/code.

**Pros**

* No need for Azure key (easy to use, suitable for research demos, video dubbing, prototype verification)
* Speech quality is usually high (Neural voices)
* Can directly list a large number of voices (list_voices)

**Note (research usage should be included in README)**

* This is calling an online service: requires network; stability is affected by service policies (for example, some implementations mention API/UA restrictions changing).
* Suitable for "reproducible experiments/demos", but if you want to make a product or strictly comply with regulations, it is recommended to go through Azure official.

### Route B: Azure Speech(Official channel, with free quota)

Azure Speech's **Free (F0)** has **0.5 million characters per month of Neural TTS free quota** (official pricing page).
And the official support uses **SSML** to fine-tune speech rate, volume, pitch, pronunciation, etc.

**Pros**

* Compliant, stable, controllable, can do quota management, SLA, etc.
* SSML documentation is complete and reproducibility is better (especially for writing papers/open source)

**Cons**

* Requires Azure account and key (but free quota is usually enough for research prototypes)

## 2.1 Voice Selection and Parameter Tuning (The 5 Most Commonly Used Knobs for Scientific Research Dubbing)

### 1.Select Voice

Use `edge_tts.list_voices()` to pull the available voice list (with locale, gender, name), then pick ShortName as the parameter to pass in.

### 2.Speech rate, 3.Volume, 4.Pitch

`edge-tts` supports rate/volume/pitch parameters (command line and code are both supported).

### 5. More detailed control: SSML (official Azure documentation system)

When you want to achieve "emphasize a certain word", "make the pause in the middle more natural", "make a certain section slower and more emphasized", use SSML. The official SSML documentation provides control methods such as prosody.

# III. Example script (rewritten version, more suitable for scientific research: reproducible, scalable, batch, and spliceable)

## 3.1 Functional requirements

This script solves the most common needs for scientific research dubbing:

1. **List and filter voices** (filter by keywords such as "en-US / zh-CN / William / Female")
2. Generate multiple audio segments in batches from a `segments.txt` (automatic numbering, controllable file naming)
3. Optional: Concatenate all segments into a total mp3 with fixed pauses (for video narration)
4. Support rate/volume/pitch, and provide an **SSML mode** (more natural pauses/emphasis can be used)
5. Provide clear error reporting for ffmpeg dependencies

## 3.2 Preparation

```bash
pip install edge-tts
# Linux:
sudo apt install -y ffmpeg
# Windows: It is recommended to install ffmpeg using winget/choco.
```

For instructions on installing and using `edge-tts`, please refer to its PyPI page.


## 3.3 segments.txt format

Use `---` to separate paragraphs; the first line of each paragraph can optionally be a title (used for filenames):

```text
[01_test]
test
---
[02_intro_a]
Fast Gaussian Blur Approximation via Summed-Area Table of Polygons.
---
[02_intro_b]
In this video, we present a qualitative comparison by showcasing results rendered using different methods.
---
[15_thanks]
Thanks for your attention.
```

## 3.4 Complete script: `ms_tts_make_segments.py`

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import asyncio
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import edge_tts


def run(cmd: List[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


@dataclass
class Segment:
    key: str          # Used for filenames and sorting, such as "02_intro_a"
    text: str         # Text to be synthesized


def sanitize_filename(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[^\w\-\.]+", "_", s)  # Only keep safe characters
    return s[:120] if len(s) > 120 else s


def parse_segments_txt(path: Path) -> List[Segment]:
    """
    Parse segments.txt:
    - Use '---' to separate paragraphs
    - The first line of each paragraph can optionally be a title (used for filenames)
    - Lines starting with '#' are treated as comments
    """
    raw = path.read_text(encoding="utf-8")
    blocks = [b.strip() for b in raw.split("---")]
    segs: List[Segment] = []
    auto_idx = 1

    for b in blocks:
        if not b:
            continue
        lines = []
        for ln in b.splitlines():
            ln = ln.rstrip("\n")
            if not ln.strip():
                continue
            if ln.lstrip().startswith("#"):
                continue
            lines.append(ln)

        if not lines:
            continue

        key = None
        m = re.match(r"^\[(.+?)\]\s*$", lines[0].strip())
        if m:
            key = m.group(1).strip()
            text = "\n".join(lines[1:]).strip()
        else:
            key = f"{auto_idx:02d}"
            text = "\n".join(lines).strip()

        if not text:
            raise ValueError(f"Empty text for segment [{key}] in {path}")

        segs.append(Segment(key=key, text=text))
        auto_idx += 1

    return segs


async def list_voices(filter_str: Optional[str]) -> None:
    voices = await edge_tts.list_voices()
    for v in voices:
        name = (v.get("Name") or "")
        short = (v.get("ShortName") or "")
        gender = (v.get("Gender") or "")
        locale = (v.get("Locale") or "")
        line = f"{short:28s} | {gender:6s} | {locale:10s} | {name}"
        if filter_str:
            if filter_str.lower() in line.lower():
                print(line)
        else:
            print(line)


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found.\n"
            "Linux: sudo apt install -y ffmpeg\n"
            "Windows: install via winget/choco, and ensure ffmpeg is in PATH."
        )


async def tts_to_mp3(
    text: str,
    out_mp3: Path,
    voice: str,
    rate: str,
    volume: str,
    pitch: str,
    use_ssml: bool,
) -> None:
    out_mp3.parent.mkdir(parents=True, exist_ok=True)
    text = text.strip()
    if not text:
        raise ValueError(f"Empty text for output: {out_mp3}")

    if use_ssml:
        # Use minimal SSML: put rate/volume/pitch in prosody
        # If you migrate to Azure Speech in the future, this SSML is basically reusable (same idea)
        ssml = f"""<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis">
  <voice name="{voice}">
    <prosody rate="{rate}" volume="{volume}" pitch="{pitch}">
      {text}
    </prosody>
  </voice>
</speak>"""
        communicate = edge_tts.Communicate(text=ssml, voice=voice)
    else:
        communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate, volume=volume, pitch=pitch)

    await communicate.save(str(out_mp3))
    print("[OK] Saved:", out_mp3)


def concat_with_silence(
    parts: List[Path],
    out_mp3: Path,
    pause_sec: float,
    sample_rate: int = 24000,
) -> None:
    """
    Use ffmpeg to concatenate multiple mp3 files with silence in between.
    """
    ensure_ffmpeg()

    out_mp3.parent.mkdir(parents=True, exist_ok=True)

    ff_inputs: List[str] = []
    filter_inputs: List[str] = []
    idx = 0

    for i, p in enumerate(parts):
        if not p.exists():
            raise FileNotFoundError(f"Missing part: {p}")
        ff_inputs += ["-i", str(p)]
        filter_inputs.append(f"[{idx}:a]")
        idx += 1

        if i != len(parts) - 1:
            ff_inputs += ["-f", "lavfi", "-t", f"{pause_sec}", "-i", f"anullsrc=r={sample_rate}:cl=mono"]
            filter_inputs.append(f"[{idx}:a]")
            idx += 1

    n = len(filter_inputs)
    filter_complex = "".join(filter_inputs) + f"concat=n={n}:v=0:a=1[aout]"

    cmd = [
        "ffmpeg", "-y",
        *ff_inputs,
        "-filter_complex", filter_complex,
        "-map", "[aout]",
        "-c:a", "libmp3lame", "-b:a", "128k",
        str(out_mp3),
    ]
    run(cmd)
    print("[OK] Merged:", out_mp3)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments", default="segments.txt", help="Input segments file")
    parser.add_argument("--out_dir", default="tts_out", help="Output directory")
    parser.add_argument("--voice", default="en-US-AndrewNeural", help="Voice short name")
    parser.add_argument("--rate", default="+0%", help='Rate like "+0%", "+10%", "-10%"')
    parser.add_argument("--volume", default="+0%", help='Volume like "+0%", "+10%", "-10%"')
    parser.add_argument("--pitch", default="+0Hz", help='Pitch like "+0Hz", "+2Hz", "-2Hz"')
    parser.add_argument("--use_ssml", action="store_true", help="Wrap text with minimal SSML prosody")

    parser.add_argument("--list_voices", action="store_true", help="List available voices and exit")
    parser.add_argument("--voice_filter", default=None, help='Filter when listing voices, e.g. "en-US" or "William"')

    parser.add_argument("--merge_all", action="store_true", help="Merge all generated mp3 into one")
    parser.add_argument("--pause_sec", type=float, default=0.45, help="Pause seconds between merged segments")
    parser.add_argument("--merged_name", default="merged.mp3", help="Merged output filename")
    parser.add_argument("--clean_tmp", action="store_true", help="Delete temp files after merging")

    parser.add_argument("--max_concurrency", type=int, default=4, help="Max parallel TTS requests")

    args = parser.parse_args()

    if args.list_voices:
        await list_voices(args.voice_filter)
        return

    out_dir = Path(args.out_dir)
    tmp_dir = out_dir / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    segs = parse_segments_txt(Path(args.segments))
    print(f"[INFO] Loaded {len(segs)} segments from {args.segments}")

    sem = asyncio.Semaphore(args.max_concurrency)
    part_paths: List[Tuple[str, Path]] = []

    async def one(seg: Segment) -> None:
        async with sem:
            fname = sanitize_filename(seg.key) + ".mp3"
            out_mp3 = tmp_dir / fname
            await tts_to_mp3(
                text=seg.text,
                out_mp3=out_mp3,
                voice=args.voice,
                rate=args.rate,
                volume=args.volume,
                pitch=args.pitch,
                use_ssml=args.use_ssml,
            )
            part_paths.append((seg.key, out_mp3))

    await asyncio.gather(*[one(s) for s in segs])

    # Output each segment to out_dir (more intuitive and convenient for video segment alignment)
    out_dir.mkdir(parents=True, exist_ok=True)
    part_paths.sort(key=lambda x: x[0])

    final_parts: List[Path] = []
    for key, p in part_paths:
        dst = out_dir / (sanitize_filename(key) + ".mp3")
        dst.write_bytes(p.read_bytes())
        final_parts.append(dst)

    if args.merge_all:
        merged = out_dir / args.merged_name
        concat_with_silence(final_parts, merged, pause_sec=args.pause_sec)

    if args.clean_tmp:
        for p in tmp_dir.glob("*.mp3"):
            try:
                p.unlink()
            except OSError:
                pass
        try:
            tmp_dir.rmdir()
        except OSError:
            pass

    print("\nAll done. Output dir:", out_dir.resolve())


if __name__ == "__main__":
    asyncio.run(main())
```

## 3.5 How to use

1. List English American voices (filter):

```bash
python ms_tts_make_segments.py --list_voices --voice_filter en-US
```

2. Batch synthesis with a specific voice (concurrency 4):

```bash
python ms_tts_make_segments.py \
  --segments segments.txt \
  --out_dir tts_out \
  --voice en-AU-WilliamNeural \
  --rate +5% --volume +0% --pitch +0Hz \
  --max_concurrency 4
```

3. Synthesize and merge into a total narration:

```bash
python ms_tts_make_segments.py \
  --segments segments.txt \
  --out_dir tts_out \
  --voice en-US-AndrewNeural \
  --rate +0% --volume +0% --pitch +0Hz \
  --merge_all --pause_sec 0.45 --merged_name narration.mp3 \
  --clean_tmp
```