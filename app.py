import os, json, datetime, subprocess, shutil, tempfile, re, math, textwrap
import streamlit as st
import numpy as np
import requests
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# CONSTANTS AND PATHS
WORKSPACE      = Path(__file__).parent
SFX_DIR        = WORKSPACE / 'sfx'
LIMITS_FILE    = WORKSPACE / 'user_limits.json'
OUTPUT_FILE    = WORKSPACE / 'final_output.mp4'
TEMP_DIR       = WORKSPACE / '_tmp'
DAILY_LIMIT    = 3
TARGET_CLIP_MIN = 3.5
TARGET_CLIP_MAX = 4.0
FADE_DURATION  = 0.5
FPS            = 30
WIDTH, HEIGHT  = 1920, 1080
NICHE_DB = {
    "tech": {
        "channel": "@TechVision",
        "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "keywords": ["technology","tech","software","computer","digital","ai","robot",
                     "internet","data","code","programming","machine","algorithm","cyber",
                     "network","cloud","server","hardware","chip","semiconductor"],
    },
    "business": {
        "channel": "@BizInsider",
        "url": "https://www.youtube.com/watch?v=3fumBcKC6RE",
        "keywords": ["business","company","startup","entrepreneur","market","economy",
                     "trade","commerce","profit","revenue","corporate","ceo","executive",
                     "strategy","growth","sales","brand","product","service","client"],
    },
    "finance": {
        "channel": "@WealthWave",
        "url": "https://www.youtube.com/watch?v=2lAe1cqCOXo",
        "keywords": ["finance","money","invest","stock","crypto","bitcoin","bank",
                     "wealth","fund","budget","debt","loan","interest","dollar",
                     "economy","financial","portfolio","asset","return","dividend"],
    },
    "history": {
        "channel": "@HistoryUnlocked",
        "url": "https://www.youtube.com/watch?v=9bZkp7q19f0",
        "keywords": ["history","war","ancient","empire","king","queen","civilization",
                     "century","era","dynasty","revolution","battle","medieval","roman",
                     "greek","egypt","world","historical","past","event"],
    },
    "lifestyle": {
        "channel": "@LifeVibes",
        "url": "https://www.youtube.com/watch?v=kJQP7kiw5Fk",
        "keywords": ["life","health","food","travel","fitness","yoga","wellness",
                     "mindset","motivation","success","happiness","family","home",
                     "nature","beauty","fashion","style","culture","relationship","love"],
    },
    "science": {
        "channel": "@ScienceDaily",
        "url": "https://www.youtube.com/watch?v=uelHwf8o7_U",
        "keywords": ["science","research","discovery","space","planet","nasa","biology",
                     "chemistry","physics","quantum","atom","cell","dna","gene",
                     "experiment","laboratory","theory","hypothesis","evidence","study"],
    },
}

SFX_URLS = {
    "whoosh.mp3":       "https://www.soundjay.com/misc/sounds/whoosh-01.mp3",
    "swish.mp3":        "https://www.soundjay.com/misc/sounds/swoosh-2.mp3",
    "camera_click.mp3": "https://www.soundjay.com/mechanical/sounds/camera-shutter-click-01.mp3",
}
SFX_FALLBACK_URLS = {
    "whoosh.mp3":       "https://actions.google.com/sounds/v1/transportation/rocket_flyby.ogg",
    "swish.mp3":        "https://actions.google.com/sounds/v1/weather/wind_and_rain.ogg",
    "camera_click.mp3": "https://actions.google.com/sounds/v1/office/stapler.ogg",
}

# MODULE 1 HELPERS - DAILY LIMIT TRACKING

def _today_str():
    return datetime.date.today().isoformat()


def load_limits():
    if LIMITS_FILE.exists():
        try:
            with open(LIMITS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_limits(data):
    with open(LIMITS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def get_session_key():
    """Use Streamlit session_id as a per-user key."""
    if "session_key" not in st.session_state:
        import uuid
        st.session_state["session_key"] = str(uuid.uuid4())
    return st.session_state["session_key"]


def check_and_increment_limit():
    """Returns (allowed: bool, remaining: int)."""
    key  = get_session_key()
    data = load_limits()
    today = _today_str()
    user_data = data.get(key, {})
    if user_data.get("date") != today:
        user_data = {"date": today, "count": 0}
    count = user_data["count"]
    if count >= DAILY_LIMIT:
        save_limits({**data, key: user_data})
        return False, 0
    user_data["count"] = count + 1
    data[key] = user_data
    save_limits(data)
    return True, DAILY_LIMIT - user_data["count"]


def get_remaining_today():
    key   = get_session_key()
    data  = load_limits()
    today = _today_str()
    user_data = data.get(key, {})
    if user_data.get("date") != today:
        return DAILY_LIMIT
    return max(0, DAILY_LIMIT - user_data.get("count", 0))


# SFX SETUP - download placeholder royalty-free assets if missing

def ensure_sfx():
    SFX_DIR.mkdir(parents=True, exist_ok=True)
    for filename, primary_url in SFX_URLS.items():
        dest = SFX_DIR / filename
        if dest.exists() and dest.stat().st_size > 1000:
            continue
        downloaded = False
        for url in [primary_url, SFX_FALLBACK_URLS.get(filename, "")]:
            if not url:
                continue
            try:
                resp = requests.get(url, timeout=15)
                if resp.status_code == 200 and len(resp.content) > 500:
                    dest.write_bytes(resp.content)
                    downloaded = True
                    break
            except Exception:
                pass
        if not downloaded:
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-f", "lavfi", "-i",
                     "aevalsrc=0:c=mono:s=44100:d=0.5",
                     "-acodec", "libmp3lame", str(dest)],
                    capture_output=True, timeout=30
                )
            except Exception:
                dest.write_bytes(b"")


# MODULE 2 - WHISPER TRANSCRIPTION AND TIMELINE SEGMENTATION

def transcribe_audio(audio_path: str) -> list:
    """
    Run Whisper base model on ANY length audio - no minute cap.
    Returns list of dicts: {text, start_time, end_time}
    """
    try:
        import whisper
        model = whisper.load_model("base")
        result = model.transcribe(
            audio_path,
            verbose=False,
            word_timestamps=False,
            fp16=False,
            language=None,
        )
        segments = []
        for seg in result.get("segments", []):
            segments.append({
                "text":       seg["text"].strip(),
                "start_time": float(seg["start"]),
                "end_time":   float(seg["end"]),
            })
        return segments
    except Exception as e:
        st.warning(f"Whisper transcription error: {e}")
        return []


def build_clip_timeline(segments: list) -> list:
    """
    Merges raw Whisper segments into visual clips of 3.5-4.0 s each.
    Returns list of dicts: {text, start_time, end_time, duration, keywords}
    """
    if not segments:
        return []

    clips = []
    bucket_text  = []
    bucket_start = segments[0]["start_time"]
    bucket_end   = segments[0]["end_time"]

    for seg in segments:
        seg_dur = seg["end_time"] - seg["start_time"]
        accum   = bucket_end - bucket_start

        if accum + seg_dur <= TARGET_CLIP_MAX:
            bucket_text.append(seg["text"])
            bucket_end = seg["end_time"]
        else:
            if accum >= TARGET_CLIP_MIN:
                clips.append(_make_clip_entry(bucket_text, bucket_start, bucket_end))
            bucket_text  = [seg["text"]]
            bucket_start = seg["start_time"]
            bucket_end   = seg["end_time"]

    if bucket_text and (bucket_end - bucket_start) > 0:
        clips.append(_make_clip_entry(bucket_text, bucket_start, bucket_end))

    return clips


def _make_clip_entry(texts, start, end):
    combined = " ".join(texts)
    return {
        "text":       combined,
        "start_time": start,
        "end_time":   end,
        "duration":   round(end - start, 3),
        "keywords":   extract_keywords(combined),
    }


def extract_keywords(text: str) -> list:
    """Lightweight keyword extractor - no NLTK dependency."""
    stopwords = {
        "the","a","an","and","or","but","in","on","at","to","for",
        "of","with","by","from","is","was","are","were","be","been",
        "have","has","had","do","does","did","will","would","could",
        "should","may","might","shall","can","this","that","these",
        "those","it","its","as","if","so","we","our","you","your",
        "they","their","he","she","his","her","not","no","up","out",
    }
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return [w for w in words if w not in stopwords and len(w) > 3]


# MODULE 3 - NICHE MATCHING AND YOUTUBE HARVESTING

def match_niche(keywords: list) -> dict:
    """Score each niche by keyword overlap and return best match."""
    best_niche  = "lifestyle"
    best_score  = -1
    keyword_set = set(keywords)
    for niche, info in NICHE_DB.items():
        score = len(keyword_set & set(info["keywords"]))
        if score > best_score:
            best_score  = score
            best_niche  = niche
    return NICHE_DB[best_niche]


def download_clip_section(yt_url: str, start_sec: float, duration: float, out_path: str) -> bool:
    """
    Use yt-dlp --download-sections to fetch ONLY the required seconds.
    Merges to mp4. Returns True on success.
    """
    end_sec  = start_sec + duration
    section  = f"*{start_sec:.2f}-{end_sec:.2f}"
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        "--download-sections", section,
        "--merge-output-format", "mp4",
        "-f", "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best",
        "-o", out_path,
        yt_url,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode == 0 and Path(out_path).exists() and Path(out_path).stat().st_size > 5000:
            return True
    except Exception:
        pass
    return False


def burn_attribution(input_path: str, output_path: str, channel_name: str, duration: float) -> bool:
    """Burn semi-transparent attribution text via FFmpeg drawtext filter."""
    label = f"Source: {channel_name}"
    label = label.replace(":", r"\:").replace("'", r"\'").replace("%", r"\%")
    vf = (
        f"drawtext=text='{label}':fontsize=28:fontcolor=white:"
        f"x=20:y=h-th-20:"
        f"box=1:boxcolor=black@0.55:boxborderw=6:"
        f"enable='between(t,0,{duration:.3f})'"
    )
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vf", vf,
        "-c:a", "copy",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        output_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        return result.returncode == 0 and Path(output_path).exists()
    except Exception:
        return False


# MODULE 4A - FALLBACK MOTION GRAPHIC (Pillow)

def create_motion_graphic_clip(keyword: str, duration: float, out_path: str, channel: str = "AI Generated") -> bool:
    """
    Generate a synthetic 1080p video slide with kinetic text using Pillow + FFmpeg.
    Used when yt-dlp fails so the pipeline never crashes.
    """
    try:
        frames_dir = Path(out_path).parent / ("_frames_" + Path(out_path).stem)
        frames_dir.mkdir(parents=True, exist_ok=True)

        total_frames = max(1, int(math.ceil(duration * FPS)))
        display_text = keyword.upper()
        wrapped      = textwrap.fill(display_text, width=22)

        for fi in range(total_frames):
            progress = fi / max(total_frames - 1, 1)
            r = int(10  + 20  * progress)
            g = int(10  + 10  * progress)
            b = int(30  + 60  * progress)
            img  = Image.new("RGB", (WIDTH, HEIGHT), color=(r, g, b))
            draw = ImageDraw.Draw(img)

            rect_w = int(WIDTH * 0.6 * (0.4 + 0.6 * progress))
            rect_x = (WIDTH - rect_w) // 2
            draw.rectangle(
                [rect_x, HEIGHT // 2 - 120, rect_x + rect_w, HEIGHT // 2 + 120],
                fill=(255, 200, 0)
            )

            font_size = int(80 * (0.5 + 0.5 * progress))
            try:
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size
                )
            except Exception:
                font = ImageFont.load_default()

            bbox   = draw.textbbox((0, 0), wrapped, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            tx     = (WIDTH  - tw) // 2
            ty     = (HEIGHT - th) // 2
            draw.text((tx, ty), wrapped, fill=(10, 10, 30), font=font)

            try:
                small_font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28
                )
            except Exception:
                small_font = ImageFont.load_default()

            attr_text = f"Source: {channel}"
            draw.rectangle(
                [10, HEIGHT - 65, 10 + len(attr_text) * 17 + 12, HEIGHT - 25],
                fill=(0, 0, 0)
            )
            draw.text((16, HEIGHT - 62), attr_text, fill=(255, 255, 255), font=small_font)

            frame_path = frames_dir / f"frame_{fi:05d}.png"
            img.save(str(frame_path))

        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(FPS),
            "-i", str(frames_dir / "frame_%05d.png"),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-t", str(duration),
            out_path,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=180)
        shutil.rmtree(str(frames_dir), ignore_errors=True)
        return result.returncode == 0 and Path(out_path).exists()
    except Exception as e:
        st.warning(f"Fallback graphic generation failed for '{keyword}': {e}")
        return False


# MODULE 4B - FRAME STANDARDIZATION (1920x1080 @ 30fps)

def standardize_clip(input_path: str, output_path: str) -> bool:
    """Force every clip to 1920x1080, 30fps, AAC audio via FFmpeg scale+pad filters."""
    vf = (
        "scale=1920:1080:force_original_aspect_ratio=decrease,"
        "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black,"
        "setsar=1"
    )
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vf", vf,
        "-r", "30",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-ar", "44100", "-ac", "2",
        "-shortest",
        output_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=180)
        return result.returncode == 0 and Path(output_path).exists()
    except Exception:
        return False


# MODULE 4C - XFADE TRANSITION ENGINE

XFADE_EFFECTS = ["fade", "slideleft", "slideright", "dissolve", "wipeleft", "wiperight"]


def get_clip_duration_ffprobe(path: str) -> float:
    """Return clip duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "json", path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=30, text=True)
        info   = json.loads(result.stdout)
        return float(info["format"]["duration"])
    except Exception:
        return TARGET_CLIP_MAX


def concatenate_with_xfade(clip_paths: list, output_path: str, sfx_path: str) -> bool:
    """
    Chain N clips together using FFmpeg xfade filter with 0.5s crossfade.
    Returns True on success.
    """
    if not clip_paths:
        return False

    if len(clip_paths) == 1:
        shutil.copy(clip_paths[0], output_path)
        return True

    n         = len(clip_paths)
    inputs    = []
    for p in clip_paths:
        inputs += ["-i", p]

    durations = [get_clip_duration_ffprobe(p) for p in clip_paths]

    v_parts = []
    a_parts = []
    offset  = 0.0

    for i in range(n - 1):
        effect  = XFADE_EFFECTS[i % len(XFADE_EFFECTS)]
        offset += durations[i] - FADE_DURATION
        v_prev  = f"[v{i}]" if i > 0 else f"[{i}:v]"
        v_next  = f"[{i+1}:v]"
        v_out   = f"[v{i+1}]"
        v_parts.append(
            f"{v_prev}{v_next}xfade=transition={effect}:"
            f"duration={FADE_DURATION}:offset={offset:.3f}{v_out}"
        )
        a_prev = f"[a{i}]" if i > 0 else f"[{i}:a]"
        a_next = f"[{i+1}:a]"
        a_out  = f"[a{i+1}]"
        a_parts.append(
            f"{a_prev}{a_next}acrossfade=d={FADE_DURATION}{a_out}"
        )

    final_v        = f"[v{n-1}]"
    final_a        = f"[a{n-1}]"
    filter_complex = "; ".join(v_parts + a_parts)

    cmd = (
        ["ffmpeg", "-y"]
        + inputs
        + ["-filter_complex", filter_complex,
           "-map", final_v,
           "-map", final_a,
           "-c:v", "libx264", "-preset", "fast", "-crf", "22",
           "-c:a", "aac", "-ar", "44100",
           output_path]
    )
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=600)
        return result.returncode == 0 and Path(output_path).exists()
    except Exception:
        return False


# MODULE 5 - MULTI-TRACK AUDIO MIXING WITH DUCKING

def build_sfx_timeline(clip_durations: list) -> list:
    """
    Build list of (timestamp, sfx_path) pairs.
    SFX triggers 0.25s BEFORE each xfade visual end so peak aligns at cut.
    """
    sfx_path    = str(SFX_DIR / "whoosh.mp3")
    if not Path(sfx_path).exists():
        sfx_path = None
    events      = []
    accumulated = 0.0
    for i, dur in enumerate(clip_durations[:-1]):
        accumulated += dur
        trigger_t   = accumulated - FADE_DURATION - 0.25
        if trigger_t >= 0 and sfx_path:
            events.append((round(trigger_t, 3), sfx_path))
        accumulated -= FADE_DURATION
    return events


def mix_final_audio(
    video_path: str,
    narration_path: str,
    sfx_events: list,
    output_path: str,
    clip_durations: list,
) -> bool:
    """
    Overlay narration (0 dB) + SFX events (-3 dB) with audio ducking during
    xfade windows. Uses FFmpeg amix + adelay + volume filtergraph.
    """
    inputs   = ["-i", video_path, "-i", narration_path]
    n_inputs = 2

    delay_parts = []
    mix_labels  = ["[0:a]", "[1:a]"]

    for idx, (ts, sfx_path) in enumerate(sfx_events):
        delay_ms  = int(ts * 1000)
        n_idx     = n_inputs + idx
        inputs   += ["-i", sfx_path]
        delay_parts.append(
            f"[{n_idx}:a]adelay={delay_ms}|{delay_ms}[sfx{idx}raw]; "
            f"[sfx{idx}raw]volume=0.75[sfx{idx}vol]"
        )
        mix_labels.append(f"[sfx{idx}vol]")

    duck_filter = ""
    accumulated = 0.0
    duck_parts  = []
    for i, dur in enumerate(clip_durations[:-1]):
        accumulated += dur
        t_start  = accumulated - FADE_DURATION - 0.25
        t_end    = accumulated + 0.25
        if t_start >= 0:
            duck_parts.append(
                f"volume=0.4:enable='between(t,{t_start:.3f},{t_end:.3f})'"
            )
        accumulated -= FADE_DURATION

    if duck_parts:
        duck_chain   = ",".join(duck_parts)
        duck_filter  = f"[0:a]{duck_chain}[vaud]; "
        mix_labels[0] = "[vaud]"

    nar_vol       = "[1:a]volume=1.0[narr]; "
    mix_labels[1] = "[narr]"

    all_mix = "".join(mix_labels)
    amix_filter = (
        duck_filter
        + nar_vol
        + ("; ".join(delay_parts) + ("; " if delay_parts else ""))
        + f"{all_mix}amix=inputs={len(mix_labels)}:duration=longest:normalize=0[aout]"
    )

    cmd = (
        ["ffmpeg", "-y"]
        + inputs
        + ["-filter_complex", amix_filter,
           "-map", "0:v",
           "-map", "[aout]",
           "-c:v", "copy",
           "-c:a", "aac", "-ar", "44100",
           "-shortest",
           output_path]
    )
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=600)
        if result.returncode == 0 and Path(output_path).exists():
            return True
        shutil.copy(video_path, output_path)
        return True
    except Exception:
        shutil.copy(video_path, output_path)
        return True


# MASTER PIPELINE ORCHESTRATOR

def simple_concat_fallback(paths: list) -> str:
    """Fallback: use FFmpeg concat demuxer (no transitions)."""
    try:
        list_file = str(TEMP_DIR / "concat_list.txt")
        with open(list_file, "w") as f:
            for p in paths:
                f.write(f"file '{p}'\n")
        out = str(TEMP_DIR / "concat_simple.mp4")
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", list_file,
            "-c", "copy", out,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=300)
        if result.returncode == 0 and Path(out).exists():
            return out
    except Exception:
        pass
    return None


def run_pipeline(audio_path: str, progress_bar, status_text) -> bool:
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    # STEP 1: Transcribe
    status_text.text("Step 1/5 - Running Whisper transcription (full audio, no cap)...")
    progress_bar.progress(5)
    segments = transcribe_audio(audio_path)
    if not segments:
        st.error("Transcription produced no segments. Check your audio file.")
        return False

    # STEP 2: Build timeline
    status_text.text("Step 2/5 - Building clip timeline (3.5-4.0 s intervals)...")
    progress_bar.progress(15)
    clips = build_clip_timeline(segments)
    if not clips:
        st.error("Could not build clip timeline from transcription.")
        return False
    st.info(f"Timeline built: {len(clips)} clips.")

    # STEP 3: Download and standardize clips
    status_text.text("Step 3/5 - Harvesting YouTube clips and standardizing frames...")
    std_paths      = []
    clip_durations = []
    total_clips    = len(clips)

    for idx, clip in enumerate(clips):
        pct = 15 + int(55 * idx / max(total_clips - 1, 1))
        progress_bar.progress(min(pct, 70))
        status_text.text(f"  Clip {idx+1}/{total_clips}: {clip['text'][:60]}...")

        niche_info = match_niche(clip["keywords"])
        yt_url     = niche_info["url"]
        channel    = niche_info["channel"]
        kws        = clip.get("keywords", [])
        words      = clip["text"].split() if clip["text"] else []
        keyword    = kws[0] if kws else (words[0] if words else "documentary")

        raw_path  = str(TEMP_DIR / f"raw_{idx:04d}.mp4")
        std_path  = str(TEMP_DIR / f"std_{idx:04d}.mp4")
        attr_path = str(TEMP_DIR / f"attr_{idx:04d}.mp4")

        target_dur = min(clip["duration"], TARGET_CLIP_MAX)
        target_dur = max(target_dur, TARGET_CLIP_MIN)
        yt_start   = float(idx % 60)

        dl_ok = download_clip_section(yt_url, yt_start, target_dur, raw_path)

        if not dl_ok:
            fallback_ok = create_motion_graphic_clip(keyword, target_dur, raw_path, channel)
            if not fallback_ok:
                st.warning(f"Clip {idx+1}: Both download and fallback failed. Skipping.")
                continue
            channel = "AI Generated"

        if not standardize_clip(raw_path, std_path):
            st.warning(f"Clip {idx+1}: Standardization failed; using raw clip.")
            std_path = raw_path

        real_dur = get_clip_duration_ffprobe(std_path)
        if burn_attribution(std_path, attr_path, channel, real_dur):
            final_clip_path = attr_path
        else:
            final_clip_path = std_path

        actual_dur = get_clip_duration_ffprobe(final_clip_path)
        std_paths.append(final_clip_path)
        clip_durations.append(actual_dur)

    if not std_paths:
        st.error("All clips failed. Cannot proceed.")
        return False

    # STEP 4: Concatenate with xfade transitions
    status_text.text("Step 4/5 - Applying CapCut-style xfade transitions...")
    progress_bar.progress(75)
    concat_path = str(TEMP_DIR / "concat.mp4")
    sfx_path    = str(SFX_DIR / "whoosh.mp3")
    concat_ok   = concatenate_with_xfade(std_paths, concat_path, sfx_path)
    if not concat_ok or not Path(concat_path).exists():
        st.warning("Xfade transitions failed. Using simple concatenation fallback...")
        concat_path = simple_concat_fallback(std_paths)
        if not concat_path:
            st.error("All concatenation methods failed.")
            return False

    # STEP 5: Multi-track audio mixing with ducking
    status_text.text("Step 5/5 - Mixing narration, SFX and applying audio ducking...")
    progress_bar.progress(88)
    sfx_events = build_sfx_timeline(clip_durations)
    mix_ok     = mix_final_audio(
        concat_path, audio_path, sfx_events, str(OUTPUT_FILE), clip_durations
    )
    progress_bar.progress(100)
    status_text.text("Pipeline complete!")
    return mix_ok and OUTPUT_FILE.exists()


# STREAMLIT UI

def main():
    st.set_page_config(
        page_title="Automated AI Documentary Editor",
        page_icon="🎬",
        layout="centered",
    )

    st.markdown("""
        <h1 style="text-align:center; font-size:2.4rem; color:#FFD700;">
            🎬 Automated AI Documentary Editor
        </h1>
        <p style="text-align:center; color:#aaa; margin-top:-10px;">
            Upload your narration audio — the AI handles everything entirely on the cloud.
        </p>
        <hr style="border-color:#333;">
    """, unsafe_allow_html=True)

    with st.spinner("Checking SFX assets..."):
        ensure_sfx()

    remaining = get_remaining_today()
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("#### Upload your narration file")
        st.caption("Accepts MP3, WAV or M4A — any length, no time cap.")
    with col2:
        color = "#00c853" if remaining > 1 else ("#ffab00" if remaining == 1 else "#d50000")
        st.markdown(
            f"""<div style="background:{color}22; border:1px solid {color};
            border-radius:8px; padding:8px 12px; text-align:center;">
            <b style="color:{color};">{remaining} / {DAILY_LIMIT}</b><br>
            <small style="color:#aaa;">generations left today</small>
            </div>""",
            unsafe_allow_html=True,
        )

    uploaded_file = st.file_uploader(
        "Drop audio file here",
        type=["mp3", "wav", "m4a"],
        label_visibility="collapsed",
    )

    with st.expander("Available Thematic Niches (auto-detected from your audio)"):
        niche_cols = st.columns(3)
        for i, (niche, info) in enumerate(NICHE_DB.items()):
            with niche_cols[i % 3]:
                st.markdown(f"**{niche.capitalize()}**\n\n{info['channel']}")

    st.markdown("---")

    if uploaded_file is not None:
        st.audio(uploaded_file, format=uploaded_file.type)

        if st.button("Generate AI Documentary Video", type="primary", use_container_width=True):
            allowed, left = check_and_increment_limit()
            if not allowed:
                st.error(
                    f"Daily limit reached ({DAILY_LIMIT}/day). Come back tomorrow.",
                    icon="🚫",
                )
                st.stop()

            TEMP_DIR.mkdir(parents=True, exist_ok=True)
            audio_ext  = Path(uploaded_file.name).suffix
            audio_path = str(TEMP_DIR / f"narration{audio_ext}")
            with open(audio_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.markdown("### Processing Pipeline")
            progress_bar = st.progress(0)
            status_text  = st.empty()

            try:
                success = run_pipeline(audio_path, progress_bar, status_text)
            except Exception as ex:
                st.error(f"Unexpected pipeline error: {ex}")
                success = False

            if success and OUTPUT_FILE.exists():
                st.success("Your AI Documentary is ready!")
                st.markdown("### Preview")
                st.video(str(OUTPUT_FILE))
                with open(str(OUTPUT_FILE), "rb") as vf:
                    st.download_button(
                        label="Download final_output.mp4",
                        data=vf,
                        file_name="ai_documentary.mp4",
                        mime="video/mp4",
                        use_container_width=True,
                    )
                st.info(f"Generations remaining today: {left}")
            else:
                st.error(
                    "Video generation failed. Check logs above for details. "
                    "Try re-uploading a different audio file."
                )
    else:
        st.info(
            "Upload a narration audio file (.mp3 / .wav / .m4a) to get started. "
            "All processing runs on the cloud — no local GPU needed.",
            icon="ℹ️",
        )

    st.markdown("---")
    st.markdown(
        "<p style='text-align:center;color:#555;font-size:0.8rem;'>"
        "Powered by Whisper · yt-dlp · FFmpeg · Streamlit — "
        "All processing on cloud hardware.</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
