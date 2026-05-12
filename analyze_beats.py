"""Analyze the music track to find beats, onset strength peaks, and the
strongest 'event' moments. Output:
  - beats.json: beat times, tempo, top-N strong beats
  - beats.png: waveform + beat grid + onset strength
"""
import json
import sys

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_beats.py path/to/music.mp3")
        sys.exit(1)
    audio_path = sys.argv[1]
    out_json = "morph_results/beats.json"
    out_png = "morph_results/beats.png"

    print(f"Loading {audio_path}...")
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    duration = len(y) / sr
    print(f"  duration: {duration:.2f}s, sample rate: {sr}")

    # Beat tracking
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
    tempo = float(tempo) if np.isscalar(tempo) else float(tempo[0])
    print(f"  tempo: {tempo:.1f} BPM, {len(beat_times)} beats detected")

    # Onset strength curve — useful for finding 'big moments'
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)
    onset_times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr)

    # Onset peaks across the WHOLE song
    onset_peaks = librosa.util.peak_pick(
        onset_env, pre_max=10, post_max=10, pre_avg=20, post_avg=20, delta=0.15, wait=10
    )
    peak_times = librosa.frames_to_time(onset_peaks, sr=sr)
    peak_strengths = onset_env[onset_peaks]

    # Pick the strongest peak inside each 6-second window — gives a map evenly
    # spread across the song instead of all the loudest hits clumped together.
    window = 6.0
    n_windows = int(np.ceil(duration / window))
    top_peaks = []
    for w in range(n_windows):
        t0, t1 = w * window, (w + 1) * window
        in_win = [(float(peak_times[i]), float(peak_strengths[i]))
                  for i in range(len(peak_times)) if t0 <= peak_times[i] < t1]
        if not in_win:
            continue
        in_win.sort(key=lambda p: p[1], reverse=True)
        # Take strongest in window, plus second-strongest if it's nearly as loud
        top_peaks.append(in_win[0])
        if len(in_win) >= 2 and in_win[1][1] > in_win[0][1] * 0.7:
            top_peaks.append(in_win[1])
    top_peaks.sort(key=lambda p: p[0])

    # Save JSON
    out = {
        "tempo_bpm": tempo,
        "duration_s": duration,
        "beats": [round(t, 3) for t in beat_times],
        "top_events": [
            {"t": round(t, 3), "strength": round(s, 3), "frame_30fps": int(round(t * 30))}
            for (t, s) in top_peaks
        ],
    }
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote {out_json}")

    # Visualization
    fig, ax = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    librosa.display.waveshow(y, sr=sr, ax=ax[0], alpha=0.6)
    for t in beat_times:
        ax[0].axvline(t, color="gray", alpha=0.3, linewidth=0.5)
    for t, s in top_peaks:
        ax[0].axvline(t, color="red", alpha=0.7, linewidth=1.5)
    ax[0].set_title(f"Waveform — gray=beats ({tempo:.0f} BPM), red=top events")
    ax[0].set_ylabel("amplitude")

    ax[1].plot(onset_times, onset_env, color="navy", linewidth=0.8)
    for t, s in top_peaks:
        ax[1].axvline(t, color="red", alpha=0.7, linewidth=1.5)
        ax[1].annotate(f"{t:.2f}s", (t, s), fontsize=8, color="red",
                       xytext=(2, 5), textcoords="offset points")
    ax[1].set_title(f"Onset strength — red lines mark {len(top_peaks)} strongest events per 6s window")
    ax[1].set_xlabel("time (s)")
    ax[1].set_ylabel("strength")
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_png, dpi=110)
    print(f"  wrote {out_png}")

    print(f"\nTop {len(top_peaks)} events (one or two per 6s window):")
    print("  time(s)  frame@30fps  strength")
    for t, s in top_peaks:
        print(f"  {t:6.2f}   {int(round(t*30)):5d}        {s:.2f}")


if __name__ == "__main__":
    main()
