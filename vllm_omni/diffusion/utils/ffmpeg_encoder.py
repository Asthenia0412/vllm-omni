# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""FFmpeg subprocess-backed encoders for chunked video MP4 output.

The existing PyAV-backed :class:`ChunkedMP4Encoder` is sufficient when a
single in-process muxer is the right answer, but PyAV runs on the CPU of one
thread. These encoders hand raw ``rgb24`` frames to an ``ffmpeg`` binary
subprocess so the codec can use multi-threading, platform hardware encoders,
or segmented parallel encoding across many processes.

Three backends live here. All three expose the same contract as the PyAV
encoder (``push`` / ``finish`` / ``abort``) so :class:`ChunkedVideoMP4Session`
can switch between them behind a single opt-in flag.

* :class:`FFmpegBinaryEncoder` — one subprocess, one full MP4.  Replaces the
  PyAV backend and gives access to encoder options PyAV does not expose.
* :class:`SegmentedFFmpegEncoder` — many subprocesses, each owns one frame
  range.  Finishes by concatenating segments with an ``ffmpeg -c copy`` run
  and muxing audio once.  See RFC #8887.

Audio muxing happens once at the end even in segmented mode: each segment
produces video-only MP4 fragments and the final concat run adds the audio
waveform exactly once so there is no risk of duplicated audio tracks or
misaligned PTS values.
"""

from __future__ import annotations

import os
import queue
import shutil
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from typing import Callable

import numpy as np

try:
    from vllm.logger import init_logger
    logger = init_logger(__name__)
except Exception:  # pragma: no cover - fallback for import-only contexts
    import logging
    logger = logging.getLogger(__name__)

try:
    from .media_utils import _validate_video_chunk, mux_av_video_audio_bytes
except Exception:  # pragma: no cover - fallback for import-only / standalone tests
    def _validate_video_chunk(chunk, *, width, height):  # type: ignore[misc]
        if chunk.ndim != 4 or chunk.shape[-1] != 3:
            raise ValueError("video chunk must have shape (T, H, W, 3)")
        if chunk.dtype != np.uint8:
            raise ValueError("video chunk must have dtype uint8")
        if chunk.shape[1] != height or chunk.shape[2] != width:
            raise ValueError(
                "video chunks in a session must use a consistent frame size"
            )

    def mux_av_video_audio_bytes(*args, **kwargs):  # type: ignore[misc]
        raise NotImplementedError(
            "media_utils.mux_av_video_audio_bytes is unavailable in this context"
        )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _ffmpeg_available() -> bool:
    """Return True if an ``ffmpeg`` binary is on the system ``PATH``."""
    return shutil.which("ffmpeg") is not None


def _probe_ffmpeg_version() -> str | None:
    """Return the ffmpeg version string, or None if unavailable."""
    if not _ffmpeg_available():
        return None
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    first = result.stdout.splitlines()[0] if result.stdout else ""
    return first.strip() or None


def _build_encode_cmd(
    *,
    width: int,
    height: int,
    fps: float,
    video_codec: str,
    crf: str,
    video_codec_options: dict[str, str] | None,
    output_path: str,
    audio_path: str | None = None,
    audio_sample_rate: int | None = None,
    audio_codec: str = "aac",
    silent: bool = True,
) -> list[str]:
    """Assemble an ``ffmpeg`` command line for a rawvideo-in, mp4-out run."""
    cmd: list[str] = ["ffmpeg", "-y"]
    # Suppress progress chatter.
    cmd += ["-loglevel", "error"]
    if silent:
        cmd += ["-nostdin"]

    # Video input: raw rgb24 on stdin.
    cmd += [
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{width}x{height}",
        "-pix_fmt", "rgb24",
        "-r", str(float(Fraction(fps).limit_denominator(10000))),
        "-i", "pipe:0",
    ]

    # Audio input, if the caller prepared one offline.
    if audio_path is not None:
        cmd += ["-i", audio_path]

    # Video encoder.
    if video_codec == "h264":
        video_encoder = "libx264"
    elif video_codec == "hevc" or video_codec == "h265":
        video_encoder = "libx265"
    else:
        video_encoder = video_codec

    cmd += [
        "-c:v", video_encoder,
        "-pix_fmt", "yuv420p",
    ]

    # Encode options as ``-key value`` pairs so every PyAV-style option can
    # pass through without ffmpeg CLI shape assumptions.
    if video_codec_options:
        for key, value in video_codec_options.items():
            cmd += [f"-{key}", str(value)]

    cmd += ["-crf", str(crf)]

    # No B-frames so segmented encodes have deterministic decoder behavior at
    # segment boundaries.  Segments with B-frames require explicit PTS offset
    # adjustments during concat.
    try:
        subprocess.run(
            ["ffmpeg", "-h", f"encoder={video_encoder}"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        cmd += ["-bf", "0"]
    except Exception:
        pass

    # Audio encoder (skip if we have no audio input).
    if audio_path is not None:
        cmd += ["-c:a", audio_codec]
        if audio_sample_rate is not None:
            cmd += ["-ar", str(audio_sample_rate)]
    else:
        cmd += ["-an"]

    cmd += [output_path]
    return cmd


def _frames_to_raw(frames: np.ndarray) -> bytes:
    """Pack a ``(T, H, W, 3)`` uint8 RGB array into contiguous raw rgb24 bytes."""
    if not frames.flags.c_contiguous:
        frames = np.ascontiguousarray(frames)
    return frames.tobytes()


# ---------------------------------------------------------------------------
# Base encoder contract
# ---------------------------------------------------------------------------


class _BaseChunkedFFmpegEncoder:
    """Shared queue + thread + validation machinery for ffmpeg-backed encoders.

    Subclasses implement :meth:`_run_encode` and :meth:`_result_path`; this
    base class owns the producer-side contract (``push`` / ``finish`` /
    ``abort``), chunk validation, in-order queueing, and the final process
    readback.
    """

    _DONE = object()
    _ABORT = object()

    def __init__(
        self,
        *,
        width: int,
        height: int,
        fps: float,
        audio_waveform: np.ndarray | None = None,
        audio_sample_rate: int | None = None,
        max_pending: int = 8,
        video_codec: str = "h264",
        audio_codec: str = "aac",
        crf: str = "18",
        video_codec_options: dict[str, str] | None = None,
    ) -> None:
        if max_pending <= 0:
            raise ValueError("max_pending must be positive")
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        if not _ffmpeg_available():
            raise RuntimeError(
                "ffmpeg binary not found on PATH; cannot use ffmpeg encoder backend"
            )
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self._audio_waveform = audio_waveform
        self._audio_sample_rate = audio_sample_rate
        self._audio_codec = audio_codec
        self._video_codec = video_codec
        self._crf = str(crf)
        self._video_codec_options = dict(video_codec_options or {})

        self._queue: queue.Queue[object] = queue.Queue(maxsize=max_pending)
        self._result: bytes | None = None
        self._error: BaseException | None = None
        self._closed = False
        self._state_lock = threading.Lock()

        self._thread = threading.Thread(
            target=self._run,
            name=f"ffmpeg-encoder-{type(self).__name__}",
            daemon=True,
        )
        self._thread.start()

    # ---- subclass hooks ---------------------------------------------------

    def _run_encode(self, collected: list[np.ndarray]) -> bytes:
        """Encode all collected chunks into final MP4 bytes."""
        raise NotImplementedError

    # ---- lifecycle --------------------------------------------------------

    def push(
        self,
        chunk: np.ndarray,
        *,
        on_consumed: Callable[[], None] | None = None,
    ) -> None:
        """Queue one ``(T, H, W, 3)`` uint8 RGB chunk."""
        _validate_video_chunk(chunk, width=self.width, height=self.height)
        with self._state_lock:
            if self._closed:
                raise RuntimeError("ffmpeg encoder is already closed")
        self._queue.put((chunk, on_consumed))

    def finish(self) -> bytes:
        """Flush the queue, wait for encoding, return MP4 bytes."""
        with self._state_lock:
            if not self._closed:
                self._closed = True
                self._queue.put(self._DONE)
        self._thread.join()
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result

    close = finish

    def abort(self) -> None:
        """Drop pending chunks and tear down the encoder.

        Already-consumed chunks cannot be returned -- the subprocess may be
        mid-encode -- but chunks still in the queue are released.
        """
        with self._state_lock:
            if not self._closed:
                self._closed = True
                # Drain whatever remains and release each callback.
                while True:
                    try:
                        entry = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    if entry is self._DONE or entry is self._ABORT:
                        continue
                    _chunk, cb = entry
                    if cb is not None:
                        try:
                            cb()
                        except BaseException:  # noqa: BLE001
                            logger.exception(
                                "ffmpeg encoder failed to release a chunk during abort"
                            )
                self._queue.put(self._ABORT)
        self._thread.join()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.finish()
        else:
            self.abort()

    # ---- worker -----------------------------------------------------------

    def _run(self) -> None:
        collected: list[np.ndarray] = []
        try:
            while True:
                entry = self._queue.get()
                if entry is self._DONE:
                    break
                if entry is self._ABORT:
                    return
                chunk, cb = entry
                collected.append(chunk)
                if cb is not None:
                    try:
                        cb()
                    except BaseException:  # noqa: BLE001
                        logger.exception(
                            "ffmpeg encoder failed to release a consumed chunk"
                        )
            self._result = self._run_encode(collected)
        except BaseException as exc:  # noqa: BLE001
            self._error = exc


# ---------------------------------------------------------------------------
# Single-process ffmpeg encoder (replaces PyAV backend)
# ---------------------------------------------------------------------------


class FFmpegBinaryEncoder(_BaseChunkedFFmpegEncoder):
    """One ``ffmpeg`` subprocess encodes all collected chunks into one MP4.

    Replaces the in-process PyAV-backed :class:`ChunkedMP4Encoder` with the
    same contract but gives access to the full ``ffmpeg`` CLI: hardware
    encoders (``h264_nvenc`` / ``hevc_nvenc`` / ``h264_vaapi``), multi-threaded
    libx264 / libx265 with all of their tuning knobs, and any codec the host
    happens to ship.
    """

    def _run_encode(self, collected: list[np.ndarray]) -> bytes:
        if not collected:
            raise RuntimeError("ffmpeg encoder received no frames")
        full = np.concatenate(collected, axis=0)
        with tempfile.TemporaryDirectory(prefix="ffmpeg-enc-") as td:
            out_path = os.path.join(td, "out.mp4")
            cmd = _build_encode_cmd(
                width=self.width,
                height=self.height,
                fps=self.fps,
                video_codec=self._video_codec,
                crf=self._crf,
                video_codec_options=self._video_codec_options,
                output_path=out_path,
                # Audio is muxed after the pure-video encode so we do not need
                # a wav intermediate file here.
                audio_path=None,
            )
            proc = subprocess.run(
                cmd,
                input=_frames_to_raw(full),
                capture_output=True,
                timeout=3600,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg single-process encode failed (code {proc.returncode}): "
                    f"{proc.stderr.decode(errors='replace')[-800:]}"
                )
            video_bytes = _read_file(out_path)

        # Mux audio exactly once on top of the pure-video stream.
        if self._audio_waveform is None:
            return video_bytes
        return _mux_audio_overlay(
            video_bytes=video_bytes,
            audio_waveform=self._audio_waveform,
            audio_sample_rate=self._audio_sample_rate,
            audio_codec=self._audio_codec,
        )


# ---------------------------------------------------------------------------
# Segmented parallel encoder (RFC #8887 first half)
# ---------------------------------------------------------------------------


class SegmentedFFmpegEncoder(_BaseChunkedFFmpegEncoder):
    """Partition collected frames into ranges and encode each range in parallel.

    All ranges encode concurrently in worker threads, each driving its own
    ``ffmpeg`` subprocess.  Once every range finishes, the segments are
    concatenated with an ``ffmpeg -c copy`` run and the audio waveform is
    muxed once.  A falling partition (earlier segments get more frames, later
    segments fewer) is available for staggered latencies: when the VAE decode
    phase hands chunks to the encoder a few frames at a time the segments
    that start first see proportionally more work so their wall clock time
    stays close to the last one's instead of early segments becoming idle and
    waiting for the tail.
    """

    def __init__(
        self,
        *,
        num_segments: int = 0,
        segment_ratio: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if num_segments < 0:
            raise ValueError("num_segments must be non-negative; 0 means auto")
        self._num_segments = num_segments
        self._segment_ratio = segment_ratio

    def _run_encode(self, collected: list[np.ndarray]) -> bytes:
        if not collected:
            raise RuntimeError("segmented ffmpeg encoder received no frames")
        full = np.concatenate(collected, axis=0)
        total_frames = int(full.shape[0])

        num_segments = self._resolve_num_segments(total_frames)
        ranges = _plan_segment_ranges(total_frames, num_segments, self._segment_ratio)

        with tempfile.TemporaryDirectory(prefix="seg-ffmpeg-") as td:
            segment_paths: list[str] = []
            errors: list[BaseException] = []
            lock = threading.Lock()

            def encode_range(start: int, end: int, index: int) -> str:
                if start >= end:
                    return ""
                seg_path = os.path.join(td, f"seg_{index:04d}.mp4")
                seg_frames = full[start:end]
                cmd = _build_encode_cmd(
                    width=self.width,
                    height=self.height,
                    fps=self.fps,
                    video_codec=self._video_codec,
                    crf=self._crf,
                    video_codec_options=self._video_codec_options,
                    output_path=seg_path,
                    audio_path=None,
                )
                proc = subprocess.run(
                    cmd,
                    input=_frames_to_raw(seg_frames),
                    capture_output=True,
                    timeout=3600,
                )
                if proc.returncode != 0:
                    err = RuntimeError(
                        f"segment {index} encode failed (code {proc.returncode}): "
                        f"{proc.stderr.decode(errors='replace')[-600:]}"
                    )
                    with lock:
                        errors.append(err)
                    return ""
                return seg_path

            # Cap worker threads so a runaway auto-count does not spawn more
            # ffmpeg processes than the host can schedule.
            max_workers = min(num_segments, 8, os.cpu_count() or 4)
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [
                    pool.submit(encode_range, start, end, idx)
                    for idx, (start, end) in enumerate(ranges)
                    if start < end
                ]
                segment_paths = [
                    fut.result() for fut in futures if not fut.cancelled()
                ]

            if errors:
                raise errors[0]

            # Verify every segment was produced.
            segment_paths = [p for p in segment_paths if p]
            if not segment_paths:
                raise RuntimeError("segmented encode produced no segments")

            final_video = _concat_segments(segment_paths, td)

        if self._audio_waveform is None:
            return final_video
        return _mux_audio_overlay(
            video_bytes=final_video,
            audio_waveform=self._audio_waveform,
            audio_sample_rate=self._audio_sample_rate,
            audio_codec=self._audio_codec,
        )

    def _resolve_num_segments(self, total_frames: int) -> int:
        if self._num_segments > 0:
            return min(self._num_segments, total_frames)
        # Auto: at least 2, at most 8, capped by total frames.  We want enough
        # parallelism to help but not so many workers that the encode of a
        # short video spends more time process-starting than encoding.
        cpu = os.cpu_count() or 4
        return max(2, min(cpu, 8, total_frames))


# ---------------------------------------------------------------------------
# Segment planning & concat helpers
# ---------------------------------------------------------------------------


def _plan_segment_ranges(
    total_frames: int,
    num_segments: int,
    ratio: str | None,
) -> list[tuple[int, int]]:
    """Return ``[(start, end), ...]`` covering every frame exactly once.

    ``ratio`` is either ``None`` (equal partitioning) or a comma-separated
    list of positive weights, one per segment.  Weights do not need to sum to
    one; they are normalized internally.  A decreasing sequence (e.g.
    ``4,3,2,1``) puts more frames on earlier segments so the tail segment
    starts with less work.  An equal sequence is the default.
    """
    if total_frames <= 0:
        return []
    if num_segments <= 1 or total_frames == 1:
        return [(0, total_frames)]

    if ratio is None:
        weights = [1.0] * num_segments
    else:
        parsed = [float(w) for w in ratio.split(",") if w.strip()]
        if len(parsed) != num_segments:
            raise ValueError(
                f"segment_ratio has {len(parsed)} weights but num_segments={num_segments}"
            )
        if any(w <= 0 for w in parsed):
            raise ValueError("segment_ratio weights must all be positive")
        weights = parsed

    total_w = sum(weights)
    # Distribute frame counts proportional to weights.  The first ``num_segments
    # - 1`` segments get floor allocations; the last picks up whatever remains
    # so every frame is accounted for.
    allocations = [int(total_frames * w / total_w) for w in weights[:-1]]
    allocated = sum(allocations)
    # Ensure every non-last segment gets at least 1 frame when possible.
    for i in range(len(allocations)):
        if allocations[i] == 0 and allocated < total_frames - 1:
            allocations[i] = 1
            allocated += 1
    last = total_frames - allocated
    allocations.append(max(1, last))

    # Build cumulative ranges, trimming any zero at the end.
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for count in allocations:
        if count <= 0:
            continue
        ranges.append((cursor, cursor + count))
        cursor += count
    # If the last is also zero (should not happen), drop it.
    if ranges and ranges[-1][0] >= ranges[-1][1]:
        ranges.pop()
    return ranges


def _concat_segments(segment_paths: list[str], workdir: str) -> bytes:
    """Run ``ffmpeg -f concat -c copy`` over ``segment_paths``."""
    if not segment_paths:
        raise RuntimeError("no segments to concatenate")
    if len(segment_paths) == 1:
        return _read_file(segment_paths[0])

    concat_list = os.path.join(workdir, "concat.txt")
    with open(concat_list, "w", encoding="utf-8") as fh:
        for path in segment_paths:
            # ffmpeg concat demuxer requires single-quoted paths and does not
            # escape inside the quotes; real paths here are tempfile names so
            # this is safe.
            fh.write(f"file '{path}'\n")

    out_path = os.path.join(workdir, "concat.mp4")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_list,
        "-c", "copy",
        out_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=3600)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg concat failed (code {proc.returncode}): "
            f"{proc.stderr.decode(errors='replace')[-600:]}"
        )
    return _read_file(out_path)


def _mux_audio_overlay(
    *,
    video_bytes: bytes,
    audio_waveform: np.ndarray,
    audio_sample_rate: int | None,
    audio_codec: str,
) -> bytes:
    """Merge pure-video MP4 bytes and an audio waveform into one MP4.

    Runs a short ``ffmpeg`` subprocess that (1) writes the audio waveform to a
    temp f32le input, (2) reads the pure-video MP4 and (3) streams them into a
    new MP4.  The video stream is copied byte-for-byte (no re-encode) and the
    audio is encoded exactly once with the requested codec.  Keeping the overlay
    in ffmpeg means we never need to touch PyAV's container-writing API, which
    varies significantly across versions.
    """
    if not video_bytes:
        raise RuntimeError("cannot overlay audio on empty video bytes")
    sample_rate = 44100 if audio_sample_rate is None else audio_sample_rate

    audio_waveform = audio_waveform.astype(np.float32)
    if audio_waveform.ndim == 1:
        audio_waveform = audio_waveform.reshape(1, -1)
    elif audio_waveform.ndim == 2 and audio_waveform.shape[0] > audio_waveform.shape[1]:
        audio_waveform = np.ascontiguousarray(audio_waveform.T)
    num_channels = audio_waveform.shape[0]

    with tempfile.TemporaryDirectory(prefix="ffmpeg-overlay-") as td:
        video_path = os.path.join(td, "video.mp4")
        with open(video_path, "wb") as fh:
            fh.write(video_bytes)

        # Audio: pipe f32le interleaved float32 samples; ffmpeg needs
        # "interleaved" channels so we transpose (C, N) -> (N, C).
        audio_interleaved = np.ascontiguousarray(audio_waveform.T).tobytes()

        audio_cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "f32le",
            "-ar", str(sample_rate),
            "-ac", str(num_channels),
            "-i", "pipe:0",
        ]
        # Run this subprocess just to write a WAV from the pipe, then read that
        # WAV file back in the overlay subprocess's second input slot.  Two
        # subprocesses is simpler than a single command with two pipe inputs
        # (ffmpeg cannot have two stdins from the parent shell).
        audio_path = os.path.join(td, "audio.wav")
        audio_proc = subprocess.run(
            audio_cmd + [audio_path],
            input=audio_interleaved,
            capture_output=True,
            timeout=3600,
        )
        if audio_proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg audio-prep failed (code {audio_proc.returncode}): "
                f"{audio_proc.stderr.decode(errors='replace')[-400:]}"
            )

        out_path = os.path.join(td, "overlay.mp4")
        overlay_cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", audio_codec,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            out_path,
        ]
        overlay_proc = subprocess.run(overlay_cmd, capture_output=True, timeout=3600)
        if overlay_proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg audio-overlay failed (code {overlay_proc.returncode}): "
                f"{overlay_proc.stderr.decode(errors='replace')[-500:]}"
            )

        return _read_file(out_path)


def _read_file(path: str) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# Encoder factory
# ---------------------------------------------------------------------------


def make_chunked_video_encoder(
    backend: str,
    *,
    width: int,
    height: int,
    fps: float,
    audio_waveform: np.ndarray | None = None,
    audio_sample_rate: int | None = None,
    max_pending: int = 2,
    video_codec: str = "h264",
    audio_codec: str = "aac",
    crf: str = "18",
    video_codec_options: dict[str, str] | None = None,
    num_segments: int = 0,
    segment_ratio: str | None = None,
):
    """Instantiate a chunked-video encoder for ``backend``.

    ``backend`` may be one of:

    * ``"pyav"`` -- the in-process PyAV-backed encoder (default).
    * ``"ffmpeg"`` -- single subprocess encoder (gives access to the full
      ffmpeg CLI: hardware encoders, multi-threading).
    * ``"segmented_ffmpeg"`` -- many subprocesses, parallel per-range
      encoding with one audio mux at the end.

    Every backend exposes the same ``push`` / ``finish`` / ``abort`` contract
    so callers do not need to know which implementation they got.
    """
    be = (backend or "pyav").lower()
    if be == "pyav":
        try:
            from .media_utils import ChunkedMP4Encoder
        except Exception as exc:  # pragma: no cover - environment gap
            raise RuntimeError(
                "pyav backend requires vllm_omni.diffusion.utils.media_utils"
            ) from exc
        return ChunkedMP4Encoder(
            width=width,
            height=height,
            fps=fps,
            audio_waveform=audio_waveform,
            audio_sample_rate=audio_sample_rate,
            max_pending=max_pending,
            video_codec=video_codec,
            audio_codec=audio_codec,
            crf=crf,
            video_codec_options=video_codec_options,
        )
    if be == "ffmpeg":
        return FFmpegBinaryEncoder(
            width=width,
            height=height,
            fps=fps,
            audio_waveform=audio_waveform,
            audio_sample_rate=audio_sample_rate,
            max_pending=max_pending,
            video_codec=video_codec,
            audio_codec=audio_codec,
            crf=crf,
            video_codec_options=video_codec_options,
        )
    if be == "segmented_ffmpeg":
        return SegmentedFFmpegEncoder(
            width=width,
            height=height,
            fps=fps,
            audio_waveform=audio_waveform,
            audio_sample_rate=audio_sample_rate,
            max_pending=max_pending,
            video_codec=video_codec,
            audio_codec=audio_codec,
            crf=crf,
            video_codec_options=video_codec_options,
            num_segments=num_segments,
            segment_ratio=segment_ratio,
        )
    raise ValueError(
        f"unknown encoder backend {backend!r}; expected one of "
        "pyav, ffmpeg, segmented_ffmpeg"
    )
