import sys
import time


class MultiStepProgress:
    """Small terminal progress helper for long evaluation jobs."""

    def __init__(self, total_steps=6, step_names=None, log_file=None):
        self.total_steps = total_steps
        self.current_step = 0
        self.step_start_times = {}
        self.step_durations = {}
        self.overall_start = time.time()
        self.log_file = log_file
        self.step_names = step_names or [
            "Load data and model",
            "Extract features",
            "Calibrate matcher",
            "Compute distance matrix",
            "Compute metrics",
            "Save and print results",
        ]

        if len(self.step_names) < total_steps:
            for i in range(len(self.step_names), total_steps):
                self.step_names.append(f"Step {i + 1}")

    def _log(self, msg):
        print(msg)
        sys.stdout.flush()
        if self.log_file:
            try:
                self.log_file.write(msg + "\n")
                self.log_file.flush()
            except Exception:
                pass

    def _format_time(self, seconds):
        if seconds < 0:
            seconds = 0
        if seconds < 60:
            return f"{seconds:.0f}s"
        if seconds < 3600:
            minutes, secs = divmod(int(seconds), 60)
            return f"{minutes}m{secs:02d}s"
        hours, minutes = divmod(int(seconds), 3600)
        minutes, secs = divmod(minutes, 60)
        return f"{hours}h{minutes:02d}m{secs:02d}s"

    def _draw_bar(self, current, total, width=40):
        total = max(total, 1)
        percent = min(max(current / total, 0.0), 1.0)
        filled = int(width * percent)
        bar = "#" * filled + "-" * (width - filled)
        return f"[{bar}] {percent * 100:5.1f}%"

    def start_step(self, step_idx, description=""):
        self.current_step = step_idx
        self.step_start_times[step_idx] = time.time()

        self._log("")
        self._log("=" * 80)
        self._log(f"Step [{step_idx + 1}/{self.total_steps}] {self.step_names[step_idx]}")
        if description:
            self._log(f"Info: {description}")
        self._log("=" * 80)

    def update_substep(self, step_idx, sub_current, sub_total, message=""):
        if step_idx not in self.step_start_times:
            self.start_step(step_idx)

        elapsed = time.time() - self.step_start_times[step_idx]
        if sub_current > 0:
            speed = sub_current / elapsed
            eta = (sub_total - sub_current) / speed if speed > 0 else 0
        else:
            eta = 0

        status = (
            f"  [progress] {self._draw_bar(sub_current, sub_total)} | "
            f"{sub_current}/{sub_total} | elapsed: {self._format_time(elapsed)} | "
            f"eta: {self._format_time(eta)}"
        )
        if message:
            status += f" | {message}"
        self._log(status)

    def complete_step(self, step_idx, summary=""):
        if step_idx not in self.step_start_times:
            return

        duration = time.time() - self.step_start_times[step_idx]
        self.step_durations[step_idx] = duration
        self._log(
            f"  [done] Step [{step_idx + 1}/{self.total_steps}] "
            f"time: {self._format_time(duration)}"
        )
        if summary:
            self._log(f"  [summary] {summary}")

        overall_elapsed = time.time() - self.overall_start
        completed = sum(1 for i in range(self.total_steps) if i in self.step_durations)
        remaining = self.total_steps - completed
        avg_time = overall_elapsed / max(completed, 1)
        eta = avg_time * remaining
        self._log(
            f"  [overall] completed: {completed}/{self.total_steps} | "
            f"total: {self._format_time(overall_elapsed)} | "
            f"remaining: {self._format_time(eta)}"
        )
        self._log("")

    def print_summary(self):
        self._log("")
        self._log("=" * 80)
        self._log("Run summary")
        self._log("=" * 80)
        self._log(f"Total time: {self._format_time(time.time() - self.overall_start)}")
        self._log("")

        for i in range(self.total_steps):
            if i in self.step_durations:
                status = "[done]"
                duration = self._format_time(self.step_durations[i])
            elif i in self.step_start_times:
                status = "[running]"
                duration = "..."
            else:
                status = "[pending]"
                duration = "-"
            self._log(f"  Step [{i + 1}/{self.total_steps}] {self.step_names[i]}: {status} ({duration})")

        self._log("=" * 80)


class SimpleProgressBar:
    """Rate-limited progress bar for a single loop."""

    def __init__(self, total, desc="Progress", width=40, file=None):
        self.total = total
        self.desc = desc
        self.width = width
        self.file = file
        self.start_time = time.time()
        self.last_update = 0

    def _log(self, msg):
        print(msg)
        sys.stdout.flush()
        if self.file:
            try:
                self.file.write(msg + "\n")
                self.file.flush()
            except Exception:
                pass

    def update(self, current, message=""):
        now = time.time()
        if now - self.last_update < 0.5 and current < self.total:
            return
        self.last_update = now

        elapsed = now - self.start_time
        percent = min(max(current / max(self.total, 1), 0.0), 1.0)
        filled = int(self.width * percent)
        bar = "#" * filled + "-" * (self.width - filled)

        if current > 0 and elapsed > 0:
            speed = current / elapsed
            eta = (self.total - current) / speed if speed > 0 else 0
            eta_str = self._format_time(eta)
        else:
            eta_str = "?"

        status = (
            f"  {self.desc} [{bar}] {current}/{self.total} "
            f"({percent * 100:.1f}%) | elapsed: {self._format_time(elapsed)} | "
            f"eta: {eta_str}"
        )
        if message:
            status += f" | {message}"
        self._log(status)

    def _format_time(self, seconds):
        if seconds < 0 or seconds == "?":
            return "?"
        if seconds < 60:
            return f"{seconds:.0f}s"
        if seconds < 3600:
            minutes, secs = divmod(int(seconds), 60)
            return f"{minutes}m{secs:02d}s"
        hours, minutes = divmod(int(seconds), 3600)
        minutes, secs = divmod(minutes, 60)
        return f"{hours}h{minutes:02d}m"

    def finish(self, message="done"):
        self.update(self.total, message)
