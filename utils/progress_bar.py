"""
进度条工具模块：为6个测试步骤提供可视化进度显示
支持多步骤进度跟踪、ETA计算、强制刷新
"""
import sys
import time
from datetime import timedelta


class MultiStepProgress:
    """
    多步骤进度管理器
    提供6个测试步骤的可视化进度跟踪
    """

    def __init__(self, total_steps=6, step_names=None, log_file=None):
        self.total_steps = total_steps
        self.current_step = 0
        self.step_start_times = {}
        self.step_durations = {}
        self.overall_start = time.time()
        self.log_file = log_file

        self.step_names = step_names or [
            "加载数据集",
            "加载预训练模型",
            "提取特征向量",
            "训练匹配器",
            "计算评价指标",
            "输出最终结果"
        ]

        if len(self.step_names) < total_steps:
            for i in range(len(self.step_names), total_steps):
                self.step_names.append(f"步骤 {i+1}")

    def _log(self, msg):
        print(msg)
        sys.stdout.flush()
        if self.log_file:
            try:
                self.log_file.write(msg + '\n')
                self.log_file.flush()
            except:
                pass

    def _format_time(self, seconds):
        if seconds < 0:
            seconds = 0
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            m, s = divmod(int(seconds), 60)
            return f"{m}m{s:02d}s"
        else:
            h, m = divmod(int(seconds), 3600)
            m, s = divmod(m, 60)
            return f"{h}h{m:02d}m{s:02d}s"

    def _draw_bar(self, current, total, width=40):
        if total <= 0:
            total = 1
        percent = current / total
        filled = int(width * percent)
        bar = '#' * filled + '-' * (width - filled)
        return f"[{bar}] {percent*100:5.1f}%"

    def start_step(self, step_idx, description=""):
        self.current_step = step_idx
        self.step_start_times[step_idx] = time.time()

        self._log("")
        self._log("=" * 80)
        self._log(f"步骤 [{step_idx+1}/{self.total_steps}] {self.step_names[step_idx]}")
        if description:
            self._log(f"说明: {description}")
        self._log("=" * 80)

    def update_substep(self, step_idx, sub_current, sub_total, message=""):
        if step_idx not in self.step_start_times:
            self.start_step(step_idx)

        elapsed = time.time() - self.step_start_times[step_idx]
        if sub_current > 0:
            speed = sub_current / elapsed
            remaining_items = sub_total - sub_current
            eta = remaining_items / speed if speed > 0 else 0
        else:
            eta = 0

        bar = self._draw_bar(sub_current, sub_total)
        elapsed_str = self._format_time(elapsed)
        eta_str = self._format_time(eta)

        status = f"  [进度] {bar} | {sub_current}/{sub_total} | 已用: {elapsed_str} | 剩余: {eta_str}"
        if message:
            status += f" | {message}"

        self._log(status)

    def complete_step(self, step_idx, summary=""):
        if step_idx in self.step_start_times:
            duration = time.time() - self.step_start_times[step_idx]
            self.step_durations[step_idx] = duration
            duration_str = self._format_time(duration)

            self._log(f"  [完成] 步骤 [{step_idx+1}/{self.total_steps}] 耗时: {duration_str}")
            if summary:
                self._log(f"  [摘要] {summary}")

            overall_elapsed = time.time() - self.overall_start
            overall_str = self._format_time(overall_elapsed)
            completed = sum(1 for i in range(self.total_steps) if i in self.step_durations)
            remaining = self.total_steps - completed
            avg_time = overall_elapsed / max(completed, 1)
            eta = avg_time * remaining

            self._log(f"  [总览] 已完成: {completed}/{self.total_steps} | 总耗时: {overall_str} | 预计剩余: {self._format_time(eta)}")
            self._log("")

    def print_summary(self):
        self._log("")
        self._log("=" * 80)
        self._log("运行摘要")
        self._log("=" * 80)

        total_duration = time.time() - self.overall_start
        self._log(f"总耗时: {self._format_time(total_duration)}")
        self._log("")

        for i in range(self.total_steps):
            if i in self.step_durations:
                status = "[完成]"
                duration = self._format_time(self.step_durations[i])
            elif i in self.step_start_times:
                status = "[运行中]"
                duration = "..."
            else:
                status = "[未开始]"
                duration = "-"
            self._log(f"  步骤 [{i+1}/{self.total_steps}] {self.step_names[i]}: {status} ({duration})")

        self._log("=" * 80)


class SimpleProgressBar:
    """
    简单进度条：用于单步骤内的循环迭代
    """

    def __init__(self, total, desc="进度", width=40, file=None):
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
                self.file.write(msg + '\n')
                self.file.flush()
            except:
                pass

    def update(self, current, message=""):
        now = time.time()
        if now - self.last_update < 0.5 and current < self.total:
            return
        self.last_update = now

        elapsed = now - self.start_time
        percent = current / max(self.total, 1)
        filled = int(self.width * percent)
        bar = '#' * filled + '-' * (self.width - filled)

        if current > 0:
            speed = current / elapsed
            eta = (self.total - current) / speed if speed > 0 else 0
            eta_str = self._format_time(eta)
        else:
            eta_str = "?"

        elapsed_str = self._format_time(elapsed)
        status = f"  {self.desc} [{bar}] {current}/{self.total} ({percent*100:.1f}%) | 用时: {elapsed_str} | 剩余: {eta_str}"
        if message:
            status += f" | {message}"
        self._log(status)

    def _format_time(self, seconds):
        if seconds < 0 or seconds == "?":
            return "?"
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            m, s = divmod(int(seconds), 60)
            return f"{m}m{s:02d}s"
        else:
            h, m = divmod(int(seconds), 3600)
            m, s = divmod(m, 60)
            return f"{h}h{m:02d}m"

    def finish(self, message="完成"):
        self.update(self.total, message)