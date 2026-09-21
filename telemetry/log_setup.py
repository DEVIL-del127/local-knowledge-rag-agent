# log_setup.py - 系统日志落盘配置(入口统一调用)
# 需求 R7: INFO 全量按天轮转 + WARNING 单独文件 + 磁盘水位告警
from __future__ import annotations

import logging
import os
from logging.handlers import TimedRotatingFileHandler


def setup_app_logging(
    log_dir: str = "logs/app",
    level: int = logging.INFO,
    console: bool = True,
) -> None:
    """配置应用日志: 落盘 + 控制台(幂等, 可重复调用)

    logs/app/
      ├─ app-2026-08-18.log     # INFO 全量, 按天轮转(保留 30 天)
      └─ warn-2026-08-18.log    # WARNING+ 单独文件(排障只看告警)
    """
    os.makedirs(log_dir, exist_ok=True)
    root = logging.getLogger()
    if root.handlers:  # 已配置过(幂等)
        return

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    info_path = os.path.join(log_dir, "app.log")
    info_handler = TimedRotatingFileHandler(
        info_path, when="midnight", backupCount=30, encoding="utf-8"
    )
    info_handler.setFormatter(fmt)
    info_handler.setLevel(level)
    root.addHandler(info_handler)

    warn_path = os.path.join(log_dir, "warn.log")
    warn_handler = TimedRotatingFileHandler(
        warn_path, when="midnight", backupCount=30, encoding="utf-8"
    )
    warn_handler.setFormatter(fmt)
    warn_handler.setLevel(logging.WARNING)
    root.addHandler(warn_handler)

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(fmt)
        console_handler.setLevel(level)
        root.addHandler(console_handler)

    root.setLevel(level)
    logging.getLogger(__name__).info("应用日志已落盘: %s", os.path.abspath(log_dir))


def check_log_disk_watermark(log_dir: str = "logs/app", warn_threshold: float = 0.8) -> bool:
    """检查日志目录所在磁盘使用率, 超阈值告警; 返回是否超阈值

    平台差异: Windows 用 shutil.disk_usage 的当前盘; 失败静默返回 False。
    """
    try:
        import shutil

        usage = shutil.disk_usage(os.path.abspath(log_dir))
        ratio = usage.used / usage.total
        if ratio >= warn_threshold:
            logging.getLogger(__name__).warning(
                "磁盘使用率 %.1f%% 超过阈值 %.0f%% (可用 %d GB)",
                ratio * 100, warn_threshold * 100, usage.free // (1024**3),
            )
            return True
    except OSError:
        pass
    return False
