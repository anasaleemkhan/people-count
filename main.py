from __future__ import annotations

import ctypes
import sys
import tkinter as tk

from app import PeopleCountApp


def enable_windows_dpi_awareness() -> None:
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def main() -> None:
    enable_windows_dpi_awareness()
    root = tk.Tk()
    PeopleCountApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
