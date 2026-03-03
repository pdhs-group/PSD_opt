# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 11:09:33 2025

@author: px2030
"""
import sys
import time

def print_highlighted(message, title=None, color="yellow", separator=True, timestamp=True, width=80):
    """
    Print a highlighted message with optional color, timestamp, and separator.

    Parameters:
        message (str): The message to print.
        title (str, optional): Title for the message (e.g., "WARNING", "INFO").
        color (str, optional): Color of the message ("red", "green", "yellow", "blue", "cyan", etc.).
        separator (bool, optional): Whether to print a separator line before the message.
        timestamp (bool, optional): Whether to include a timestamp.
        width (int, optional): The width of the separator line.

    Colors supported:
        - "red", "green", "yellow", "blue", "magenta", "cyan", "white"
    """

    # ANSI color codes
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "reset": "\033[0m"
    }
    
    color_code = colors.get(color.lower(), colors["yellow"])
    
    # Build the output string
    output = ""
    
    if separator:
        output += "=" * width + "\n"  # Print a separator line
    
    if timestamp:
        time_str = time.strftime("[%Y-%m-%d %H:%M:%S]")
        output += f"{time_str} "
    
    if title:
        output += f"[{title.upper()}] "

    output += f"{color_code}{message}{colors['reset']}"  # Apply color formatting

    print(output, file=sys.stdout)