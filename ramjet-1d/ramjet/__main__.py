"""python3 -m ramjet — le meme point d'entree que main.py."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
