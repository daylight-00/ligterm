"""`python -m ligterm score ...`: the command line of LigTerm (see `ligterm.cli`)."""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
