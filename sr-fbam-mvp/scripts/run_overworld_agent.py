#!/usr/bin/env python3
#!/usr/bin/env python3
"""
Compatibility wrapper that delegates to the packaged CLI.
"""

from srfbam.cli.overworld_agent import main


if __name__ == "__main__":
    raise SystemExit(main())
