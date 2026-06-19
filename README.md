<!--
BADGES (ONLY FOR PUBLIC REPOS)
<p align="center">
<a href="https://github.com/antodiazcano/template-project/actions/workflows/ci.yml">
  <img src="https://github.com/antodiazcano/template-project/actions/workflows/ci.yml/badge.svg" alt="CI">
</a>
  <img src="https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12%20|%203.13-blue">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
</p>
-->

# template-project

## What to do at first

First, create the virtual environment (the ".venv" part is to use that name for the virtual environment, but you can choose what you want):

    uv venv .venv

(Optional) If you want to select the Python version in the virtual environment:

    uv python pin 3.13.1
    uv python install 3.13.1
    uv venv --python 3.13.1

Then, activate it:

    source .venv/bin/activate

Use ```Ctrl``` + ```Shift``` + ```p``` to select the interpreter and run

    uv sync

to install the predefined packages.

Finally, change "name" and "description" in the ```pyproject.toml```.

## Dealing with dependencies

To add a package, just execute:

    uv add package

To remove a package, just execute:

    uv remove package

Each time a new package is installed/removed it is automatically reflected in the ```pyproject.toml```.

If we want just a group of dependencies you can do:

    uv sync --group <name of the group>

To include a dependency in a group, you can do:

    uv add package --group

You can ignore Pylint, Mypy or any other package rules in the ```pyproject.toml```.

## Saving Tokens with LLMs

- Use [rtk](https://github.com/rtk-ai/rtk):

        # 1. Install for your AI tool
        rtk init -g                     # Claude Code / Copilot (default)
        rtk init -g --gemini            # Gemini CLI
        rtk init -g --codex             # Codex (OpenAI)
        rtk init -g --agent cursor      # Cursor
        rtk init --agent windsurf       # Windsurf
        rtk init --agent cline          # Cline / Roo Code
        rtk init --agent kilocode       # Kilo Code
        rtk init --agent antigravity    # Google Antigravity
        
        # 2. Restart your AI tool, then test
        git status  # Automatically rewritten to rtk git status

- Use [Caveman mode](https://github.com/om-patel5/Caveman-Claude).

- Use lightweight models for some specific tasks like context compression. This [video](https://www.youtube.com/watch?v=NoF-YajElIM) explains it.

## Sanity Checks

With the ```Makefile``` you can use

    make install

to install the dependencies,

    make format

to run Black and format the code,

    make lint

to check the code with Ruff, Bandit, Mypy, Flake8, Complexipy and Pylint,

    make test
    
to run the tests,

    make clean

to delete "trash" directories like ```__pycache__``` and

    make all

to run install, format, lint, test and clean sequentially with just one command.

By default, Pylint only fails under a mark of 8 and no complete coverage is required, but it is a good practice to check periodically if a 10/10 mark is achieved in Pylint and a 100% test coverage is achieved in coverage. To see what lines are not covered, just execute

    pytest --cov-report term-missing

and they will be shown.

## Others

To see the documentation run:

    mkdocs serve
