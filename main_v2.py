"""
Entry point for the AI Natural Language Terminal v2.0.
Runs an interactive CLI loop that accepts English or Manglish input, retrieves RAG context,
generates bash command candidates via T5 + Ollama, and executes the user-chosen command.
"""

import os
import logging
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich import box
from memory import TerminalMemory
from translator import CorrectedTerminalTranslator
from executor import TerminalSession, is_dangerous
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(
    filename="terminal.log",
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def _load_malayalam():
    try:
        from native import MalayalamTranslator
        return MalayalamTranslator()
    except ImportError:
        return None
    except Exception as e:
        print(f"[Malayalam] Failed to load: {e}")
        return None

def _show_candidates(
    console: Console,
    candidates: list[str],
    descriptions: dict[str, str] | None = None,
) -> str | None:
    descs = descriptions or {}

    if len(candidates) == 1:
        cmd  = candidates[0]
        desc = descs.get(cmd, "")
        suffix = f"  [dim]{desc}[/dim]" if desc else ""
        console.print(
            f"\n[cyan]AI Suggests:[/cyan] [bold white on blue] {cmd} [/bold white on blue]{suffix}"
        )
        return cmd

    table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
    table.add_column("#",           style="bold yellow", width=3)
    table.add_column("Command",     style="bold white on blue")
    table.add_column("Description", style="dim")
    for i, cmd in enumerate(candidates, 1):
        table.add_row(str(i), f" {cmd} ", descs.get(cmd, ""))

    console.print("\n[cyan]AI Suggestions:[/cyan]")
    console.print(table)

    choices = [str(i) for i in range(1, len(candidates) + 1)] + ["n"]
    try:
        pick = Prompt.ask("Choose a command", choices=choices, default="1")
    except KeyboardInterrupt:
        console.print("\n[yellow]Skipped.[/yellow]")
        return None
    return None if pick == "n" else candidates[int(pick) - 1]

def main():
    console = Console()

    mal = _load_malayalam()
    mal_status = "[green]Malayalam ON[/green]" if mal else "[dim]Malayalam OFF[/dim]"

    console.print(Panel(
        "[bold cyan]AI NATURAL LANGUAGE TERMINAL v2.0[/bold cyan]\n"
        "[italic]RAG + T5 + Ollama (llama3.2) Correction[/italic]\n"
        f"{mal_status}\n"
        "[dim]Type 'exit' or 'quit' to leave  ·  Ctrl+C to cancel[/dim]"
    ))

    try:
        memory     = TerminalMemory("LINUX_TERMINAL_COMMANDS_CLEANED.jsonl")
        translator = CorrectedTerminalTranslator()
        session    = TerminalSession()
    except Exception as e:
        console.print(f"[bold red]Startup error:[/bold red] {e}")
        return

    while True:
        try:
            folder     = os.path.basename(session.cwd) or "/"
            user_input = Prompt.ask(
                f"\n[bold green]({folder})[/bold green] Instruction"
            ).strip()
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Type 'exit' to quit.[/yellow]")
            continue

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            console.print("[bold cyan]Goodbye![/bold cyan]")
            break

        english_input = user_input
        if mal and mal.is_manglish(user_input):
            try:
                mal_script, english_input = mal.translate(user_input)
                console.print(
                    f"[dim] Malayalam:[/dim] {mal_script}\n"
                    f"[dim]English  :[/dim] {english_input}"
                )
                logging.info(f"Malayalam | Input: {user_input!r} → {english_input!r}")
            except Exception as e:
                console.print(f"[yellow]Translation failed, using input as-is: {e}[/yellow]")
                english_input = user_input

        matches = memory.get_context(english_input)

        candidates = translator.generate_candidates(english_input, matches, n=2)

        if not candidates:
            console.print("[bold red]No command generated. Please rephrase.[/bold red]")
            logging.warning(f"Empty generation | Input: {english_input!r}")
            continue

        descriptions = getattr(translator, "descriptions", {})
        cmd = _show_candidates(console, candidates, descriptions)
        if cmd is None:
            console.print("[yellow]Skipped.[/yellow]")
            continue

        if is_dangerous(cmd):
            console.print("[bold red]⚠  WARNING: Potentially destructive command![/bold red]")
            try:
                confirm = Prompt.ask("Are you SURE?", choices=["yes", "no"], default="no")
            except KeyboardInterrupt:
                console.print("\n[yellow]Cancelled.[/yellow]")
                continue
            if confirm != "yes":
                console.print("[yellow]Command cancelled.[/yellow]")
                continue

        try:
            execute = Prompt.ask("Execute?", choices=["y", "n"], default="y")
        except KeyboardInterrupt:
            console.print("\n[yellow]Skipped.[/yellow]")
            continue

        if execute == "y":
            output = session.run(cmd)
            logging.info(f"Input: {user_input!r} | Cmd: {cmd!r} | Dir: {session.cwd}")
            console.print(Panel(output, title="Output", border_style="green"))

            pass

if __name__ == "__main__":
    main()
