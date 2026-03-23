"""AIOpsLab CLI"""
import click
from rich.console import Console
from aiopslab.core.framework import AIOpsLab

console = Console()


@click.group()
def main():
    """AIOpsLab - Framework for AIOps Research and Experimentation"""
    pass


@main.command()
@click.option("--name", default="aiopslab", help="Cluster name")
@click.option("--config", help="Kind config file")
def setup(name, config):
    """Setup kind cluster"""
    console.print(f"[bold blue]Setting up cluster: {name}[/bold blue]")
    lab = AIOpsLab()
    lab.setup_cluster(name)
    console.print("[bold green]✓ Cluster ready[/bold green]")


@main.command()
@click.argument("config_file")
def run(config_file):
    """Run experiment from config file"""
    import yaml
    
    console.print(f"[bold blue]Running experiment: {config_file}[/bold blue]")
    
    with open(config_file) as f:
        config = yaml.safe_load(f)
    
    lab = AIOpsLab(config.get("framework"))
    results = lab.run_experiment(config.get("experiment"))
    
    console.print("[bold green]✓ Experiment completed[/bold green]")
    console.print(results)


@main.command()
def cleanup():
    """Cleanup all resources"""
    console.print("[bold yellow]Cleaning up resources...[/bold yellow]")
    lab = AIOpsLab()
    lab.cleanup()
    console.print("[bold green]✓ Cleanup complete[/bold green]")


@main.command()
def list_datasets():
    """List available datasets"""
    lab = AIOpsLab()
    datasets = lab.dataset_adapter.list_available()
    
    console.print("[bold]Available datasets:[/bold]")
    for ds in datasets:
        console.print(f"  • {ds}")


if __name__ == "__main__":
    main()
