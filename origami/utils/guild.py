import os
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional

from dotenv import dotenv_values
from guild.commands import runs_impl
from guild.ipy import RunsDataFrame, RunsSeries, _runs_cmd_args
from guild.run import Run
from matplotlib import pyplot as plt


def print_guild_scalars(**kwargs) -> None:
    """Prints scalars in a nice column format. To parse the scalars, use the following
    output-scalars definition in the guild.yml file:

    output-scalars:
      - step: '\|  step: (\step)'
      - '\|  (\key): (\value)'

    """
    for key, val in kwargs.items():
        print(f"|  {key}: {val}", end="  ")
    print("|")


def _plot_scalar_series(ax: plt.Axes, run: RunsSeries, scalar: str, label_fn: Callable, **kwargs) -> None:
    scalars = run.scalars_detail().query(f"tag == '{scalar}'")
    steps = scalars["step"]
    loss = scalars["val"]
    if "label" not in kwargs:
        kwargs["label"] = label_fn(run)
    ax.plot(steps, loss, **kwargs)


def _make_label(run: RunsSeries, flags: list, include_run_id: bool = True) -> str:
    if not flags:
        return run.run
    guild_flags = run.guild_flags().iloc[0].to_dict()
    s = [f"{flag}={guild_flags[flag]}" for flag in flags]
    if include_run_id:
        return f"{run.run}: " + ", ".join(s)
    return ", ".join(s)


def plot_scalar_history(
    runs: RunsSeries | RunsDataFrame,
    scalar: str = "train_loss",
    label_flags: list = None,
    plot_means: bool = False,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> None:
    """Plots a matplotlib plot of the given run(s) and scalar. If label_flags are given,
    annotates the runs with the flags and their values in the legend. Otherwise it uses
    the run ID as label.

    if plot_means is True, it plots the individual runs in a semi-transparent style and
    then plots the mean of the runs in a solid style.

    Returns the figure and axes objects from matplotlib
    """
    if fig is None:
        fig, ax = plt.subplots()
        fig.set_size_inches((12, 8))
        fig.set_dpi(300)

    if isinstance(runs, RunsSeries):
        _plot_scalar_series(ax, runs, scalar, lambda r: _make_label(r, label_flags), **kwargs)
        ax.set_xlabel("steps")
        ax.set_ylabel(scalar)
        ax.set_title(f"{scalar} for guild run {runs.run}")

    elif isinstance(runs, RunsDataFrame):
        if plot_means:
            # plot individual runs with thin line, no labels (recursively)
            kw = kwargs.copy()
            kw.update({"linewidth": 0.5, "alpha": 0.3, "label": None})
            plot_scalar_history(runs, scalar, fig=fig, ax=ax, **kw)

            # calculate mean of runs
            scalars = runs.scalars_detail()
            scalars = scalars[scalars["tag"] == scalar]
            scalars = scalars.groupby(["step"]).mean(numeric_only=True)

            # plot mean of runs with thick line and labels
            kw = kwargs.copy()
            kw.update({"linewidth": 2, "alpha": 1.0})
            ax.plot(
                scalars.index, scalars["val"], label=_make_label(runs.iloc[0], label_flags, include_run_id=False), **kw
            )
        else:
            for _, row in runs.iterrows():
                _plot_scalar_series(ax, row, scalar, lambda r: _make_label(r, label_flags), **kwargs)

            ax.legend(title="runs")
            ax.set_xlabel("steps")
            ax.set_ylabel(scalar)

    ax.set_title(f"{scalar} for multiple runs")
    ax.legend(title="runs")

    return fig, ax


def detect_remote() -> bool:
    """returns True if this experiment runs on GCP."""

    # TODO find better way of detecting remote execution that works for all clouds
    return os.environ.get("USER", None) == "gcpuser"


def load_secrets():
    """loads secrets from .env.local or .env.remote depending on where the experiment
    is executed."""

    env = "remote" if detect_remote() else "local"
    secrets = dotenv_values(f".env.{env}")
    print(f"\nloading {env} secrets.\n")
    return secrets


def get_runs(**kw) -> list[Run]:
    """get access to runs objects based on filters."""
    return runs_impl.filtered_runs(_runs_cmd_args(**kw))


def get_run(**kw) -> Run:
    """get access to the first matching run object based on filters."""
    runs = get_runs(**kw)
    assert len(runs) > 0, "No matching runs found."
    return runs[0]


def get_run_path(**kw) -> Path:
    """returns the path of a run given various filters. If multiple runs match, it returns
    the path of the latest run."""

    run = get_run(**kw)
    return Path(run.dir)


def get_run_flags(**kw) -> SimpleNamespace:
    """returns the path of a run given various filters. If multiple runs match, it returns
    the path of the latest run."""

    run = get_run(**kw)
    return SimpleNamespace(**run["flags"])
