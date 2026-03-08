"""Jupyter notebook callback for live training visualization.

Displays interactive progress bars and live metric plots using ipywidgets.
Uses Plotly FigureWidget for flicker-free interactive charts when available,
falls back to matplotlib otherwise.

Example:
    ```python
    from origami.training import OrigamiTrainer, NotebookCallback

    trainer = OrigamiTrainer(
        model=model,
        tokenizer=tokenizer,
        train_data=train_data,
        eval_data=eval_data,
        callbacks=[NotebookCallback(metrics=["loss", "val_loss", "lr", "val_acc"])],
    )
    trainer.train()
    ```
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from .callbacks import TrainerCallback

if TYPE_CHECKING:
    from .trainer import EpochStats, OrigamiTrainer, TrainResult

# Batch-level metrics extracted from TrainResult state on every step.
# Maps metric name to the attribute on TrainResult.
_BATCH_METRICS: dict[str, str] = {
    "loss": "current_batch_loss",
    "lr": "current_lr",
    "batch_dt": "current_batch_dt",
}


class _NotebookCallbackBase(TrainerCallback):
    """Shared logic for notebook callbacks (progress bars, data recording, EMA).

    Subclasses must implement ``_init_plot()`` and ``_update_plot()``.
    """

    def __init__(
        self,
        metrics: list[str] | None = None,
        plot_update_interval: int = 10,
        smoothing: float = 0.9,
    ) -> None:
        self.metrics = metrics or ["loss"]
        self.plot_update_interval = plot_update_interval
        self.smoothing = smoothing

        # Per-metric data: metric_name -> list of (step, value)
        self._series: dict[str, list[tuple[int, float]]] = {m: [] for m in self.metrics}
        # EMA state for batch-level metrics
        self._ema: dict[str, float | None] = {}

        # Widgets (created on train_begin)
        self._widget = None

    def on_train_begin(
        self,
        trainer: OrigamiTrainer,
        state: TrainResult,
        payload: Any,
    ) -> None:
        if not trainer.is_main_process:
            return

        import ipywidgets as widgets
        from IPython.display import display

        # Overall progress bar
        self._overall_progress = widgets.IntProgress(
            value=state.epoch,
            min=0,
            max=trainer.config.num_epochs,
            bar_style="info",
            layout=widgets.Layout(flex="1"),
        )
        self._overall_label = widgets.HTML(
            value=self._format_overall_label(state.epoch, trainer.config.num_epochs),
            layout=widgets.Layout(width="160px", flex="0 0 auto"),
        )

        # Epoch progress bar
        self._epoch_progress = widgets.IntProgress(
            value=0,
            min=0,
            max=max(1, trainer.steps_per_epoch),
            bar_style="",
            layout=widgets.Layout(flex="1"),
        )
        self._epoch_label = widgets.HTML(
            value="",
            layout=widgets.Layout(width="160px", flex="0 0 auto"),
        )

        # Status line
        self._status = widgets.HTML(value="")

        # Create plot (subclass sets self._plot_container)
        self._init_plot()

        # Assemble layout
        self._widget = widgets.VBox(
            [
                widgets.HBox([self._overall_label, self._overall_progress]),
                widgets.HBox([self._epoch_label, self._epoch_progress]),
                self._status,
                self._plot_container,
            ],
            layout=widgets.Layout(width="100%", overflow_x="hidden"),
        )

        display(self._widget)

    def on_epoch_begin(
        self,
        trainer: OrigamiTrainer,
        state: TrainResult,
        payload: Any,
    ) -> None:
        if not trainer.is_main_process or self._widget is None:
            return

        self._epoch_progress.max = max(1, trainer.steps_per_epoch)
        self._epoch_progress.value = state.epoch_resume_step
        self._epoch_label.value = self._format_epoch_label(
            state.epoch + 1, state.epoch_resume_step, trainer.steps_per_epoch
        )

    def on_batch_end(
        self,
        trainer: OrigamiTrainer,
        state: TrainResult,
        payload: Any,
    ) -> None:
        if not trainer.is_main_process or self._widget is None:
            return

        # Record batch-level metrics
        step = state.global_step
        for metric_name in self.metrics:
            attr = _BATCH_METRICS.get(metric_name)
            if attr is None:
                continue
            raw = getattr(state, attr)
            smoothed = self._smooth(metric_name, raw)
            self._series[metric_name].append((step, smoothed))

        # Update epoch progress
        batch_in_epoch = state.epoch_step - state.epoch_resume_step
        self._epoch_progress.value = min(batch_in_epoch, self._epoch_progress.max)
        self._epoch_label.value = self._format_epoch_label(
            state.epoch + 1, batch_in_epoch, trainer.steps_per_epoch
        )

        # Update status
        self._status.value = (
            f"<code>step: {state.global_step}/{trainer.total_steps} | "
            f"loss: {state.current_batch_loss:.4f} | "
            f"lr: {state.current_lr:.2e} | "
            f"batch: {state.current_batch_dt * 1000:.0f}ms</code>"
        )

        # Update plot periodically
        if step % self.plot_update_interval == 0:
            self._update_plot()

    def on_epoch_end(
        self,
        trainer: OrigamiTrainer,
        state: TrainResult,
        payload: EpochStats | None,
    ) -> None:
        if not trainer.is_main_process or self._widget is None:
            return

        self._overall_progress.value = state.epoch + 1
        self._overall_label.value = self._format_overall_label(
            state.epoch + 1, trainer.config.num_epochs
        )
        self._epoch_progress.value = self._epoch_progress.max
        self._update_plot()

    def on_evaluate(
        self,
        trainer: OrigamiTrainer,
        state: TrainResult,
        payload: dict[str, float],
    ) -> None:
        if not trainer.is_main_process or self._widget is None:
            return
        if not payload:
            return

        step = state.global_step
        for metric_name in self.metrics:
            if metric_name in payload:
                self._series[metric_name].append((step, payload[metric_name]))

        self._update_plot()

    def on_train_end(
        self,
        trainer: OrigamiTrainer,
        state: TrainResult,
        payload: Any,
    ) -> None:
        if not trainer.is_main_process or self._widget is None:
            return

        self._update_plot()

        if state.completed:
            self._overall_progress.bar_style = "success"
            self._status.value = "<code>Training complete.</code>"
        elif state.interrupted:
            self._overall_progress.bar_style = "warning"
            self._status.value = (
                f"<code>Training interrupted at epoch {state.epoch + 1}, "
                f"step {state.global_step}.</code>"
            )

    def on_interrupt(
        self,
        trainer: OrigamiTrainer,
        state: TrainResult,
        payload: Any,
    ) -> None:
        # on_train_end handles the UI update
        pass

    # --- EMA smoothing ---

    def _smooth(self, key: str, value: float) -> float:
        """Apply exponential moving average smoothing."""
        prev = self._ema.get(key)
        if prev is None:
            self._ema[key] = value
            return value
        smoothed = self.smoothing * prev + (1 - self.smoothing) * value
        self._ema[key] = smoothed
        return smoothed

    # --- Subclass hooks ---

    def _init_plot(self) -> None:
        """Create the plot widget. Must set ``self._plot_container``."""
        raise NotImplementedError

    def _update_plot(self) -> None:
        """Update the plot with current data from ``self._series``."""
        raise NotImplementedError

    # --- Label formatters ---

    @staticmethod
    def _format_overall_label(current: int, total: int) -> str:
        return f"<b style='width:120px;display:inline-block'>Epoch {current}/{total}</b>"

    @staticmethod
    def _format_epoch_label(epoch: int, batch: int, total: int) -> str:
        return (
            f"<span style='width:120px;display:inline-block'>"
            f"Epoch {epoch} batch {batch}/{total}</span>"
        )


# ---------------------------------------------------------------------------
# Plotly backend
# ---------------------------------------------------------------------------


class PlotlyNotebookCallback(_NotebookCallbackBase):
    """Plotly FigureWidget-based notebook callback.

    Uses in-place trace data updates for flicker-free rendering.
    Provides interactive hover tooltips and synchronized crosshairs
    across subplots.
    """

    @staticmethod
    def _sort_metrics(metrics: list[str]) -> list[str]:
        """Sort metrics so loss and val_loss come first (top row)."""
        priority = {"loss": 0, "val_loss": 1}
        return sorted(metrics, key=lambda m: (priority.get(m, 2), m))

    def _init_plot(self) -> None:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Sort so loss/val_loss are in the top row
        ordered = self._sort_metrics(self.metrics)

        n_metrics = len(ordered)
        ncols = min(2, n_metrics)
        nrows = math.ceil(n_metrics / ncols)

        fig = make_subplots(
            rows=nrows,
            cols=ncols,
            shared_xaxes="all",
            subplot_titles=ordered,
            vertical_spacing=0.22 if nrows > 1 else 0.2,
            horizontal_spacing=0.10,
        )

        # Pre-create one trace per metric
        self._trace_index: dict[str, int] = {}
        # Track y-axis refs for loss metrics to link them
        loss_yaxis_ref: str | None = None

        for i, metric_name in enumerate(ordered):
            row, col = divmod(i, ncols)
            row += 1
            col += 1  # plotly is 1-indexed

            value_fmt = ".2e" if metric_name == "lr" else ".4f"
            trace = go.Scatter(
                x=[],
                y=[],
                mode="lines",
                name=metric_name,
                line={"width": 1.5},
                hovertemplate=(
                    f"<b>{metric_name}</b><br>"
                    f"step: %{{x}}<br>"
                    f"value: %{{y:{value_fmt}}}"
                    "<extra></extra>"
                ),
            )
            fig.add_trace(trace, row=row, col=col)
            self._trace_index[metric_name] = len(fig.data) - 1

            # LR y-axis: scientific notation
            if metric_name == "lr":
                fig.update_yaxes(exponentformat="e", row=row, col=col)

            # Link y-axes for loss and val_loss
            if metric_name in ("loss", "val_loss"):
                # Get the yaxis key for this subplot (e.g. "yaxis", "yaxis2", ...)
                yaxis_key = fig.get_subplot(row, col).yaxis.plotly_name
                if loss_yaxis_ref is None:
                    # First loss metric — this is the reference axis
                    loss_yaxis_ref = yaxis_key.replace("axis", "")
                else:
                    # Second loss metric — link to the first
                    fig.layout[yaxis_key].matches = loss_yaxis_ref

        # "step" label on all subplots, and force tick labels visible
        # (shared_xaxes hides them on non-bottom rows by default)
        fig.update_xaxes(title_text="step", showticklabels=True)

        # Hide unused subplots
        for i in range(n_metrics, nrows * ncols):
            row, col = divmod(i, ncols)
            fig.update_xaxes(visible=False, row=row + 1, col=col + 1)
            fig.update_yaxes(visible=False, row=row + 1, col=col + 1)

        # Global styling
        fig.update_layout(
            template="plotly_white",
            hovermode="x unified",
            showlegend=False,
            height=400 * nrows,
            margin={"l": 60, "r": 20, "t": 60, "b": 40},
            font={"size": 11},
            autosize=True,
        )

        # Spike lines for crosshair
        fig.update_xaxes(
            showspikes=True,
            spikemode="across",
            spikethickness=1,
            spikecolor="#999",
            spikedash="dot",
        )

        fig.update_yaxes(gridcolor="#eee")

        # Light gray frame around each subplot
        fig.update_xaxes(
            showline=True, linewidth=1, linecolor="#ddd", mirror=True,
        )
        fig.update_yaxes(
            showline=True, linewidth=1, linecolor="#ddd", mirror=True,
        )

        self._fig = go.FigureWidget(fig)
        self._plot_container = self._fig

    def _update_plot(self) -> None:
        with self._fig.batch_update():
            for metric_name, idx in self._trace_index.items():
                points = self._series.get(metric_name, [])
                if points:
                    xs, ys = zip(*points, strict=True)
                    self._fig.data[idx].x = xs
                    self._fig.data[idx].y = ys


# ---------------------------------------------------------------------------
# Matplotlib fallback backend
# ---------------------------------------------------------------------------


class MatplotlibNotebookCallback(_NotebookCallbackBase):
    """Matplotlib-based notebook callback (fallback when plotly is unavailable).

    Uses an ipywidgets Output widget with clear_output(wait=True) to
    re-render plots. May flicker on fast updates.
    """

    def _init_plot(self) -> None:
        import ipywidgets as widgets
        import matplotlib.pyplot as plt

        n_metrics = len(self.metrics)
        ncols = min(2, n_metrics)
        nrows = math.ceil(n_metrics / ncols)

        self._plot_output = widgets.Output()

        with plt.rc_context({
            "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7,
            "xtick.labelsize": 6, "ytick.labelsize": 6,
        }):
            self._fig, axes = plt.subplots(
                nrows, ncols, figsize=(4 * ncols, 2.5 * nrows), squeeze=False, dpi=300
            )
        self._fig.set_tight_layout(True)

        self._axes: dict[str, Any] = {}
        for i, metric_name in enumerate(self.metrics):
            row, col = divmod(i, ncols)
            ax = axes[row][col]
            ax.set_xlabel("step", fontsize=7)
            ax.set_title(metric_name, fontsize=8)
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.3)
            self._axes[metric_name] = ax

        for i in range(n_metrics, nrows * ncols):
            row, col = divmod(i, ncols)
            axes[row][col].set_visible(False)

        self._plot_container = self._plot_output
        plt.close(self._fig)

    def _update_plot(self) -> None:
        from IPython.display import display

        any_data = any(len(pts) > 0 for pts in self._series.values())
        if not any_data:
            return

        for metric_name, ax in self._axes.items():
            points = self._series.get(metric_name, [])
            ax.clear()
            ax.set_xlabel("step", fontsize=7)
            ax.set_title(metric_name, fontsize=8)
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.3)

            if not points:
                ax.text(
                    0.5, 0.5, "waiting for data...",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=7, color="gray",
                )
                continue

            xs, ys = zip(*points, strict=True)
            ax.plot(xs, ys, linewidth=1, alpha=0.8)

            if metric_name == "lr":
                ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
                ax.yaxis.get_offset_text().set_fontsize(6)

        self._plot_output.clear_output(wait=True)
        with self._plot_output:
            display(self._fig)


# ---------------------------------------------------------------------------
# Default alias: Plotly if available, else matplotlib
# ---------------------------------------------------------------------------

try:
    import plotly  # noqa: F401

    NotebookCallback = PlotlyNotebookCallback
except ImportError:
    NotebookCallback = MatplotlibNotebookCallback
