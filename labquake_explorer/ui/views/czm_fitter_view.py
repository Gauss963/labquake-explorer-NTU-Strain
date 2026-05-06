import tkinter as tk
from tkinter import ttk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy import optimize, signal

from labquake_explorer.data.data_processor import DataProcessor
from labquake_explorer.utils.cohesive_crack import CohesiveCrack


AUTO_FIT_DISTANCE_TO_FAULT_M = 0.005
AUTO_FIT_WINDOW_S = (-0.040, 0.040)
AUTO_FIT_WINDOW_MS = (
    AUTO_FIT_WINDOW_S[0] * 1000.0,
    AUTO_FIT_WINDOW_S[1] * 1000.0,
)
AUTO_FIT_ONSET_SEARCH_WINDOW_S = AUTO_FIT_WINDOW_S
AUTO_FIT_ONSET_THRESHOLD_FRACTION = 0.15
AUTO_FIT_DISPLAY_THRESHOLD_FRACTION = 0.05
AUTO_FIT_LEFT_BASELINE_WINDOW_S = 0.002
AUTO_FIT_MIN_SAMPLES = 25
AUTO_FIT_GAMMA_UPPER_BOUND_J_PER_M2 = 1e6
AUTO_FIT_PHYSICAL_BRANCH_RMSE_TOLERANCE = 1.15
AUTO_FIT_MAX_EVALS = 250
AUTO_FIT_YOUNGS_MODULUS_PA = 7.662e9
AUTO_FIT_POISSON_RATIO = 0.2
AUTO_FIT_DENSITY_KG_PER_M3 = 1148.0


def _get_cs(E, nu, rho):
    shear_modulus = E / (2.0 * (1.0 + nu))
    return float(np.sqrt(shear_modulus / rho))


def _get_cd(E, nu, rho):
    return float(np.sqrt(E * (1.0 - nu) / (rho * (1.0 + nu) * (1.0 - 2.0 * nu))))


def _compute_xc_modeii_si(E, nu, tau_c_pa, gc_j_m2):
    e_prime = E / (1.0 - nu**2)
    return float((9.0 * np.pi / 32.0) * (e_prime / tau_c_pa**2) * gc_j_m2)


def _moving_average(values, window):
    data = np.asarray(values, dtype=np.float64)
    if window <= 1:
        return data
    window = min(window, data.size)
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(data, kernel, mode="same")


def _rezero_delta_tau_from_window_start(relative_time_s, delta_tau_mpa):
    time_s = np.asarray(relative_time_s, dtype=np.float64)
    signal_mpa = np.asarray(delta_tau_mpa, dtype=np.float64).copy()
    if time_s.size == 0 or signal_mpa.size == 0:
        return signal_mpa

    baseline_end_s = min(float(time_s[0]) + AUTO_FIT_LEFT_BASELINE_WINDOW_S, float(time_s[-1]))
    baseline_mask = time_s <= baseline_end_s
    if not np.any(baseline_mask):
        baseline_count = max(1, min(signal_mpa.size, signal_mpa.size // 100 or 1))
        baseline = float(np.mean(signal_mpa[:baseline_count]))
    else:
        baseline = float(np.mean(signal_mpa[baseline_mask]))
    return signal_mpa - baseline


def _estimate_peak_time_s(relative_time_s, delta_tau_mpa):
    time_s = np.asarray(relative_time_s, dtype=np.float64)
    smooth_signal = _moving_average(delta_tau_mpa, 401)
    search_mask = (time_s >= AUTO_FIT_WINDOW_S[0]) & (time_s <= AUTO_FIT_WINDOW_S[1])
    search_time = time_s[search_mask]
    search_signal = smooth_signal[search_mask]
    if search_signal.size == 0:
        raise ValueError(f"No samples inside peak search window {AUTO_FIT_WINDOW_S}.")

    peak_idx = int(np.argmax(search_signal))
    peak_time_s = float(search_time[peak_idx])
    peak_delta_tau_mpa = float(search_signal[peak_idx])

    if 0 < peak_idx < len(search_signal) - 1:
        t0 = float(search_time[peak_idx - 1])
        t1 = float(search_time[peak_idx])
        t2 = float(search_time[peak_idx + 1])
        y0 = float(search_signal[peak_idx - 1])
        y1 = float(search_signal[peak_idx])
        y2 = float(search_signal[peak_idx + 1])
        denom = y0 - 2.0 * y1 + y2
        if denom != 0.0:
            frac = 0.5 * (y0 - y2) / denom
            peak_time_s = t1 + frac * (t1 - t0)

    return {
        "peak_time_s": peak_time_s,
        "peak_delta_tau_mpa": peak_delta_tau_mpa,
        "smooth_delta_tau_mpa": smooth_signal,
    }


def _estimate_onset_time_s(relative_time_s, delta_tau_mpa):
    time_s = np.asarray(relative_time_s, dtype=np.float64)
    smooth_signal = _moving_average(delta_tau_mpa, 401)
    search_mask = (time_s >= AUTO_FIT_ONSET_SEARCH_WINDOW_S[0]) & (time_s <= AUTO_FIT_ONSET_SEARCH_WINDOW_S[1])
    search_time = time_s[search_mask]
    search_signal = smooth_signal[search_mask]
    abs_signal = np.abs(search_signal)
    peak_abs = float(np.max(abs_signal)) if abs_signal.size else 0.0
    threshold_mpa = AUTO_FIT_ONSET_THRESHOLD_FRACTION * peak_abs

    crossing_indices = np.where(abs_signal >= threshold_mpa)[0] if peak_abs > 0 else np.array([], dtype=np.int64)
    if crossing_indices.size == 0:
        onset_time_s = float(search_time[np.argmax(abs_signal)])
    else:
        idx = int(crossing_indices[0])
        onset_time_s = float(search_time[idx])
        if idx > 0:
            t0 = float(search_time[idx - 1])
            t1 = float(search_time[idx])
            y0 = float(abs_signal[idx - 1])
            y1 = float(abs_signal[idx])
            if y1 != y0:
                frac = (threshold_mpa - y0) / (y1 - y0)
                onset_time_s = t0 + frac * (t1 - t0)

    return {
        "onset_time_s": onset_time_s,
        "threshold_mpa": threshold_mpa,
        "smooth_delta_tau_mpa": smooth_signal,
    }


def _interpolate_crossing_time_s(t0_s, y0, t1_s, y1):
    if y1 == y0:
        return float(t1_s)
    frac = (0.0 - y0) / (y1 - y0)
    return float(t0_s + frac * (t1_s - t0_s))


def _estimate_display_line_positions(fit_time_s, fit_model_mpa):
    time_s = np.asarray(fit_time_s, dtype=np.float64)
    model_mpa = np.asarray(fit_model_mpa, dtype=np.float64)
    if time_s.size == 0 or model_mpa.size == 0 or time_s.size != model_mpa.size:
        return None

    amplitude = float(np.max(np.abs(model_mpa)))
    if not np.isfinite(amplitude) or amplitude <= 0.0:
        return {
            "display_start_s": float(time_s[0]),
            "model_peak_time_s": float(time_s[np.argmax(model_mpa)]),
            "display_end_s": float(time_s[-1]),
        }

    threshold = AUTO_FIT_DISPLAY_THRESHOLD_FRACTION * amplitude
    mask = np.abs(model_mpa) >= threshold
    if not np.any(mask):
        return {
            "display_start_s": float(time_s[0]),
            "model_peak_time_s": float(time_s[np.argmax(model_mpa)]),
            "display_end_s": float(time_s[-1]),
        }

    indices = np.flatnonzero(mask)
    return {
        "display_start_s": float(time_s[indices[0]]),
        "model_peak_time_s": float(time_s[np.argmax(model_mpa)]),
        "display_end_s": float(time_s[indices[-1]]),
    }


def _build_fit_problem(relative_time_s, delta_tau_mpa, peak_info):
    time_s = np.asarray(relative_time_s, dtype=np.float64)
    signal_mpa = np.asarray(delta_tau_mpa, dtype=np.float64)
    smooth_signal = np.asarray(peak_info["smooth_delta_tau_mpa"], dtype=np.float64)
    peak_time_s = float(peak_info["peak_time_s"])

    analysis_mask = (time_s >= AUTO_FIT_WINDOW_S[0]) & (time_s <= AUTO_FIT_WINDOW_S[1])
    analysis_indices = np.flatnonzero(analysis_mask)
    if analysis_indices.size == 0:
        raise ValueError(f"No samples found inside zoom window {AUTO_FIT_WINDOW_S}.")

    peak_idx = int(np.argmin(np.abs(time_s - peak_time_s)))
    start_idx = int(analysis_indices[0])
    end_idx = int(analysis_indices[-1])

    for idx in range(peak_idx - 1, start_idx - 1, -1):
        y0 = float(smooth_signal[idx])
        y1 = float(smooth_signal[idx + 1])
        if y0 <= 0.0 < y1:
            start_idx = idx
            break

    for idx in range(peak_idx, end_idx):
        y0 = float(smooth_signal[idx])
        y1 = float(smooth_signal[idx + 1])
        if y0 >= 0.0 > y1:
            end_idx = idx + 1
            break

    fit_start_s = float(time_s[start_idx])
    fit_end_s = float(time_s[end_idx])
    if start_idx < peak_idx:
        fit_start_s = _interpolate_crossing_time_s(
            float(time_s[start_idx]),
            float(smooth_signal[start_idx]),
            float(time_s[start_idx + 1]),
            float(smooth_signal[start_idx + 1]),
        )
    if peak_idx < end_idx:
        fit_end_s = _interpolate_crossing_time_s(
            float(time_s[end_idx - 1]),
            float(smooth_signal[end_idx - 1]),
            float(time_s[end_idx]),
            float(smooth_signal[end_idx]),
        )

    fit_mask = (time_s >= fit_start_s) & (time_s <= fit_end_s)
    if int(np.count_nonzero(fit_mask)) < AUTO_FIT_MIN_SAMPLES:
        pad = max(1, (AUTO_FIT_MIN_SAMPLES - int(np.count_nonzero(fit_mask))) // 2 + 1)
        start_idx = max(0, int(np.argmin(np.abs(time_s - fit_start_s))) - pad)
        end_idx = min(len(time_s) - 1, end_idx + pad)
        fit_start_s = float(time_s[start_idx])
        fit_end_s = float(time_s[end_idx])
        fit_mask = (time_s >= fit_start_s) & (time_s <= fit_end_s)

    fit_time_s = time_s[fit_mask]
    fit_data_mpa = signal_mpa[fit_mask]
    baseline_mask = np.zeros_like(fit_time_s, dtype=bool)
    edge_count = min(5, fit_time_s.size)
    baseline_mask[:edge_count] = True
    baseline_mask[-edge_count:] = True

    return {
        "fit_mask": fit_mask,
        "fit_time_s": fit_time_s,
        "fit_data_mpa": fit_data_mpa,
        "fit_baseline_mask": baseline_mask,
        "fit_start_s": fit_start_s,
        "fit_end_s": fit_end_s,
    }


def _estimate_velocity_from_arrivals(arrival_times_s, positions_m, labels):
    positions = np.asarray([positions_m[label] for label in labels], dtype=np.float64)
    times = np.asarray([arrival_times_s[label] for label in labels], dtype=np.float64)
    slope_s_per_m, intercept_s = np.polyfit(positions, times, 1)
    cf_mps = float(np.inf if slope_s_per_m == 0 else 1.0 / abs(slope_s_per_m))
    direction = f"{labels[0]}->{labels[-1]}" if slope_s_per_m > 0 else f"{labels[-1]}->{labels[0]}"
    return {
        "cf_mps": cf_mps,
        "direction": direction,
        "slope_s_per_m": float(slope_s_per_m),
        "intercept_s": float(intercept_s),
    }


def _model_delta_tau(theta, *, fit_problem, x_sign, cs_mps, cd_mps):
    cf_ratio, gamma_j_m2, tau_c_mpa, t0_s = theta
    if cf_ratio <= 0 or gamma_j_m2 <= 0 or tau_c_mpa <= 0:
        return np.full_like(fit_problem["fit_data_mpa"], 1e6)

    cf_mps = cf_ratio * cs_mps
    tau_c_pa = tau_c_mpa * 1e6
    try:
        xc_m = _compute_xc_modeii_si(AUTO_FIT_YOUNGS_MODULUS_PA, AUTO_FIT_POISSON_RATIO, tau_c_pa, gamma_j_m2)
        x_m = x_sign * cf_mps * (np.asarray(fit_problem["fit_time_s"], dtype=np.float64) - t0_s)
        _, model_tau_pa, _ = CohesiveCrack.delta_sigmas(
            x_m,
            AUTO_FIT_DISTANCE_TO_FAULT_M,
            xc_m,
            cf_mps,
            cs_mps,
            cd_mps,
            AUTO_FIT_POISSON_RATIO,
            gamma_j_m2,
            AUTO_FIT_YOUNGS_MODULUS_PA,
        )
        model_tau_mpa = np.asarray(model_tau_pa, dtype=np.float64) / 1e6
    except Exception:
        return np.full_like(fit_problem["fit_data_mpa"], 1e6)

    if not np.all(np.isfinite(model_tau_mpa)):
        return np.full_like(fit_problem["fit_data_mpa"], 1e6)

    fit_time_s = np.asarray(fit_problem["fit_time_s"], dtype=np.float64)
    baseline_mask = np.asarray(fit_problem["fit_baseline_mask"], dtype=bool)
    baseline_idx = np.flatnonzero(baseline_mask)
    if baseline_idx.size < 2:
        baseline_mean = float(np.mean(model_tau_mpa[baseline_mask])) if np.any(baseline_mask) else float(np.mean(model_tau_mpa))
        return model_tau_mpa - baseline_mean

    split = baseline_idx.size // 2
    left_idx = baseline_idx[:split]
    right_idx = baseline_idx[split:]
    if left_idx.size == 0 or right_idx.size == 0:
        baseline_mean = float(np.mean(model_tau_mpa[baseline_mask]))
        return model_tau_mpa - baseline_mean

    t_left = float(np.mean(fit_time_s[left_idx]))
    t_right = float(np.mean(fit_time_s[right_idx]))
    y_left = float(np.mean(model_tau_mpa[left_idx]))
    y_right = float(np.mean(model_tau_mpa[right_idx]))
    if np.isclose(t_right, t_left):
        baseline = np.full_like(model_tau_mpa, 0.5 * (y_left + y_right))
    else:
        baseline = y_left + (y_right - y_left) * (fit_time_s - t_left) / (t_right - t_left)
    return model_tau_mpa - baseline


def _residuals_fixed_cf(theta, *, fit_problem, cf_ratio, x_sign, cs_mps, cd_mps):
    local_theta = np.concatenate(([cf_ratio], np.asarray(theta, dtype=np.float64)))
    return _model_delta_tau(
        local_theta,
        fit_problem=fit_problem,
        x_sign=x_sign,
        cs_mps=cs_mps,
        cd_mps=cd_mps,
    ) - fit_problem["fit_data_mpa"]


class CZMFitterView(tk.Toplevel):
    def __init__(self, parent, run_idx, event_idx):
        self.parent = parent
        super().__init__(self.parent.root)
        self.title("Cohesive Zone Model Fitting")

        self.run_idx = run_idx
        self.event_idx = event_idx
        self.event = None
        self.filtering = False
        self.data_manager = self.parent.data_manager
        self.strain_gauge = tk.IntVar(value=6)
        self.gauge_selection = tk.StringVar()
        self.num_gauges = None
        self.available_gauge_indices = []
        self.available_gauge_options = []
        self.gauge_option_to_index = {}
        self.gauge_index_to_option = {}
        self.auto_fit_result = None

        self.E = AUTO_FIT_YOUNGS_MODULUS_PA
        self.nu = AUTO_FIT_POISSON_RATIO
        self.C_s = _get_cs(self.E, self.nu, AUTO_FIT_DENSITY_KG_PER_M3)
        self.C_d = _get_cd(self.E, self.nu, AUTO_FIT_DENSITY_KG_PER_M3)

        self.create_matplotlib_figure()

        self.vlines = []
        self.vlines_twin = []
        self.active_line_idx = None
        self.drag_active = False
        self.user_adjusted_lines = False

        self.Cf = tk.DoubleVar()
        self.y = tk.DoubleVar()
        self.Xc = tk.DoubleVar()
        self.Gc = tk.DoubleVar()

        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.create_control_frame()
        self.create_parameters_frame()
        self.load_event(self.event_idx)

        self.canvas.mpl_connect("button_press_event", self.on_mouse_press)
        self.canvas.mpl_connect("button_release_event", self.on_mouse_release)
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)

        self.init_event_combobox()
        self.event_combobox.bind("<<ComboboxSelected>>", self.on_event_changed)
        self.filter_spinbox.bind("<Return>", self.update_plot)

        self.update_plot(preserve_xlim=False)

    def create_control_frame(self):
        control_frame = ttk.Frame(self)
        control_frame.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        ttk.Label(control_frame, text="Event Index:").pack(side=tk.LEFT, padx=5)
        self.event_combobox = ttk.Combobox(control_frame, width=10)
        self.event_combobox.pack(side=tk.LEFT, padx=5)

        ttk.Label(control_frame, text="Strain Gauge:").pack(side=tk.LEFT, padx=5)
        self.gauge_combobox = ttk.Combobox(
            control_frame,
            textvariable=self.gauge_selection,
            width=12,
            state="readonly",
        )
        self.gauge_combobox.pack(side=tk.LEFT, padx=5)
        self.gauge_combobox.bind("<<ComboboxSelected>>", self.on_gauge_changed)

        filter_frame = ttk.Frame(control_frame)
        filter_frame.pack(side=tk.LEFT, padx=10)

        ttk.Label(filter_frame, text="Filter Window:").pack(side=tk.LEFT, padx=2)
        self.filter_window = tk.IntVar(value=51)
        self.filter_spinbox = ttk.Spinbox(
            filter_frame,
            from_=3,
            to=201,
            increment=2,
            textvariable=self.filter_window,
            width=10,
            validate="focusout",
            validatecommand=(self.register(self.validate_filter_window), "%P"),
        )
        self.filter_spinbox.pack(side=tk.LEFT)

        self.filter_button = tk.Button(
            control_frame,
            text="Filter Off",
            relief="raised",
            command=self.toggle_filter,
        )
        self.filter_button.pack(side=tk.LEFT, padx=5)

    def create_matplotlib_figure(self):
        self.fig = plt.figure(figsize=(10, 6))
        self.gs = self.fig.add_gridspec(2, hspace=0.3)
        self.axs = self.gs.subplots(sharex=True)
        for ax in self.axs:
            ax.grid(True)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

        toolbar_frame = ttk.Frame(self)
        toolbar_frame.grid(row=2, column=0, padx=0, pady=0, sticky="ew")
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

    def create_parameters_frame(self):
        params_frame = ttk.Frame(self)
        params_frame.grid(row=3, column=0, padx=5, pady=5, sticky="ew")

        param_configs = [
            ("Cf", self.Cf, 10),
            ("y", self.y, 1e-3),
            ("Xc", self.Xc, 1e-4),
            ("Gc", self.Gc, 1),
        ]
        for label_text, var, increment in param_configs:
            frame = ttk.Frame(params_frame)
            frame.pack(side=tk.LEFT, padx=10)
            ttk.Label(frame, text=label_text).pack(side=tk.LEFT, padx=2)
            spinbox = ttk.Spinbox(
                frame,
                textvariable=var,
                width=10,
                from_=0,
                to=1e6,
                increment=increment,
            )
            spinbox.pack(side=tk.LEFT)

        button_frame = ttk.Frame(params_frame)
        button_frame.pack(side=tk.LEFT, padx=10)

        ttk.Button(button_frame, text="Update", command=self.update_plot).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Fit", command=self.fit_parameters).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save", command=self.save_parameters).pack(side=tk.LEFT, padx=5)

    def load_event(self, event_idx):
        for line in self.vlines:
            line.remove()
        for line in self.vlines_twin:
            line.remove()
        self.vlines = []
        self.vlines_twin = []

        self.event_idx = event_idx
        self.event = self.data_manager.get_data(f"runs/[{self.run_idx}]/events/[{self.event_idx}]")
        self.user_adjusted_lines = False

        self.num_gauges = len(self.event["strain"]["original"]["raw"])
        self._configure_available_gauges()
        self.auto_fit_result = self._compute_script_style_auto_fit()
        has_pick_arrivals = isinstance(self.event.get("pick_arrivals"), dict)

        if "czm_parms" in self.event:
            params = self.event["czm_parms"]
            if isinstance(params, list) and len(params) == 8:
                self._set_parameters(*params[:4])
                vline_x0, vline_x1 = params[4], params[5]
                vline_x2 = vline_x1 * 2 - vline_x0
                self.x_lim_min, self.x_lim_max = params[6], params[7]
                self._set_selected_gauge(self.available_gauge_indices[0])
            elif isinstance(params, dict):
                self._set_parameters(params["Cf"], params["y"], params["Xc"], params["Gc"])
                vline_x0, vline_x1, vline_x2 = params["x_min"], params["x_tip"], params["x_max"]
                self.x_lim_min, self.x_lim_max = params["x_lim_min"], params["x_lim_max"]
                if "strain_gauge" in params and params["strain_gauge"] in self.available_gauge_indices:
                    self._set_selected_gauge(params["strain_gauge"])
                else:
                    self._set_selected_gauge(self.available_gauge_indices[0])
            self._plot_vertical_lines([vline_x0, vline_x1, vline_x2])
            self.event["czm_parms"] = {
                "Cf": self.Cf.get(),
                "y": self.y.get(),
                "Xc": self.Xc.get(),
                "Gc": self.Gc.get(),
                "strain_gauge": self.strain_gauge.get(),
                "x_min": vline_x0,
                "x_tip": vline_x1,
                "x_max": vline_x2,
                "x_lim_min": self.x_lim_min,
                "x_lim_max": self.x_lim_max,
            }
        else:
            self._set_selected_gauge(self.available_gauge_indices[0])
            self._apply_auto_fit_defaults()

        self.axs[0].set_xlim(self.x_lim_min, self.x_lim_max)

    def _set_parameters(self, Cf, y, Xc, Gc):
        self.Cf.set(Cf)
        self.y.set(y)
        self.Xc.set(Xc)
        self.Gc.set(Gc)

    def _set_default_parameters(self):
        self.x_lim_min, self.x_lim_max = -0.1, 0.1
        try:
            rupture_speed = self.data_manager.get_data(
                f"runs/[{self.run_idx}]/events/[{self.event_idx}]/rupture_speed"
            )
            self.Cf.set(np.abs(rupture_speed))
        except Exception:
            self.Cf.set(10)
        self.y.set(AUTO_FIT_DISTANCE_TO_FAULT_M)
        self.Xc.set(1)
        self.Gc.set(1)

    def _plot_vertical_lines(self, positions):
        if not hasattr(self, "axs"):
            return
        display_scale = 1000.0 if isinstance(self.event.get("pick_arrivals"), dict) else 1.0
        for i, x_pos in enumerate(positions):
            color = "r" if i == 1 else "g"
            x_display = x_pos * display_scale
            vline = self.axs[0].axvline(x=x_display, color=color, linestyle="--", alpha=0.5)
            vline_twin = self.axs[1].axvline(x=x_display, color=color, linestyle="--", alpha=0.5)
            self.vlines.append(vline)
            self.vlines_twin.append(vline_twin)

    def init_event_combobox(self):
        n_events = len(self.data_manager.get_data(f"runs/[{self.run_idx}]/events"))
        options = [f"{i}" for i in range(n_events)]
        self.event_combobox.config(values=options, state="readonly")
        self.event_combobox.current(self.event_idx)

    def on_event_changed(self, event=None):
        self.load_event(int(self.event_combobox.get()))
        self.update_plot(preserve_xlim=False)

    def on_gauge_changed(self, event=None):
        selected = self.gauge_combobox.get()
        raw_index = self.gauge_option_to_index.get(selected)
        if raw_index is None:
            return
        self.strain_gauge.set(raw_index)
        self.gauge_selection.set(selected)
        if "czm_parms" not in self.event:
            self._apply_auto_fit_to_selected_gauge()
        self.update_plot()

    def toggle_filter(self):
        self.filtering = not self.filtering
        if self.filtering:
            self.filter_button.config(text="Filter On", relief="sunken")
        else:
            self.filter_button.config(text="Filter Off", relief="raised")
        self.update_plot()

    def save_parameters(self):
        if len(self.vlines) >= 3:
            scale = 1e-3 if isinstance(self.event.get("pick_arrivals"), dict) else 1.0
            vline_x0 = self.vlines[0].get_xdata()[0] * scale
            vline_x1 = self.vlines[1].get_xdata()[0] * scale
            vline_x2 = self.vlines[2].get_xdata()[0] * scale
            xlim_min, xlim_max = self.axs[0].get_xlim()
            self.x_lim_min, self.x_lim_max = xlim_min * scale, xlim_max * scale
            params = {
                "Cf": self.Cf.get(),
                "y": self.y.get(),
                "Xc": self.Xc.get(),
                "Gc": self.Gc.get(),
                "strain_gauge": self.strain_gauge.get(),
                "x_min": vline_x0,
                "x_tip": vline_x1,
                "x_max": vline_x2,
                "x_lim_min": self.x_lim_min,
                "x_lim_max": self.x_lim_max,
            }
            self.event["czm_parms"] = params
            self.data_manager.set_data(
                f"runs/[{self.run_idx}]/events/[{self.event_idx}]/czm_parms",
                params,
                True,
            )
            self.parent.refresh_tree()
            print(f"Saved parameters for event {self.event_idx}: {params}")

    def update_plot(self, event=None, preserve_xlim=True):
        has_pick_arrivals = isinstance(self.event.get("pick_arrivals"), dict)
        current_display_xlim = None
        if preserve_xlim and hasattr(self, "axs") and len(self.axs) > 0:
            try:
                current_display_xlim = tuple(float(value) for value in self.axs[0].get_xlim())
            except Exception:
                current_display_xlim = None
        line_unit_scale = 1e-3 if has_pick_arrivals else 1.0
        line_positions = []
        if self.vlines:
            line_positions = [line.get_xdata()[0] * line_unit_scale for line in self.vlines]
        elif "czm_parms" in self.event:
            params = self.event["czm_parms"]
            if isinstance(params, list) and len(params) == 8:
                line_positions = [params[4], params[5], params[5] * 2 - params[4]]
            elif isinstance(params, dict):
                line_positions = [params["x_min"], params["x_tip"], params["x_max"]]
        if not line_positions:
            line_positions = np.linspace(self.x_lim_min, self.x_lim_max, 5)[1:-1]

        for ax in self.axs:
            ax.clear()

        gauge_idx = self.strain_gauge.get()
        self._set_selected_gauge(gauge_idx)
        selected_label = self.gauge_index_to_option.get(gauge_idx)
        pick_arrivals = self.event.get("pick_arrivals", {})
        colors_by_label = pick_arrivals.get("colors_by_label", {})
        gauge_color = colors_by_label.get(selected_label, "C0")
        display_xlim = (self.x_lim_min, self.x_lim_max)
        plot_line_positions = list(line_positions)

        if selected_label and isinstance(pick_arrivals, dict) and selected_label in (pick_arrivals.get("delta_tau_mpa_by_label") or {}):
            t = np.asarray(pick_arrivals["relative_time_s"], dtype=np.float64)
            delta_tau = _rezero_delta_tau_from_window_start(
                t,
                np.asarray(pick_arrivals["delta_tau_mpa_by_label"][selected_label], dtype=np.float64),
            )
            t_ms = t * 1000.0
            display_xlim = current_display_xlim if current_display_xlim is not None else AUTO_FIT_WINDOW_MS
            plot_line_positions = [position * 1000.0 for position in line_positions]
            if self.filtering:
                window_length = self.filter_window.get()
                if window_length % 2 == 0:
                    window_length += 1
                    self.filter_window.set(window_length)
                delta_tau = signal.savgol_filter(delta_tau, window_length, 2)

            fit_mask = (t >= line_positions[0]) & (t <= line_positions[2])
            fit_time = t[fit_mask]
            fit_data = delta_tau[fit_mask]
            script_fit_result = None
            if (
                not self.user_adjusted_lines
                and self.auto_fit_result is not None
                and selected_label in self.auto_fit_result["fit_map"]
            ):
                script_fit_result = self.auto_fit_result["fit_map"][selected_label]

            if script_fit_result is not None:
                fit_time = np.asarray(script_fit_result["fit_time_s"], dtype=np.float64)
                model_tau = np.asarray(script_fit_result["fit_model_mpa"], dtype=np.float64)
                fit_data = np.asarray(script_fit_result["fit_data_mpa"], dtype=np.float64)
                x_fit = -fit_time * self.Cf.get()
                x_fit_zeroed = x_fit + float(script_fit_result["peak_time_s"]) * self.Cf.get()
                delta_sigma_xx, _, _ = CohesiveCrack.delta_sigmas(
                    x_fit_zeroed,
                    self.y.get(),
                    self.Xc.get(),
                    self.Cf.get(),
                    self.C_s,
                    self.C_d,
                    self.nu,
                    self.Gc.get(),
                    self.E,
                )
                delta_e_xx = np.asarray(delta_sigma_xx, dtype=np.float64) / self.E
                delta_e_xx -= delta_e_xx[0]
            elif fit_time.size > 0:
                baseline_mask = np.zeros_like(fit_time, dtype=bool)
                edge_count = min(5, fit_time.size)
                baseline_mask[:edge_count] = True
                baseline_mask[-edge_count:] = True

                e_prime = self.E / (1.0 - self.nu**2)
                tau_c_pa = np.sqrt(max((9.0 * np.pi / 32.0) * e_prime * self.Gc.get() / max(self.Xc.get(), 1e-12), 1e-12))
                tau_c_mpa = tau_c_pa / 1e6
                fit_problem = {
                    "fit_time_s": fit_time,
                    "fit_data_mpa": fit_data,
                    "fit_baseline_mask": baseline_mask,
                }
                x_sign = -1
                if self.auto_fit_result is not None:
                    x_sign = int(self.auto_fit_result["fit_summary"]["x_sign"])
                theta = np.array([self.Cf.get() / self.C_s, self.Gc.get(), tau_c_mpa, line_positions[1]], dtype=np.float64)
                model_tau = _model_delta_tau(
                    theta,
                    fit_problem=fit_problem,
                    x_sign=x_sign,
                    cs_mps=self.C_s,
                    cd_mps=self.C_d,
                )

                x_fit = -fit_time * self.Cf.get()
                x_fit_zeroed = x_fit + line_positions[1] * self.Cf.get()
                delta_sigma_xx, _, _ = CohesiveCrack.delta_sigmas(
                    x_fit_zeroed,
                    self.y.get(),
                    self.Xc.get(),
                    self.Cf.get(),
                    self.C_s,
                    self.C_d,
                    self.nu,
                    self.Gc.get(),
                    self.E,
                )
                delta_e_xx = np.asarray(delta_sigma_xx, dtype=np.float64) / self.E
                delta_e_xx -= delta_e_xx[0]
            else:
                model_tau = np.array([])
                fit_time = np.array([])
                delta_e_xx = np.array([])

            self.axs[0].plot(t_ms, delta_tau, color=gauge_color, linewidth=1.2, label=r"$\Delta\tau$ data")
            if fit_time.size > 0:
                fit_time_ms = fit_time * 1000.0
                self.axs[0].plot(fit_time_ms, model_tau, color="black", linewidth=1.5, label="CZM fit")
                self.axs[1].plot(fit_time_ms, delta_e_xx, color="black", linestyle="--", linewidth=1.5, label="CZM Exx (display only)")
        else:
            t = self.event["strain"]["original"]["time"] - self.event["event_time"]
            exy = DataProcessor.voltage_to_strain(self.event["strain"]["original"]["raw"][gauge_idx])
            if self.filtering:
                window_length = self.filter_window.get()
                if window_length % 2 == 0:
                    window_length += 1
                    self.filter_window.set(window_length)
                exy = signal.savgol_filter(exy, window_length, 2)
            idx_zero_xy = np.argmin(np.abs(t - line_positions[2]))
            self.axs[0].plot(t * 1000.0, exy - exy[idx_zero_xy], color=gauge_color, linewidth=1.2, label="Exy")
            plot_line_positions = [position * 1000.0 for position in line_positions]
            display_xlim = current_display_xlim if current_display_xlim is not None else (self.x_lim_min * 1000.0, self.x_lim_max * 1000.0)

        fit_span_color = "tab:red"
        for ax in self.axs:
            ax.axvspan(plot_line_positions[0], plot_line_positions[2], color=fit_span_color, alpha=0.08, zorder=-120)
            ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.8)

        self.axs[1].set_xlabel("Time relative to trigger (ms)")
        self.axs[0].set_ylabel(r"$\Delta \tau$ (MPa)")
        self.axs[1].set_ylabel("Exx")
        block = self.event.get("block")
        title = f"{self.data_manager.get_data('name')} run{self.run_idx:02d} event{self.event_idx}"
        if block is not None:
            title += f" block{int(block)}"
        self.fig.suptitle(title)

        self.vlines = []
        self.vlines_twin = []
        line_colors = ["tab:purple", "tab:red", "tab:green"]
        line_styles = [":", "--", "--"]
        for i, x_pos in enumerate(plot_line_positions):
            color = line_colors[i] if i < len(line_colors) else "g"
            linestyle = line_styles[i] if i < len(line_styles) else "--"
            vline = self.axs[0].axvline(x=x_pos, color=color, linestyle=linestyle, alpha=0.75)
            self.vlines.append(vline)
            vline_twin = self.axs[1].axvline(x=x_pos, color=color, linestyle=linestyle, alpha=0.75)
            self.vlines_twin.append(vline_twin)

        self.axs[0].legend()
        self.axs[1].legend()
        for ax in self.axs:
            ax.set_xlim(display_xlim[0], display_xlim[1])
            ax.margins(x=0.0)
        self.canvas.draw()

    def is_navigation_active(self):
        return self.toolbar.mode in ["pan/zoom", "zoom rect"]

    def on_mouse_press(self, event):
        if self.is_navigation_active():
            return
        if event.button == 1 and event.inaxes:
            for i, vline in enumerate(self.vlines):
                line_x = vline.get_xdata()[0]
                xlim_temp = self.axs[0].get_xlim()
                if abs(event.xdata - line_x) < 0.01 * (xlim_temp[1] - xlim_temp[0]):
                    self.drag_active = True
                    self.active_line_idx = i
                    break

    def on_mouse_release(self, event):
        dragged = self.drag_active and self.active_line_idx is not None
        self.drag_active = False
        self.active_line_idx = None
        if dragged:
            self.user_adjusted_lines = True
            self.update_plot()

    def on_mouse_move(self, event):
        if self.is_navigation_active():
            return
        if self.drag_active and event.inaxes and self.active_line_idx is not None:
            new_x = event.xdata
            self.vlines[self.active_line_idx].set_xdata([new_x, new_x])
            self.vlines_twin[self.active_line_idx].set_xdata([new_x, new_x])
            self.canvas.draw_idle()

    def validate_filter_window(self, value):
        if value == "":
            return True
        try:
            val = int(value)
            return 3 <= val <= 201 and val % 2 == 1
        except ValueError:
            return False

    def fit_parameters(self):
        auto_fit = self._compute_script_style_auto_fit(cf_override=self.Cf.get())
        if auto_fit is None:
            print("Script-style auto fit is unavailable for this event.")
            return

        self.auto_fit_result = auto_fit
        self.user_adjusted_lines = False
        self.Cf.set(float(auto_fit["fit_summary"]["cf_mps"]))
        self.y.set(AUTO_FIT_DISTANCE_TO_FAULT_M)
        self._apply_auto_fit_to_selected_gauge()
        self.update_plot()

        selected_label = self.gauge_index_to_option.get(self.strain_gauge.get(), "selected")
        fit_result = auto_fit["fit_map"].get(selected_label, next(iter(auto_fit["fit_map"].values())))
        print(
            f"Auto-fit ({selected_label}): Cf={auto_fit['fit_summary']['cf_mps']:.1f} m/s, "
            f"Gc={fit_result['gamma_j_m2']:.3e} J/m^2, Xc={fit_result['xc_mm']:.2f} mm, "
            f"t0={fit_result['t0_s'] * 1000.0:.3f} ms, branch={auto_fit['fit_summary']['x_sign']}"
        )

    def _configure_available_gauges(self):
        strain = self.event.get("strain", {})
        enabled = strain.get("enabled_channels")
        fitting = strain.get("fitting_channels")
        enabled = list(enabled) if enabled is not None else []
        fitting = list(fitting) if fitting is not None else []
        labels = strain.get("labels")

        preferred = [idx for idx, flag in enumerate(fitting) if flag]
        if not preferred:
            preferred = [idx for idx, flag in enumerate(enabled) if flag]
        if not preferred:
            preferred = list(range(self.num_gauges))

        if labels is not None and len(labels) == len(preferred):
            option_pairs = [(str(label), raw_idx) for label, raw_idx in zip(labels, preferred)]
        else:
            option_pairs = [(f"ch{raw_idx}", raw_idx) for raw_idx in preferred]

        self.available_gauge_indices = [raw_idx for _, raw_idx in option_pairs]
        self.available_gauge_options = [label for label, _ in option_pairs]
        self.gauge_option_to_index = dict(option_pairs)
        self.gauge_index_to_option = {raw_idx: label for label, raw_idx in option_pairs}
        self.gauge_combobox.config(values=self.available_gauge_options)

    def _set_selected_gauge(self, raw_index):
        if raw_index not in self.gauge_index_to_option:
            raw_index = self.available_gauge_indices[0]
        self.strain_gauge.set(raw_index)
        self.gauge_selection.set(self.gauge_index_to_option[raw_index])
        self.gauge_combobox.set(self.gauge_selection.get())

    def _apply_auto_fit_defaults(self):
        self._set_default_parameters()
        self.auto_fit_result = self._compute_script_style_auto_fit()
        if self.auto_fit_result is None:
            return
        self.Cf.set(float(self.auto_fit_result["fit_summary"]["cf_mps"]))
        self.y.set(AUTO_FIT_DISTANCE_TO_FAULT_M)
        self.x_lim_min, self.x_lim_max = AUTO_FIT_WINDOW_S
        self._apply_auto_fit_to_selected_gauge()

    def _apply_auto_fit_to_selected_gauge(self):
        if not self.auto_fit_result:
            return
        self.user_adjusted_lines = False
        selected_label = self.gauge_index_to_option.get(self.strain_gauge.get())
        if selected_label not in self.auto_fit_result["fit_map"]:
            selected_label = next(iter(self.auto_fit_result["fit_map"]))
        fit_result = self.auto_fit_result["fit_map"][selected_label]
        self.Xc.set(float(fit_result["xc_m"]))
        self.Gc.set(float(fit_result["gamma_j_m2"]))
        for line in self.vlines:
            line.remove()
        for line in self.vlines_twin:
            line.remove()
        self.vlines = []
        self.vlines_twin = []
        self._plot_vertical_lines(
            [
                float(fit_result.get("display_start_s", fit_result["fit_start_s"])),
                float(fit_result.get("model_peak_time_s", fit_result["t0_s"])),
                float(fit_result.get("display_end_s", fit_result["fit_end_s"])),
            ]
        )

    def _compute_script_style_auto_fit(self, cf_override=None):
        pick_arrivals = self.event.get("pick_arrivals")
        if not isinstance(pick_arrivals, dict):
            return None

        relative_time_s = np.asarray(pick_arrivals.get("relative_time_s"), dtype=np.float64)
        delta_tau_by_label = pick_arrivals.get("delta_tau_mpa_by_label") or {}
        if relative_time_s.size == 0 or not delta_tau_by_label:
            return None

        labels = [label for label in self.available_gauge_options if label in delta_tau_by_label]
        if len(labels) < 2:
            return None

        raw_positions = list(self.event["strain"].get("locations") or [])
        positions_mm = {}
        for raw_idx, label in zip(self.available_gauge_indices, self.available_gauge_options):
            if raw_idx < len(raw_positions):
                positions_mm[label] = float(raw_positions[raw_idx])
        if len(positions_mm) < len(labels):
            positions_mm = {label: float(idx * 100.0) for idx, label in enumerate(labels)}
        min_position_mm = min(positions_mm[label] for label in labels)
        positions_m = {label: (positions_mm[label] - min_position_mm) / 1000.0 for label in labels}

        peak_map = {}
        onset_map = {}
        arrival_times_s = {}
        for label in labels:
            delta_tau_signal = _rezero_delta_tau_from_window_start(relative_time_s, delta_tau_by_label[label])
            peak_info = _estimate_peak_time_s(relative_time_s, delta_tau_signal)
            onset_info = _estimate_onset_time_s(relative_time_s, delta_tau_signal)
            peak_map[label] = peak_info
            onset_map[label] = onset_info
            arrival_times_s[label] = float(peak_info["peak_time_s"])

        cf_result = _estimate_velocity_from_arrivals(arrival_times_s, positions_m, labels)
        cf_mps = float(cf_override) if cf_override is not None else float(cf_result["cf_mps"])
        if not np.isfinite(cf_mps) or cf_mps <= 0:
            return None

        branch_results = {}
        for x_sign in (-1, 1):
            fit_map = {}
            channel_rmses = []
            gamma_upper_hits = 0
            messages = []
            total_nfev = 0
            success = True

            for label in labels:
                delta_tau_signal = _rezero_delta_tau_from_window_start(relative_time_s, delta_tau_by_label[label])
                fit_problem = _build_fit_problem(relative_time_s, delta_tau_signal, peak_map[label])
                x0 = np.array([2_000.0, 5.0, float(peak_map[label]["peak_time_s"])], dtype=np.float64)
                lower = np.array([1e-3, 1e-3, float(fit_problem["fit_start_s"])], dtype=np.float64)
                upper = np.array(
                    [
                        AUTO_FIT_GAMMA_UPPER_BOUND_J_PER_M2,
                        1_000.0,
                        float(fit_problem["fit_end_s"]),
                    ],
                    dtype=np.float64,
                )
                result = optimize.least_squares(
                    _residuals_fixed_cf,
                    x0,
                    bounds=(lower, upper),
                    kwargs={
                        "fit_problem": fit_problem,
                        "cf_ratio": float(cf_mps / self.C_s),
                        "x_sign": x_sign,
                        "cs_mps": self.C_s,
                        "cd_mps": self.C_d,
                    },
                    loss="soft_l1",
                    f_scale=0.2,
                    max_nfev=AUTO_FIT_MAX_EVALS,
                )

                gamma_j_m2, tau_c_mpa, t0_s = np.asarray(result.x, dtype=np.float64)
                local_theta = np.array([float(cf_mps / self.C_s), gamma_j_m2, tau_c_mpa, t0_s], dtype=np.float64)
                fit_model_mpa = _model_delta_tau(
                    local_theta,
                    fit_problem=fit_problem,
                    x_sign=x_sign,
                    cs_mps=self.C_s,
                    cd_mps=self.C_d,
                )
                fit_data_mpa = np.asarray(fit_problem["fit_data_mpa"], dtype=np.float64)
                xc_m = _compute_xc_modeii_si(self.E, self.nu, float(tau_c_mpa) * 1e6, float(gamma_j_m2))
                rmse_mpa = float(np.sqrt(np.mean((fit_model_mpa - fit_data_mpa) ** 2)))

                if float(gamma_j_m2) >= 0.99 * AUTO_FIT_GAMMA_UPPER_BOUND_J_PER_M2:
                    gamma_upper_hits += 1
                channel_rmses.append(rmse_mpa)
                total_nfev += int(result.nfev)
                success = success and bool(result.success)
                messages.append(f"{label}:{result.message}")
                fit_map[label] = {
                    "gamma_j_m2": float(gamma_j_m2),
                    "tau_c_mpa": float(tau_c_mpa),
                    "xc_m": float(xc_m),
                    "xc_mm": float(xc_m * 1000.0),
                    "t0_s": float(t0_s),
                    "rmse_mpa": rmse_mpa,
                    "fit_model_mpa": fit_model_mpa,
                    "fit_data_mpa": fit_data_mpa,
                    "fit_time_s": np.asarray(fit_problem["fit_time_s"]),
                    "fit_mask": np.asarray(fit_problem["fit_mask"]),
                    "fit_start_s": float(fit_problem["fit_start_s"]),
                    "fit_end_s": float(fit_problem["fit_end_s"]),
                    "onset_time_s": float(onset_map[label]["onset_time_s"]),
                    "peak_time_s": float(peak_map[label]["peak_time_s"]),
                    "peak_delta_tau_mpa": float(peak_map[label]["peak_delta_tau_mpa"]),
                    "success": bool(result.success),
                }
                display_positions = _estimate_display_line_positions(
                    fit_map[label]["fit_time_s"],
                    fit_map[label]["fit_model_mpa"],
                )
                if display_positions is not None:
                    fit_map[label].update(display_positions)

            fit_summary = {
                "cf_mps": float(cf_mps),
                "cf_ratio": float(cf_mps / self.C_s),
                "x_sign": x_sign,
                "rmse_mpa": float(np.mean(channel_rmses)),
                "success": success,
                "nfev": total_nfev,
                "message": " | ".join(messages),
            }
            branch_results[x_sign] = {
                "fit_map": fit_map,
                "fit_summary": fit_summary,
                "gamma_upper_hits": gamma_upper_hits,
            }

        best_rmse_sign = min(branch_results, key=lambda sign: float(branch_results[sign]["fit_summary"]["rmse_mpa"]))
        selected_sign = best_rmse_sign
        alternative_sign = -best_rmse_sign
        best_result = branch_results[best_rmse_sign]
        alt_result = branch_results[alternative_sign]
        if (
            int(alt_result["gamma_upper_hits"]) < int(best_result["gamma_upper_hits"])
            and float(alt_result["fit_summary"]["rmse_mpa"])
            <= float(best_result["fit_summary"]["rmse_mpa"]) * AUTO_FIT_PHYSICAL_BRANCH_RMSE_TOLERANCE
        ):
            selected_sign = alternative_sign

        selected = branch_results[selected_sign]
        fit_summary = dict(selected["fit_summary"])
        fit_summary["selected_by"] = "physical_branch" if selected_sign != best_rmse_sign else "lowest_rmse"
        fit_summary["peak_velocity_direction"] = cf_result["direction"]
        fit_summary["peak_arrival_times_s"] = arrival_times_s
        return {
            "fit_map": selected["fit_map"],
            "fit_summary": fit_summary,
            "branch_results": branch_results,
        }


if __name__ == "__main__":
    pass
