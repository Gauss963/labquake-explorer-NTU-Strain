import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import scipy
from scipy import signal
import warnings

ZOOM_WINDOW_S = 0.01
ROSETTE_LABELS = ("B1", "B3", "B5", "B7")
ACTIVE_GAUGE_INDEX = {"B1": 6, "B3": 8, "B5": 10, "B7": 12}
GAUGE_COLOR_BY_INDEX = {6: "C0", 8: "C1", 10: "C3", 12: "C2"}


def subtract_pre_event_baseline_1d(values, relative_time_s):
    arr = np.asarray(values, dtype=np.float64).copy()
    tt = np.asarray(relative_time_s, dtype=np.float64)
    if arr.size != tt.size or arr.size == 0:
        return arr
    pre_event_mask = tt < 0.0
    if np.any(pre_event_mask):
        baseline = float(np.mean(arr[pre_event_mask]))
    else:
        baseline_width = max(1, min(arr.size, arr.size // 10 or 1))
        baseline = float(np.mean(arr[:baseline_width]))
    return arr - baseline


def compute_accelerometer_displacement_um(event, relative_time_s):
    """Estimate displacement from accelerometer data by double integration."""
    acceleration = event.get("acceleration_m_per_s2")
    if acceleration is None:
        return np.full_like(relative_time_s, np.nan, dtype=np.float64)

    tt = np.asarray(relative_time_s, dtype=np.float64)
    acc = np.asarray(acceleration, dtype=np.float64)
    if tt.size != acc.size or tt.size == 0:
        return np.full_like(tt, np.nan, dtype=np.float64)

    pre_event_mask = tt < 0.0
    if np.any(pre_event_mask):
        baseline = float(np.mean(acc[pre_event_mask]))
    else:
        baseline = float(np.mean(acc))
    acc = acc - baseline

    velocity = scipy.integrate.cumulative_trapezoid(acc, tt, initial=0.0)
    if np.any(pre_event_mask):
        velocity -= float(np.mean(velocity[pre_event_mask]))
    else:
        velocity -= velocity[0]

    displacement_m = scipy.integrate.cumulative_trapezoid(velocity, tt, initial=0.0)
    if np.any(pre_event_mask):
        displacement_m -= float(np.mean(displacement_m[pre_event_mask]))
    else:
        displacement_m -= displacement_m[0]

    return displacement_m * 1e6


def compute_strain_ylim(locations, enabled_channels):
    """Compute a tight y-limit around active strain-gauge locations."""
    active_locations = [
        float(locations[i]) for i, enabled in enumerate(enabled_channels) if enabled
    ]
    if not active_locations:
        return 1.0, -1.0

    ymin = min(active_locations)
    ymax = max(active_locations)
    if np.isclose(ymin, ymax):
        padding = max(20.0, abs(ymin) * 0.1 + 10.0)
    else:
        padding = max(20.0, 0.15 * (ymax - ymin))
    return ymax + padding, ymin - padding


def compute_channel_spacing_mm(locations, enabled_channels):
    """Estimate spacing between active gauges from their fault locations."""
    active_locations = sorted(
        {float(locations[i]) for i, enabled in enumerate(enabled_channels) if enabled}
    )
    if len(active_locations) < 2:
        return 100.0

    diffs = np.diff(active_locations)
    positive_diffs = diffs[diffs > 0.0]
    if positive_diffs.size == 0:
        return 100.0
    return float(np.min(positive_diffs))


def remove_pre_event_baseline(signals, relative_time_s):
    """Center each channel on its pre-event baseline for display."""
    centered = np.asarray(signals, dtype=np.float64).copy()
    tt = np.asarray(relative_time_s, dtype=np.float64)
    if centered.ndim != 2 or tt.size != centered.shape[1]:
        return centered

    pre_event_mask = tt < 0.0
    for i in range(centered.shape[0]):
        channel = centered[i]
        if np.any(pre_event_mask):
            baseline = float(np.mean(channel[pre_event_mask]))
        else:
            baseline_width = max(1, min(channel.size, channel.size // 10 or 1))
            baseline = float(np.mean(channel[:baseline_width]))
        centered[i] = channel - baseline
    return centered


def fallback_plot_data(event):
    event_time = float(event["event_time"])
    rel_time = np.asarray(event["time"], dtype=np.float64) - event_time
    tau_mpa = np.asarray(event["shear_stress"], dtype=np.float64)
    mu = np.divide(
        np.asarray(event["shear_stress"], dtype=np.float64),
        np.asarray(event["normal_stress"], dtype=np.float64),
        out=np.zeros_like(tau_mpa),
        where=np.abs(np.asarray(event["normal_stress"], dtype=np.float64)) > 1e-12,
    )
    delta_lp_um = subtract_pre_event_baseline_1d(
        np.asarray(event["LP_displacement"], dtype=np.float64) * 1e4,
        rel_time,
    )
    delta_acc_um = compute_accelerometer_displacement_um(event, rel_time)

    display_by_index = {}
    if "on_fault_shear_stress_mpa" in event["strain"]:
        strain_rel_time = np.asarray(event["strain"]["time"], dtype=np.float64) - event_time
        for label in ROSETTE_LABELS:
            if label in event["strain"]["on_fault_shear_stress_mpa"]:
                display_by_index[ACTIVE_GAUGE_INDEX[label]] = subtract_pre_event_baseline_1d(
                    np.asarray(event["strain"]["on_fault_shear_stress_mpa"][label], dtype=np.float64),
                    strain_rel_time,
                )
        lower_rel_time = strain_rel_time
    else:
        lower_rel_time = np.asarray(event["strain"]["original"]["time"], dtype=np.float64) - event_time
        raw = remove_pre_event_baseline(event["strain"]["original"]["raw"], lower_rel_time)
        for label, gauge_idx in ACTIVE_GAUGE_INDEX.items():
            if gauge_idx < raw.shape[0]:
                display_by_index[gauge_idx] = raw[gauge_idx]

    return {
        "top_rel_time": rel_time,
        "tau_mpa": tau_mpa,
        "mu": mu,
        "delta_lp_um": delta_lp_um,
        "delta_acc_um": delta_acc_um,
        "lower_rel_time": lower_rel_time,
        "lower_abs_time": lower_rel_time + event_time,
        "lower_by_index": display_by_index,
        "window_half_width_s": float(np.max(np.abs(lower_rel_time))) if lower_rel_time.size else ZOOM_WINDOW_S,
    }


def load_zoom_plot_data(event):
    if "pick_arrivals" not in event:
        return fallback_plot_data(event)

    pick_arrivals = event["pick_arrivals"]
    colors_by_label = pick_arrivals.get("colors_by_label", {})
    for label, gauge_idx in ACTIVE_GAUGE_INDEX.items():
        if label in colors_by_label:
            GAUGE_COLOR_BY_INDEX[gauge_idx] = colors_by_label[label]

    return {
        "top_rel_time": np.asarray(pick_arrivals["relative_time_s"], dtype=np.float64),
        "tau_mpa": np.asarray(pick_arrivals["tau_mpa"], dtype=np.float64),
        "mu": np.asarray(pick_arrivals["mu"], dtype=np.float64),
        "delta_lp_um": np.asarray(pick_arrivals["delta_lp_um"], dtype=np.float64),
        "delta_acc_um": np.asarray(pick_arrivals["delta_um"], dtype=np.float64),
        "lower_rel_time": np.asarray(pick_arrivals["relative_time_s"], dtype=np.float64),
        "lower_abs_time": np.asarray(pick_arrivals["time"], dtype=np.float64),
        "lower_by_index": {
            ACTIVE_GAUGE_INDEX[label]: np.asarray(values, dtype=np.float64)
            for label, values in pick_arrivals["delta_tau_mpa_by_label"].items()
            if label in ACTIVE_GAUGE_INDEX
        },
        "default_picked_idx_by_index": {
            ACTIVE_GAUGE_INDEX[label]: int(idx)
            for label, idx in pick_arrivals.get("default_picked_idx_by_label", {}).items()
            if label in ACTIVE_GAUGE_INDEX
        },
        "default_pick_time_by_index": {
            ACTIVE_GAUGE_INDEX[label]: float(value)
            for label, value in pick_arrivals.get("default_pick_time_by_label", {}).items()
            if label in ACTIVE_GAUGE_INDEX
        },
        "window_half_width_s": float(pick_arrivals.get("window_half_width_s", ZOOM_WINDOW_S)),
    }


class DynamicStrainArrivalPickerView(tk.Toplevel):
    def __init__(self, parent, run_idx, event_idx):
        self.parent = parent
        super().__init__(self.parent.root)
        self.title("Pick Arrivals")

        # Configure row and column properties
        # 7 columns, 4 rows
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(5, weight=1)


        # [0, 0]
        ttk.Label(self, text="Event Index:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.event_combobox = ttk.Combobox(self, width=10)
        # [0, 1]
        self.event_combobox.grid(row=0, column=1, padx=5, pady=5)
        self.enabled_channels_mb = tk.Menubutton(self, text="Enabled Channels")
        # [0, 2]
        self.enabled_channels_mb.grid(row=0, column=2, padx=5, pady=5)
        self.enabled_channels_mb.menu = tk.Menu(self.enabled_channels_mb, tearoff=0)
        self.enabled_channels_mb["menu"] = self.enabled_channels_mb.menu
        self.fitting_channels_mb = tk.Menubutton(self, text="Fitting Channels")
        # [0, 3]
        self.fitting_channels_mb.grid(row=0, column=3, padx=5, pady=5)
        self.fitting_channels_mb.menu = tk.Menu(self.fitting_channels_mb, tearoff=0)
        self.fitting_channels_mb["menu"] = self.fitting_channels_mb.menu
        # [0, 4]
        self.cf_label = ttk.Label(self, text="Cf=0.00m/s")
        self.cf_label.grid(row=0, column=4, padx=5, pady=5, sticky="e")
        # [0, 5]
        self.magic_button = tk.Button(self, text="Magic", command=self.magic)
        self.magic_button.grid(row=0, column=5, padx=5, pady=5, sticky="e")
        # [0, 6]
        self.save_button = tk.Button(self, text="Save", command=self.save)
        self.save_button.grid(row=0, column=6, padx=5, pady=5, sticky="e")

        # [1, 0::]
        self.fig = plt.figure(figsize=(7, 7), constrained_layout=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=1, column=0, columnspan=8, padx=5, pady=5, sticky="nsew")
        self.axs = None


        # [2, 0::]
        toolbar_frame = ttk.Frame(self)
        toolbar_frame.grid(row=2, column=0, columnspan=7, padx=0, pady=0, sticky="ew")
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()


        # [3, 0]
        ttk.Label(self, text="Filter:", justify="left").grid(row=3, column=0, padx=5, pady=5, sticky="ew")
        self.filter_combobox = ttk.Combobox(self, state="disabled")
        # [3, 1]
        self.filter_combobox.grid(row=3, column=1, padx=5, pady=5, sticky="ew")
        self.filter_combobox["values"] = ("scipy.savgol_filter")
        self.filter_combobox.current(0)
        # [3, 2]
        ttk.Label(self, text="Window length", justify="right").grid(row=3, column=2, padx=5, pady=5, sticky="w")
        # [3, 3]
        self.filter_window_length = tk.StringVar(value=51)
        self.filter_window_length_box = ttk.Spinbox(self, from_=2, to=201, increment=2, textvariable=self.filter_window_length)
        self.filter_window_length_box.grid(row=3, column=3, padx=5, pady=5, sticky="ew")
        # [3, 4]
        self.filter_toggle = tk.Button(self, text="Filter Off", relief="raised", command=self.toggle_filter)
        self.filter_toggle.grid(row=3, column=4, padx=5, pady=5, sticky="ew")
        

        # Data
        self.run_idx = run_idx
        self.event_idx = event_idx
        self.event = None
        self.enabled_channels = None
        self.fitting_channels = None
        self.picked_idx = None
        self.fitting_markers = []
        self.not_fitting_markers = []
        self.offset = [0, 0]
        self.current_artist = None
        self.currently_dragging = False
        self.rupture_speed = None
        self.fitted_line = None
        self.filtering = False
        self.xlim = None
        self.plot_data = None
        self.lower_time_abs = None
        self.init_event_combobox()
        self.on_selected_event_changed()
        self.on_resize()

        # Event bindings
        self.fig.canvas.mpl_connect("pick_event", self.on_pick)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("resize_event", self.on_resize)
        self.fig.canvas.mpl_connect("scroll_event", self.on_resize)
        self.event_combobox.bind("<<ComboboxSelected>>", self.on_selected_event_changed)
        self.filter_window_length_box.bind("<ButtonRelease>", self.on_filter_window_length_box_changed)

    def plot(self):
        exp_number = int(self.parent.data_manager.get_data("name")[1:5])
        # print(exp_number)
        linestyle = ".-"

        self.fig.clear()
        self.fitting_markers = []
        self.not_fitting_markers = []
        self.fitted_line = []
        gs = self.fig.add_gridspec(5, hspace=0, height_ratios=[1, 1, 1, 1, 10])
        self.axs = gs.subplots(sharex=True)
        self.axs[0].set_ylabel(r"$\tau$ (MPa)")
        self.axs[1].set_ylabel(r"$\mu$")
        self.axs[2].set_ylabel(r"$\delta_\mathrm{LP}\ \mathrm{({\mu}m)}$")
        self.axs[3].set_ylabel(r"$\delta\ \mathrm{({\mu}m)}$")

        self.plot_data = load_zoom_plot_data(self.event)
        t = np.asarray(self.plot_data["top_rel_time"], dtype=np.float64)
        tau_mpa = np.asarray(self.plot_data["tau_mpa"], dtype=np.float64)
        mu = np.asarray(self.plot_data["mu"], dtype=np.float64)
        delta_lp_um = np.asarray(self.plot_data["delta_lp_um"], dtype=np.float64)
        delta_acc_um = np.asarray(self.plot_data["delta_acc_um"], dtype=np.float64)

        self.axs[0].plot(t, tau_mpa, linestyle, color="C0")
        self.axs[1].plot(t, mu, linestyle, color="C0")
        self.axs[2].plot(t, delta_lp_um, linestyle, color="C0")
        self.axs[3].plot(t, delta_acc_um, linestyle, color="C0")

        tt = np.asarray(self.plot_data["lower_rel_time"], dtype=np.float64)
        self.lower_time_abs = np.asarray(self.plot_data["lower_abs_time"], dtype=np.float64)

        n_channels = len(self.event["strain"]["locations"])
        if self.enabled_channels is None:
            if "enabled_channels" in self.event["strain"]:
                self.enabled_channels = self.event["strain"]["enabled_channels"]
            else:
                if exp_number >= 5958:
                    self.enabled_channels = [i < 13 for i in range(n_channels)]
                else:
                    self.enabled_channels = [i % 2 == 0 for i in range(n_channels)]
        if self.fitting_channels is None:
            if "fitting_channels" in self.event["strain"]:
                self.fitting_channels = self.event["strain"]["fitting_channels"]
            else:
                if exp_number >= 5958:
                    self.fitting_channels = [False, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False]
                else: 
                    # self.fitting_channels = [i % 2 == 0 for i in range(n_channels)]
                    self.fitting_channels = [False, False, False, False, False, False, True, False, True, False, True, False, True, False, False, False]
        if (not "locations" in self.event["strain"]) or (not len(self.event["strain"]["locations"]) == n_channels):

            if exp_number >= 5958:
                self.event["strain"]["locations"] = [2 + 12 * i for i in range(16)]
                self.event["strain"]["locations"][-3:] = [2, 74, 146]
            else:
                self.event["strain"]["locations"] = [10.5 + 12 * int(i / 2) for i in range(n_channels)]
        
        self.lines = [None for i in range(n_channels)]

        if self.filtering:
            nf = int(self.filter_window_length.get())
            filtered_by_index = {}
            for i, values in self.plot_data["lower_by_index"].items():
                filtered_by_index[i] = scipy.signal.savgol_filter(np.asarray(values, dtype=np.float64), nf, 2)
        else:
            filtered_by_index = {
                i: np.asarray(values, dtype=np.float64)
                for i, values in self.plot_data["lower_by_index"].items()
            }
        line_idx = 0
        ratios = np.ones(n_channels)
        channel_spacing_mm = compute_channel_spacing_mm(
            self.event["strain"]["locations"], self.enabled_channels
        )
        target_span_mm = 0.8 * channel_spacing_mm
        for i in range(n_channels):
            if not self.enabled_channels[i]:
                continue
            if i not in filtered_by_index:
                continue
            loc = self.event["strain"]["locations"][i]
            self.axs[4].plot([tt[0], tt[-1]], [loc, loc], "k:", zorder=-101)
            values = filtered_by_index[i]
            amplitude = float(np.ptp(values))
            ratios[i] = 0.0 if np.isclose(amplitude, 0.0) else -target_span_mm / amplitude
            line_color = GAUGE_COLOR_BY_INDEX.get(i, f"C{line_idx}")
            self.lines[i] = self.axs[4].plot(tt, values * ratios[i] + loc, color=line_color, zorder=-100)
            line_idx += 1
        self.axs[4].set_ylabel("location along fault (mm)")
        self.axs[4].set_xlabel("time - %f (s)" %  self.event["event_time"])
        self.axs[4].set_ylim(
            *compute_strain_ylim(self.event["strain"]["locations"], self.enabled_channels)
        )
        if self.xlim is None:
            half_window_s = float(self.plot_data.get("window_half_width_s", ZOOM_WINDOW_S))
            self.axs[0].set_xlim(-half_window_s, half_window_s)
        else:
            self.axs[0].set_xlim(self.xlim)
        
        self.fig.suptitle("%s run%02d event%d" % (self.parent.data_manager.get_data("name"), 
                                                  self.run_idx, 
                                                  self.event_idx))
        
        if self.picked_idx is None:
            middle_idx = int(len(tt) / 2)
            self.picked_idx = [middle_idx for i in range(n_channels)]
            original = self.event["strain"]["original"]
            use_manual_picks = bool(original.get("manual_picks", False))

            if use_manual_picks:
                original_time_abs = np.asarray(original["time"], dtype=np.float64)
                saved_times_abs = None
                if "rupture_arrival_time" in original:
                    saved_times_abs = np.asarray(original["rupture_arrival_time"], dtype=np.float64)
                elif "picked_idx" in original:
                    saved_times_abs = np.asarray(
                        [
                            original_time_abs[min(max(int(idx), 0), len(original_time_abs) - 1)]
                            for idx in original["picked_idx"]
                        ],
                        dtype=np.float64,
                    )
                if saved_times_abs is not None and self.lower_time_abs is not None:
                    for i in range(min(n_channels, len(saved_times_abs))):
                        self.picked_idx[i] = int(np.argmin(np.abs(self.lower_time_abs - saved_times_abs[i])))
            else:
                default_idx_by_index = self.plot_data.get("default_picked_idx_by_index", {})
                default_time_by_index = self.plot_data.get("default_pick_time_by_index", {})
                for i in range(n_channels):
                    if i in default_idx_by_index:
                        self.picked_idx[i] = int(default_idx_by_index[i])
                    elif i in default_time_by_index and self.lower_time_abs is not None:
                        self.picked_idx[i] = int(
                            np.argmin(np.abs(self.lower_time_abs - float(default_time_by_index[i])))
                        )
                    elif self.lines[i] is not None:
                        _, y_display = self.lines[i][0].get_data()
                        self.picked_idx[i] = int(np.argmin(y_display))
        self.draw_markers()

    def draw_markers(self):
        width, height = self.get_circle_dims()
        for marker in self.fitting_markers:
            marker.remove()
        for marker in self.not_fitting_markers:
            marker.remove()
        self.fitting_markers = []
        self.not_fitting_markers = []
        for i in range(len(self.picked_idx)):
            if not self.enabled_channels[i]:
                continue
            idx = self.picked_idx[i]
            if self.fitting_channels[i]:
                color = "red"
            else:
                color = "black"
            (x , y) = self.lines[i][0].get_data()
            marker = patches.Ellipse((x[idx], y[idx]), width=width, height=height, color=color, fill=False, lw=2, picker=8, label=str(i))
            self.axs[4].add_patch(marker)
            if self.fitting_channels[i]:
                self.fitting_markers.append(marker)
            else:
                self.not_fitting_markers.append(marker)
        self.canvas.draw()


    def on_pick(self, event):
        if self.current_artist is None:
            self.current_artist = event.artist
            if isinstance(event.artist, patches.Ellipse):
                x0, y0 = self.current_artist.center
                x1, y1 = event.mouseevent.xdata, event.mouseevent.ydata
                self.offset = [(x0 - x1), (y0 - y1)]

    def on_motion(self, event):
        if not self.currently_dragging:
            return
        if self.current_artist is None:
            return
        if isinstance(self.current_artist, patches.Ellipse):
                try:
                    channel = int(self.current_artist.get_label())
                    dx, dy = self.offset
                    cx, cy = event.xdata + dx, event.ydata + dy
                    xl = self.axs[4].get_xlim()
                    yl = self.axs[4].get_ylim()
                    yw = yl[-1] - yl[0]
                    xw = xl[-1] - xl[0]
                    (x , y) = self.lines[channel][0].get_data()
                    idx = np.argmin(((x - cx) / xw) ** 2 + ((y - cy) / yw) ** 2)
                    self.current_artist.set_center((x[idx], y[idx]))
                    self.picked_idx[channel] = idx
                    self.update_fitted_line()
                except:
                    pass

    def on_press(self, event):
        self.currently_dragging = True

    def on_release(self, event):
        self.current_artist = None
        self.currently_dragging = False
        self.on_resize()

    def get_circle_dims(self):
        # self.update()
        # self.canvas.draw()
        xl = self.axs[4].get_xlim()
        yl = self.axs[4].get_ylim()
        ratio = (yl[-1] - yl[0]) / (xl[-1] - xl[0])
        bbox = self.axs[4].get_window_extent()
        ax_size = [bbox.width, bbox.height]
        ratio *= ax_size[0] / ax_size[1]
        width = (xl[-1] - xl[0]) / ax_size[0] * 0.1 * self.fig.dpi
        return width, width * ratio

    def on_resize(self, event=None):
        self.canvas.draw()
        if not self.axs is None:
            width, height = self.get_circle_dims()
            for marker in self.fitting_markers:
                marker.set_width(width)
                marker.set_height(height)
            for marker in self.not_fitting_markers:
                marker.set_width(width)
                marker.set_height(height)
            self.canvas.draw()
        self.xlim = self.axs[0].get_xlim()
    
    def update_fitted_line(self):
        x = np.empty(len(self.fitting_markers))
        y = np.empty(len(self.fitting_markers))
        for i in range(len(self.fitting_markers)):
            idx = int(self.fitting_markers[i].get_label())
            x[i] = self.fitting_markers[i].get_center()[0]
            y[i] = self.event["strain"]["locations"][idx]
        a = np.polyfit(y, x, 1)
        if self.fitted_line:
            self.fitted_line[0].remove()
        y_limits = self.axs[4].get_ylim()
        y_line = np.asarray([y_limits[0], y_limits[1]], dtype=np.float64)
        x_line = a[0] * y_line + a[1]
        self.fitted_line = self.axs[4].plot(x_line, y_line, "r--")
        self.canvas.draw()
        warnings.filterwarnings("ignore", message="divide by zero encountered in double_scalars")
        self.rupture_speed = -1e-3 / a[0]
        warnings.filterwarnings("default", message="divide by zero encountered in double_scalars")
        if np.abs(self.rupture_speed) < 1e4:
            self.cf_label.configure(text=f"Cf = {self.rupture_speed:.2f} m/s")
        else:
            self.cf_label.configure(text=f"Cf = {self.rupture_speed:.2e} m/s")
        self.update()
    
    def save(self):
        self.event["strain"]["enabled_channels"] = self.enabled_channels
        self.event["strain"]["fitting_channels"] = self.fitting_channels
        self.event["rupture_speed"] = self.rupture_speed
        original_time_abs = np.asarray(self.event["strain"]["original"]["time"], dtype=np.float64)
        saved_idx = []
        saved_time = []
        for idx in self.picked_idx:
            display_idx = min(max(int(idx), 0), len(self.lower_time_abs) - 1)
            target_time_abs = self.lower_time_abs[display_idx]
            original_idx = int(np.argmin(np.abs(original_time_abs - target_time_abs)))
            saved_idx.append(original_idx)
            saved_time.append(original_time_abs[original_idx])
        self.event["strain"]["original"]["picked_idx"] = saved_idx
        self.event["strain"]["original"]["rupture_arrival_time"] = np.asarray(saved_time, dtype=np.float64)
        self.event["strain"]["original"]["manual_picks"] = True
        self.parent.refresh_tree()
        print(f"Saved runs[{self.run_idx}]/events[{self.event_idx}] to data.")

    def init_event_combobox(self):
        n_events = len(self.parent.data_manager.get_data(f"runs/[{self.run_idx}]/events"))
        options = [f"{i}" for i in range(n_events)]
        self.event_combobox.config(values=options, state="readonly")
        self.event_combobox.current(self.event_idx)

    def init_enabled_channels_mb(self):
        n = len(self.enabled_channels)
        self.enabled_channels_mb.menu.delete(0, "end")
        self.enabled_channels_mb.items = [tk.IntVar() for i in range(n)]
        for i in range(n):
            self.enabled_channels_mb.items[i].set(self.enabled_channels[i])
            self.enabled_channels_mb.menu.add_checkbutton( label="channel %d" %i, variable=self.enabled_channels_mb.items[i], command=self.enabled_channels_changed)

    def init_fitting_channels_mb(self):
        n = len(self.fitting_channels)
        self.fitting_channels_mb.menu.delete(0, "end")
        self.fitting_channels_mb.items = [tk.IntVar() for i in range(n)]
        for i in range(n):
            self.fitting_channels_mb.items[i].set(self.fitting_channels[i])
            self.fitting_channels_mb.menu.add_checkbutton( label="channel %d" %i, variable=self.fitting_channels_mb.items[i], command=self.fitting_channels_changed)

    def on_selected_event_changed(self, event=None):
        self.event_idx = int(self.event_combobox.get())
        self.event = self.parent.data_manager.get_data(f"runs/[{self.run_idx}]/events/[{self.event_idx}]")
        if "enabled_channels" in self.event["strain"]:
            self.enabled_channels = self.event["strain"]["enabled_channels"]
        else:
            self.enabled_channels = None
        if "fitting_channels" in self.event["strain"]:
            self.fitting_channels = self.event["strain"]["fitting_channels"]
        else:
            self.fitting_channels = None
        self.picked_idx = None
        self.xlim = None
        self.fitting_markers = []
        self.not_fitting_markers = []
        self.offset = [0, 0]
        self.current_artist = None
        self.currently_dragging = False
        self.rupture_speed = None
        self.fitted_line = None
        self.plot_data = None
        self.lower_time_abs = None
        self.plot()
        self.init_enabled_channels_mb()
        self.init_fitting_channels_mb()
        self.update_fitted_line()
    

    def enabled_channels_changed(self):
        n = len(self.enabled_channels)
        for i in range(n):
            self.enabled_channels[i] = self.enabled_channels_mb.items[i].get()
        self.plot()
        self.update_fitted_line()

    def fitting_channels_changed(self):
        n = len(self.fitting_channels)
        for i in range(n):
            self.fitting_channels[i] = self.fitting_channels_mb.items[i].get()
        self.draw_markers()
        self.update_fitted_line()

    def toggle_filter(self):
        self.filtering = not self.filtering
        if self.filtering:
            self.filter_toggle.config(text="Filter On")
            self.filter_toggle.config(relief="sunken")
        else:
            self.filter_toggle.config(text="Filter Off")
            self.filter_toggle.config(relief="raised")
        self.plot()
        self.update_fitted_line()

    def on_filter_window_length_box_changed(self, event=None):
        if int(self.filter_window_length.get()) % 2 == 0:
            self.filter_window_length.set(int(self.filter_window_length.get()))
        self.plot()
        self.update_fitted_line()

    def magic(self):
        # self.event["strain"]["enabled_channels"] = [1,0, 1,0, 1,0, 1,0, 1,0, 1,0, 1,0, 1,0]
        # self.event["strain"]["fitting_channels"] = [0,0, 1,0, 1,0, 0,0, 0,0, 0,0, 0,0, 0,0]
        # self.on_selected_event_changed()
        # (x , y) = self.lines[0][0].get_data()
        # self.event["strain"]["original"]["picked_idx"][0] = np.argmin(y)
        # (x , y) = self.lines[2][0].get_data()
        # self.event["strain"]["original"]["picked_idx"][2] = np.argmin(y)
        # (x , y) = self.lines[4][0].get_data()
        # self.event["strain"]["original"]["picked_idx"][4] = np.argmin(y)
        # self.on_selected_event_changed()

        # self.event["strain"]["enabled_channels"] = [0,1, 0,1, 0,1, 0,1, 0,1, 0,1, 0,1, 0,1]
        # self.event["strain"]["fitting_channels"] = [0,0, 0,0, 0,0, 0,0, 0,1, 0,1, 0,1, 0,1]
        # self.on_selected_event_changed()

        for i in range(len(self.lines)):
            if self.lines[i] is None:
                continue
            (x , y) = self.lines[i][0].get_data()
            self.picked_idx[i] = np.argmin(y)
        self.draw_markers()

if __name__ == "__main__":
    pass
