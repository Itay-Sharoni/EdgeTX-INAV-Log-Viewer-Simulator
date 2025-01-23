import csv
import datetime
import math
import time
import tkinter as tk
from tkinter import filedialog

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
import numpy as np

import matplotlib.patches as patches

#############################################################################
# 1) Configuration for Stick Channels & Reversals
#############################################################################

# Which channel corresponds to each stick axis?
LEFT_STICK_X_CHANNEL = 4
LEFT_STICK_Y_CHANNEL = 3
RIGHT_STICK_X_CHANNEL = 1
RIGHT_STICK_Y_CHANNEL = 2

# Reversal flags (False = normal, True = reversed)
LEFT_STICK_X_REVERSE = False
LEFT_STICK_Y_REVERSE = False
RIGHT_STICK_X_REVERSE = False
RIGHT_STICK_Y_REVERSE = False

# Battery column name in your CSV (change if your RC logs it differently)
RX_BATT_COLNAME = "RxBt(V)"

#############################################################################
# GLOBAL playback state
#############################################################################
is_paused = False
playback_speed = 1.0
start_wall_time = None
current_log_time = 0.0
min_log_time = 0.0
max_log_time = 1.0

#############################################################################
# We'll store all flight data + figure objects in a single dictionary: data
#############################################################################
data = {
    "rel_times": None,
    "pitch": None,
    "roll": None,
    "alt": None,
    "alt_max": 10,
    "gspd": None,
    "ch1": None,
    "ch2": None,
    "ch3": None,
    "ch4": None,
    "x_data": None,
    "y_data": None,
    "start_dt": None,
    "filename": None,
    # Battery array (or None if not found)
    "rx_bat": None,
    # placeholders for the figure artists
    "sky_patch": None,
    "ground_patch": None,
    "alt_line": None,
    "alt_text": None,
    "time_text": None,
    "left_stick": None,
    "right_stick": None,
    "track_line": None,
    "track_dot": None,
    "speed_text": None,
    "ax_gps": None,
    # We'll display battery text near the alt gauge
    "rx_bat_text": None,
    "drawn_artists": [],
}

#############################################################################
# 2) CSV Reading
#############################################################################

def parse_datetime(date_str, time_str):
    dt_str = f"{date_str} {time_str}"
    fmt = "%Y-%m-%d %H:%M:%S.%f"
    return datetime.datetime.strptime(dt_str, fmt)

def read_log_file(filepath):
    """
    Expects columns in CSV header (at least):
      Date, Time,
      Ptch(rad), Roll(rad), Yaw(rad), Alt(m),
      CH1(us), CH2(us), CH3(us), CH4(us),
      GPS (e.g. "31.876592 34.777213"),
      GSpd(kmh),
      (optional) RxBt(V) if you want battery.

    Returns a tuple of:
      (rel_times,
       pitch, roll, yaw, alt,
       ch1, ch2, ch3, ch4,
       lat_arr, lon_arr,
       gspd_arr,
       start_dt,
       rx_bat_arr or None)
    """
    with open(filepath, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            raise ValueError("CSV missing header or empty.")

        # Required indices
        try:
            i_date  = header.index("Date")
            i_time  = header.index("Time")
            i_pitch = header.index("Ptch(rad)")
            i_roll  = header.index("Roll(rad)")
            i_yaw   = header.index("Yaw(rad)")
            i_alt   = header.index("Alt(m)")
            i_ch1   = header.index("CH1(us)")
            i_ch2   = header.index("CH2(us)")
            i_ch3   = header.index("CH3(us)")
            i_ch4   = header.index("CH4(us)")
            i_gps   = header.index("GPS")
            i_gspd  = header.index("GSpd(kmh)")
        except ValueError:
            raise ValueError("Missing required columns in CSV (Date, Time, Ptch(rad), etc.).")

        # Battery col might not exist
        try:
            i_bat   = header.index(RX_BATT_COLNAME)
            have_batt = True
        except ValueError:
            i_bat = None
            have_batt = False

        datetimes = []
        pitch_list, roll_list, yaw_list, alt_list = [], [], [], []
        ch1_list, ch2_list, ch3_list, ch4_list = [], [], [], []
        lat_list, lon_list = [], []
        gspd_list = []
        rx_bat_list = [] if have_batt else None

        for row in reader:
            if not row:
                continue
            try:
                dt   = parse_datetime(row[i_date], row[i_time])
                p    = float(row[i_pitch])
                r    = float(row[i_roll])
                yw   = float(row[i_yaw])
                a    = float(row[i_alt])
                c1   = float(row[i_ch1])
                c2   = float(row[i_ch2])
                c3   = float(row[i_ch3])
                c4   = float(row[i_ch4])
                gps_str = row[i_gps].strip()
                lat_str, lon_str = gps_str.split()
                lat_val = float(lat_str)
                lon_val = float(lon_str)
                gspd_val= float(row[i_gspd])

                if have_batt:
                    bat_val = float(row[i_bat])
                else:
                    bat_val = None

            except:
                # skip invalid row
                continue

            datetimes.append(dt)
            pitch_list.append(p)
            roll_list.append(r)
            yaw_list.append(yw)
            alt_list.append(a)
            ch1_list.append(c1)
            ch2_list.append(c2)
            ch3_list.append(c3)
            ch4_list.append(c4)
            lat_list.append(lat_val)
            lon_list.append(lon_val)
            gspd_list.append(gspd_val)
            if have_batt:
                rx_bat_list.append(bat_val)

    if len(datetimes) < 2:
        raise ValueError("Not enough data rows in CSV to animate.")

    start_dt = datetimes[0]
    rel_times = np.array([(d - start_dt).total_seconds() for d in datetimes])

    if have_batt:
        rx_bat_array = np.array(rx_bat_list)
    else:
        rx_bat_array = None

    return (
        rel_times,
        np.array(pitch_list), np.array(roll_list), np.array(yaw_list),
        np.array(alt_list),
        np.array(ch1_list), np.array(ch2_list), np.array(ch3_list), np.array(ch4_list),
        np.array(lat_list), np.array(lon_list),
        np.array(gspd_list),
        start_dt,
        rx_bat_array
    )

def latlon_to_xy(lat_arr, lon_arr):
    lat0, lon0 = lat_arr[0], lon_arr[0]
    scale = 10000.0
    x_list, y_list = [], []
    for la, lo in zip(lat_arr, lon_arr):
        dlat = la - lat0
        dlon = lo - lon0
        x_list.append(dlon*scale)
        y_list.append(dlat*scale)
    return np.array(x_list), np.array(y_list)

#############################################################################
# 3) Utility & Channel Normalization
#############################################################################

def normalize_channel(value, vmin=1000, vmax=2000):
    """Map servo pulses (1000..2000) to -1..+1."""
    return 2.0 * (value - vmin) / (vmax - vmin) - 1.0

def find_frame_for_time(rel_times, t):
    """Simple clamp + linear search. For large logs, consider searchsorted."""
    n = len(rel_times)
    if t <= rel_times[0]:
        return 0
    if t >= rel_times[-1]:
        return n-1
    for i in range(n-1):
        if rel_times[i] <= t < rel_times[i+1]:
            return i
    return n-1

#############################################################################
# 4) Matplotlib Patches for Attitude
#############################################################################

def create_polygon(verts, color):
    poly = patches.Polygon(verts, facecolor=color)
    poly._original_verts = np.array(verts, dtype=float)
    return poly

def create_horizon_shapes():
    width, height = 10, 10
    sky_verts = [
        [-width,    0],
        [ width,    0],
        [ width, height],
        [-width, height]
    ]
    ground_verts = [
        [-width,     -height],
        [ width,     -height],
        [ width, 0],
        [-width, 0]
    ]
    sky = create_polygon(sky_verts, 'skyblue')
    ground = create_polygon(ground_verts, 'saddlebrown')
    return sky, ground

def transform_horizon(poly_patch, roll, pitch):
    pitch_scale = 5.0
    dy = pitch_scale * pitch
    c = math.cos(roll)
    s = math.sin(roll)
    R = np.array([[ c, -s],
                  [ s,  c ]])
    orig = poly_patch._original_verts
    new_verts = []
    for (x, y) in orig:
        xy_rot = R @ np.array([x, y])
        xy_rot[1] += dy
        new_verts.append(xy_rot)
    poly_patch.set_xy(new_verts)

def create_plane_symbol(ax):
    ax.plot([-0.5, 0.5], [0,0], color='black', lw=2)
    ax.plot([0,0], [0,0.2], color='black', lw=2)

#############################################################################
# 5) Real-Time Playback (Animation)
#############################################################################
def on_key(event):
    global is_paused, playback_speed, start_wall_time, current_log_time
    if event.key == ' ':
        # toggle pause
        is_paused = not is_paused
        if not is_paused:
            # unpausing
            start_wall_time = time.time()
    elif event.key == 'up':
        # speed up by +0.1
        playback_speed += 0.1
        playback_speed = round(playback_speed, 1)  # keep one decimal place
        print(f"Speed = {playback_speed:.1f}x")
    elif event.key == 'down':
        # speed down by -0.1, clamp min 0.1
        playback_speed = max(0.1, playback_speed - 0.1)
        playback_speed = round(playback_speed, 1)
        print(f"Speed = {playback_speed:.1f}x")
    elif event.key.lower() == 'r':
        # reset
        is_paused = True
        current_log_time = 0.0
        start_wall_time = None
        print("Playback reset to start.")

def animate(_frame):
    global is_paused, playback_speed, start_wall_time, current_log_time
    if data["rel_times"] is None:
        # no file loaded yet => do nothing
        return []

    if is_paused:
        return data["drawn_artists"]

    if start_wall_time is None:
        start_wall_time = time.time()

    elapsed = time.time() - start_wall_time
    delta_log = elapsed*abs(playback_speed)
    new_time = current_log_time + delta_log if (playback_speed>=0) else (current_log_time - delta_log)

    # clamp
    global min_log_time, max_log_time
    new_time = max(min_log_time, min(new_time, max_log_time))
    idx = find_frame_for_time(data["rel_times"], new_time)
    current_log_time = new_time
    start_wall_time = time.time()

    # gather
    roll  = data["roll"][idx]
    pitch = data["pitch"][idx]
    alt   = data["alt"][idx]
    t_sec = data["rel_times"][idx]
    gspd  = data["gspd"][idx]

    ch1_val = data["ch1"][idx]
    ch2_val = data["ch2"][idx]
    ch3_val = data["ch3"][idx]
    ch4_val = data["ch4"][idx]

    # battery if available
    rxbat_text = ""
    if data["rx_bat"] is not None:
        bat_val = data["rx_bat"][idx]
        rxbat_text = f"RxBat: {bat_val:.2f} V"
    else:
        rxbat_text = "RxBat: --"

    data["rx_bat_text"].set_text(rxbat_text)

    # 1) Attitude
    transform_horizon(data["sky_patch"], roll, pitch)
    transform_horizon(data["ground_patch"], roll, pitch)

    # 2) Altitude
    frac = 0
    if data["alt_max"]>0:
        frac = alt/data["alt_max"]
    frac = max(0, min(1, frac))
    data["alt_line"].set_data([1,1],[0,frac])
    data["alt_text"].set_position((0.8, frac))
    data["alt_text"].set_text(f"{alt:.1f} m")

    # 3) Time
    dt_now = data["start_dt"] + datetime.timedelta(seconds=t_sec)
    data["time_text"].set_text(dt_now.strftime("%H:%M:%S.%f")[:-3])

    # 4) Sticks
    def get_stick_value(ch_val, reverse_flag):
        n = normalize_channel(ch_val)
        return -n if reverse_flag else n

    # Left stick X
    if   LEFT_STICK_X_CHANNEL==1: lsx = get_stick_value(ch1_val, LEFT_STICK_X_REVERSE)
    elif LEFT_STICK_X_CHANNEL==2: lsx = get_stick_value(ch2_val, LEFT_STICK_X_REVERSE)
    elif LEFT_STICK_X_CHANNEL==3: lsx = get_stick_value(ch3_val, LEFT_STICK_X_REVERSE)
    elif LEFT_STICK_X_CHANNEL==4: lsx = get_stick_value(ch4_val, LEFT_STICK_X_REVERSE)
    else: lsx=0.0

    # Left stick Y
    if   LEFT_STICK_Y_CHANNEL==1: lsy = get_stick_value(ch1_val, LEFT_STICK_Y_REVERSE)
    elif LEFT_STICK_Y_CHANNEL==2: lsy = get_stick_value(ch2_val, LEFT_STICK_Y_REVERSE)
    elif LEFT_STICK_Y_CHANNEL==3: lsy = get_stick_value(ch3_val, LEFT_STICK_Y_REVERSE)
    elif LEFT_STICK_Y_CHANNEL==4: lsy = get_stick_value(ch4_val, LEFT_STICK_Y_REVERSE)
    else: lsy=0.0

    # Right stick X
    if   RIGHT_STICK_X_CHANNEL==1: rsx = get_stick_value(ch1_val, RIGHT_STICK_X_REVERSE)
    elif RIGHT_STICK_X_CHANNEL==2: rsx = get_stick_value(ch2_val, RIGHT_STICK_X_REVERSE)
    elif RIGHT_STICK_X_CHANNEL==3: rsx = get_stick_value(ch3_val, RIGHT_STICK_X_REVERSE)
    elif RIGHT_STICK_X_CHANNEL==4: rsx = get_stick_value(ch4_val, RIGHT_STICK_X_REVERSE)
    else: rsx=0.0

    # Right stick Y
    if   RIGHT_STICK_Y_CHANNEL==1: rsy = get_stick_value(ch1_val, RIGHT_STICK_Y_REVERSE)
    elif RIGHT_STICK_Y_CHANNEL==2: rsy = get_stick_value(ch2_val, RIGHT_STICK_Y_REVERSE)
    elif RIGHT_STICK_Y_CHANNEL==3: rsy = get_stick_value(ch3_val, RIGHT_STICK_Y_REVERSE)
    elif RIGHT_STICK_Y_CHANNEL==4: rsy = get_stick_value(ch4_val, RIGHT_STICK_Y_REVERSE)
    else: rsy=0.0

    data["left_stick"].set_offsets([lsx, lsy])
    data["right_stick"].set_offsets([rsx, rsy])

    # 5) GPS
    x = data["x_data"][idx]
    y = data["y_data"][idx]
    data["track_line"].set_data(data["x_data"][:idx+1], data["y_data"][:idx+1])
    data["track_dot"].set_data([x],[y])

    # 6) Ground Speed
    data["speed_text"].set_text(f"{gspd:.1f} km/h")

    return data["drawn_artists"]


#############################################################################
# 6) UI: "Load File" button, no reset button. We rely on 'R' key.
#############################################################################
def on_load_file(_event):
    """
    Button callback to pick a CSV, parse it, re-init everything,
    set suptitle, and auto-play.
    """
    global is_paused, current_log_time, start_wall_time, min_log_time, max_log_time
    filepath = filedialog.askopenfilename(
        title="Select CSV",
        filetypes=[("CSV Files","*.csv"), ("All Files","*.*")]
    )
    if not filepath:
        print("No file chosen.")
        return

    print(f"Loading file: {filepath}")
    try:
        (
            rel_times, pitch, roll, yaw, alt,
            ch1, ch2, ch3, ch4,
            lat_arr, lon_arr,
            gspd,
            start_dt,
            rx_bat
        ) = read_log_file(filepath)
    except Exception as e:
        print(f"Failed to load file: {e}")
        return

    # store in data
    data["rel_times"] = rel_times
    data["pitch"] = pitch
    data["roll"] = roll
    data["alt"]  = alt
    data["gspd"] = gspd
    data["ch1"]  = ch1
    data["ch2"]  = ch2
    data["ch3"]  = ch3
    data["ch4"]  = ch4
    data["start_dt"] = start_dt
    data["filename"] = filepath
    data["rx_bat"]   = rx_bat  # None if not found

    # min/max log_time
    min_log_time = rel_times[0]
    max_log_time = rel_times[-1]

    alt_max = max(alt.max(), 10)
    data["alt_max"] = alt_max

    # lat/lon -> x_data,y_data
    x_data, y_data = latlon_to_xy(lat_arr, lon_arr)
    data["x_data"] = x_data
    data["y_data"] = y_data

    # set up x/y limits for the gps
    ax_gps = data["ax_gps"]
    x_min, x_max = x_data.min(), x_data.max()
    y_min, y_max = y_data.min(), y_data.max()
    dx = x_max - x_min
    dy = y_max - y_min
    margin_x = 0.1*dx
    margin_y = 0.1*dy
    ax_gps.set_xlim(x_min - margin_x, x_max + margin_x)
    ax_gps.set_ylim(y_min - margin_y, y_max + margin_y)

    # reset playback, but auto-play
    is_paused = False
    current_log_time = 0.0
    start_wall_time = None

    # update figure suptitle
    plt.suptitle(f"Loaded: {filepath}")

    print("File loaded successfully. Auto-play started.")
    print("Press R to reset, Space to pause/play, Up/Down to change speed.")


#############################################################################
# 7) Main: build the figure, create "Load File" button, etc.
#############################################################################
def main():
    global data
    global min_log_time, max_log_time

    # build figure
    fig = plt.figure(figsize=(10,8))
    # bigger bottom row for GPS
    gs = fig.add_gridspec(3,2, height_ratios=[2,2,3], width_ratios=[2,1])

    # Attitude indicator
    ax_att = fig.add_subplot(gs[0,0])
    ax_att.set_title("Attitude Indicator")
    ax_att.set_xlim(-2, 2)
    ax_att.set_ylim(-2, 2)
    ax_att.set_xticks([])
    ax_att.set_yticks([])
    ax_att.axhline(0,color='gray',lw=0.5)
    ax_att.axvline(0,color='gray',lw=0.5)

    sky_patch, ground_patch = create_horizon_shapes()
    sky_patch = ax_att.add_patch(sky_patch)
    ground_patch = ax_att.add_patch(ground_patch)
    create_plane_symbol(ax_att)

    time_text = ax_att.text(0.05,0.9,"No File", transform=ax_att.transAxes, fontsize=10)

    # Alt gauge
    ax_alt = fig.add_subplot(gs[0,1])
    ax_alt.set_title("Altitude")
    ax_alt.set_xlim(0,1)
    ax_alt.set_ylim(0,1)
    ax_alt.set_xticks([])
    ax_alt.set_yticks([])
    alt_line, = ax_alt.plot([1,1],[0,0], lw=10, color='blue')
    alt_text = ax_alt.text(0.8,0.0,"0.0 m", fontsize=10, color='blue',
                           ha='right', va='center', transform=ax_alt.transAxes)

    # Battery text (near top-left of alt gauge)
    rx_bat_text = ax_alt.text(0.05, 0.95, "RxBat: --", fontsize=9, color='magenta',
                              transform=ax_alt.transAxes, ha='left', va='top')

    # Left stick
    ax_left_stick = fig.add_subplot(gs[1,0])
    ax_left_stick.set_title("Left Stick")
    ax_left_stick.set_xlim(-1,1)
    ax_left_stick.set_ylim(-1,1)
    ax_left_stick.axhline(0,color='gray',lw=0.5,ls='--')
    ax_left_stick.axvline(0,color='gray',lw=0.5,ls='--')
    left_stick = ax_left_stick.scatter([0],[0], c='red', s=80)

    # Right stick
    ax_right_stick = fig.add_subplot(gs[1,1])
    ax_right_stick.set_title("Right Stick")
    ax_right_stick.set_xlim(-1,1)
    ax_right_stick.set_ylim(-1,1)
    ax_right_stick.axhline(0,color='gray',lw=0.5,ls='--')
    ax_right_stick.axvline(0,color='gray',lw=0.5,ls='--')
    right_stick = ax_right_stick.scatter([0],[0], c='green', s=80)

    # GPS track
    ax_gps = fig.add_subplot(gs[2,:])
    ax_gps.set_title("GPS Track (No Map)")
    ax_gps.set_xlabel("X offset")
    ax_gps.set_ylabel("Y offset")
    track_line, = ax_gps.plot([],[], color='blue', lw=2)
    track_dot,  = ax_gps.plot([],[], 'ro', markersize=5)
    speed_text = ax_gps.text(0.05,0.9,"0.0 km/h", transform=ax_gps.transAxes, fontsize=10)

    # store references in data
    data["sky_patch"] = sky_patch
    data["ground_patch"] = ground_patch
    data["time_text"] = time_text
    data["alt_line"] = alt_line
    data["alt_text"] = alt_text
    data["left_stick"] = left_stick
    data["right_stick"] = right_stick
    data["track_line"] = track_line
    data["track_dot"]  = track_dot
    data["speed_text"] = speed_text
    data["ax_gps"]     = ax_gps
    data["rx_bat_text"] = rx_bat_text

    # We'll just start with empty arrays => no motion until user loads a file
    data["drawn_artists"] = [
        sky_patch, ground_patch,
        alt_line, alt_text,
        time_text, left_stick, right_stick,
        track_line, track_dot,
        speed_text, rx_bat_text
    ]

    # HEADLINE & USAGE
    usage_text = (
        "Press 'Load File' to pick CSV\n"
        "Space = Pause/Play\n"
        "R = Reset\n"
        "Up/Down = Speed +/- 0.1\n"
    )
    fig.text(0.01, 0.98, usage_text, fontsize=9, va='top')

    # Add a "Load File" button
    load_ax = fig.add_axes([0.0, 0.0, 0.1, 0.05])  # x,y,w,h in figure fraction
    load_button = Button(load_ax, "Load File", color='lightgray', hovercolor='0.95')
    load_button.on_clicked(on_load_file)

    # Key press
    fig.canvas.mpl_connect("key_press_event", on_key)

    # Setup the animation
    ani = FuncAnimation(fig, animate,
                        frames=np.repeat(0,999999),
                        interval=50,
                        blit=False,
                        repeat=False)

    plt.suptitle("No file loaded")
    plt.show()

if __name__ == "__main__":
    main()
