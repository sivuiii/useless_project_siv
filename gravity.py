import pygetwindow as gw
import threading
import time
import ctypes

# --- CONFIGURATION ---
UPDATE_RATE_HZ = 60
GRAVITY = 1200.0  # Pixels per second squared. A higher value means faster falling.
FRICTION = 0.98   # How quickly a window loses horizontal speed. 1.0 = no friction.
THROW_MULTIPLIER = 1.5 # How much to multiply the mouse movement to make throws stronger.

# --- GLOBAL VARIABLES ---
# A dictionary to track the state of each window
# Key: Window HWND (unique ID), Value: WindowState object
managed_windows = {}
lock = threading.Lock()
running = True
screen_width, screen_height = 0, 0

class WindowState:
    """A class to hold the velocity and state for each window."""
    def __init__(self, window):
        self.window = window
        self.vx = 0  # Horizontal velocity in pixels per second
        self.vy = 0  # Vertical velocity in pixels per second
        self.last_x = window.left
        self.last_y = window.top
        self.is_user_controlled = False

def update_windows():
    """Single loop to discover, update, and move windows."""
    global running
    last_active_hwnd = None
    delta_time = 1.0 / UPDATE_RATE_HZ

    while running:
        # --- 1. DISCOVER AND SYNC WINDOWS ---
        try:
            current_windows = {w._hWnd: w for w in gw.getAllWindows() if w.visible and not w.isMinimized and w.title != ""}
        except Exception:
            continue

        with lock:
            # Add new windows
            for hwnd, win in current_windows.items():
                if hwnd not in managed_windows:
                    # --- FIX: Handle fleeting windows that close before they can be managed ---
                    try:
                        if win.width > 0 and win.height > 0:
                            print(f"Found new window: {win.title}")
                            managed_windows[hwnd] = WindowState(win)
                    except gw.PyGetWindowException:
                        # This happens when a window is destroyed between the getAllWindows() call
                        # and accessing its properties. We can safely ignore it.
                        print(f"Skipping a fleeting window.")
                        continue

            # Remove closed windows
            closed_windows = [hwnd for hwnd in managed_windows if hwnd not in current_windows]
            for hwnd in closed_windows:
                if hwnd in managed_windows:
                    print(f"Window closed: {managed_windows[hwnd].window.title}")
                    del managed_windows[hwnd]

            # --- 2. DETERMINE USER CONTROL ---
            active_win = gw.getActiveWindow()
            active_hwnd = active_win._hWnd if active_win else None

            for hwnd, state in managed_windows.items():
                state.is_user_controlled = (hwnd == active_hwnd)

            # --- 3. UPDATE VELOCITY AND POSITION FOR EACH WINDOW ---
            for hwnd, state in managed_windows.items():
                win = current_windows.get(hwnd)
                if not win: continue

                if state.is_user_controlled:
                    # User is dragging this window. Calculate velocity based on mouse movement.
                    state.vx = (win.left - state.last_x) / delta_time * THROW_MULTIPLIER
                    state.vy = (win.top - state.last_y) / delta_time * THROW_MULTIPLIER
                else:
                    # This window is in free-fall. Apply gravity.
                    state.vy += GRAVITY * delta_time
                    # Apply friction to horizontal movement
                    state.vx *= FRICTION

                # Update position based on velocity
                new_x = win.left + state.vx * delta_time
                new_y = win.top + state.vy * delta_time

                # Boundary checks and bounce
                if new_x <= 0 or new_x + win.width >= screen_width:
                    state.vx *= -0.5 # Bounce off sides with some energy loss
                if new_y <= 0 or new_y + win.height >= screen_height:
                    state.vy *= -0.4 # Bounce off top/bottom with more energy loss

                # Clamp position to stay within screen bounds
                clamped_x = max(0, min(new_x, screen_width - win.width))
                clamped_y = max(0, min(new_y, screen_height - win.height))

                # Move the window if it's not being controlled by the user
                if not state.is_user_controlled:
                    try:
                        if win.left != int(clamped_x) or win.top != int(clamped_y):
                            win.moveTo(int(clamped_x), int(clamped_y))
                    except gw.PyGetWindowException:
                        continue
                
                # Update last known position for the next frame
                state.last_x = clamped_x
                state.last_y = clamped_y


        time.sleep(delta_time)

def main():
    """Main function to start the threads."""
    global running, screen_width, screen_height
    print("Starting Simple Gravitational Windows...")
    print("Press Ctrl+C in this terminal to quit.")
    
    user32 = ctypes.windll.user32
    screen_width = user32.GetSystemMetrics(0)
    screen_height = user32.GetSystemMetrics(1)

    main_thread = threading.Thread(target=update_windows)
    main_thread.start()

    try:
        main_thread.join()
    except KeyboardInterrupt:
        print("\nShutting down...")
        running = False
        main_thread.join()
        print("Shutdown complete.")

if __name__ == "__main__":
    main()
