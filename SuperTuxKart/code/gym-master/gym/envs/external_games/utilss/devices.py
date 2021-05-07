import subprocess
from gym import logger


class SimulatedKeyboard:
    KEYS = [
        'BackSpace','Tab','Linefeed','Clear','Return','Pause','Scroll_Lock','Sys_Req','Escape',
        'Delete','Multi_key','Codeinput','SingleCandidate','MultipleCandidate','PreviousCandidate',
        'Home','Left','Up','Right','Down','Prior',
        'Page_Up','Next','Page_Down','End','Begin',
        'Select','Print','Execute','Insert',
        'Undo','Redo','Menu',
        'Find','Cancel','Help','Break','Mode_switch','script_switch',
        'Num_Lock','KP_Space','KP_Tab','KP_Enter',
        'KP_F1','KP_F2','KP_F3','KP_F4','KP_Home','KP_Left','KP_Up','KP_Right','KP_Down','KP_Prior','KP_Page_Up',
        'KP_Next','KP_Page_Down','KP_End','KP_Begin','KP_Insert','KP_Delete',
        'KP_Equal','KP_Multiply','KP_Add','KP_Separator','KP_Subtract','KP_Decimal','KP_Divide',
        'KP_0','KP_1','KP_2','KP_3','KP_4','KP_5','KP_6','KP_7','KP_8','KP_9',
        'F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','L1','F12','L2',
        'F13','L3','F14','L4','F15','L5','F16','L6','F17','L7','F18','L8','F19','L9',
        'F20','L10','F21','R1','F22','R2','F23','R3','F24','R4','F25','R5','F26','R6',
        'F27','R7','F28','R8','F29','R9','F30','R10','F31','R11','F32','R12','F33','R13','F34','R14','F35','R15',
        'Shift_L','Shift_R','Control_L','Control_R','Caps_Lock','Shift_Lock',
        'Meta_L','Meta_R','Alt_L','Alt_R','Super_L','Super_R','Hyper_L','Hyper_R',
        'space','exclam','quotedbl','numbersign','dollar','percent',
        'ampersand','apostrophe','quoteright','parenleft','parenright',
        'asterisk','plus','comma','minus','period','slash',
        '0','1','2','3','4','5','6','7','8','9',
        'colon','semicolon','less','equal','greater','question','at',
        'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
        'bracketleft','backslash','bracketright','asciicircum','underscore','grave','quoteleft',
        'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
        'braceleft','bar','braceright','asciitilde'
    ]
    
    KEY_TRANSLATE = {
        'enter': 'Return',
        'esc': 'Escape',
        
        'up': 'Up',
        'down': 'Down',
        'left': 'Left',
        'right': 'Right',
    }
    
    def __init__(self):
        self._pressed_keys = set()
    
    def key_press(self, key: str) -> None:
        """Simulates a key press on the keyboard for a given key.
        See https://cgit.freedesktop.org/xorg/proto/x11proto/plain/keysymdef.h.
        """
        logger.debug(f"[KBD] Press {key}")
        keycode = self._translate(key)
        subprocess.call(['xdotool', 'key', keycode])
        
        if keycode in self._pressed_keys:
            self._pressed_keys.remove(key)
    
    def key_down(self, key: str) -> None:
        """Simulates a key down action on the keyboard for a given key.
        See https://cgit.freedesktop.org/xorg/proto/x11proto/plain/keysymdef.h.
        """
        logger.debug(f"[KBD] Down {key}")
        keycode = self._translate(key)
        if keycode not in self._pressed_keys:
            self._pressed_keys.add(keycode)
            subprocess.call(['xdotool', 'keydown', keycode])
    
    def key_up(self, key: str) -> None:
        """Simulates a key up action on the keyboard for a given key.
        See https://cgit.freedesktop.org/xorg/proto/x11proto/plain/keysymdef.h.
        """
        logger.debug(f"[KBD] Up {key}")
        keycode = self._translate(key)
        if keycode in self._pressed_keys:
            self._pressed_keys.remove(keycode)
            subprocess.call(['xdotool', 'keyup', keycode])
    
    def enter(self) -> None:
        """Simulates an enter press on the keyboard.
        """
        logger.debug("[KBD] Press enter")
        self.key_press('Return')
        
    def esc(self) -> None:
        """Simulates an esc press on the keyboard.
        """
        logger.debug("[KBD] Press esc")
        self.key_press('Escape')
    
    def space(self) -> None:
        """Simulates a space press on the keyboard.
        """
        logger.debug("[KBD] Press space")
        self.key_press('space')
    
    def arrow_up(self) -> None:
        """Simulates an arrow up press on the keyboard.
        """
        logger.debug("[KBD] Press up")
        self.key_press('Up')
    
    def arrow_down(self) -> None:
        """Simulates an arrow down press on the keyboard.
        """
        logger.debug("[KBD] Press down")
        self.key_press('Down')
    
    def arrow_left(self) -> None:
        """Simulates an arrow left press on the keyboard.
        """
        logger.debug("[KBD] Press left")
        self.key_press('Left')
    
    def arrow_right(self) -> None:
        """Simulates an arrow right press on the keyboard.
        """
        logger.debug("[KBD] Press right")
        self.key_press('Right')
        
    def release_all(self) -> None:
        """Releases all the keys currently down
        """
        for keycode in self._pressed_keys:
            subprocess.call(['xdotool', 'keyup', keycode])
            
        self._pressed_keys.clear()
        
    def is_pressed(self, key: str) -> bool:
        """Returns true if the specified key is pressed, false otherwise
        """
        return key in self._pressed_keys
    
    def _translate(self, key: str) -> int:
        if key in SimulatedKeyboard.KEYS:
            return key
        else:
            return SimulatedKeyboard.KEY_TRANSLATE[key.lower()]


class SimulatedMouse:
    """A simulated mouse. Use it to simulate mouse movements or click actions
    """
    
    def __init__(self):
        import pyautogui
        self._pyautogui = pyautogui
    
    def left_click(self) -> None:
        """Simulates a mouse left button click
        """
        logger.debug("[MOUSE] Click")
        self._pyautogui.click()
    
    def right_click(self) -> None:
        """Simulates a mouse right button click
        """
        logger.debug("[MOUSE] Right Click")
        self._pyautogui.click(button='right')
    
    def move(self, x: int, y: int) -> None:
        """Simulates a mouse move action. It moves the cursor by x, y.
        """
        logger.debug("[MOUSE] Move")
        self._pyautogui.move(x, y)
