from pyray import *
import random

init_window(400, 200, "Hello GUI test")
set_target_fps(60)

showMessageBox = True

while not window_should_close():

    begin_drawing()
    clear_background(get_color(gui_get_style(GuiControlProperty.BASE_COLOR_NORMAL,GuiDefaultProperty.BACKGROUND_COLOR)))
    if(gui_button(Rectangle(24, 24, 120, 30), "Click me")):
        showMessageBox=True

    if(showMessageBox):
        result = gui_message_box(Rectangle(85, 70, 250, 100), "title","message", "nice:cool")
        if(result>=0):
            showMessageBox=False

    end_drawing()

close_window()