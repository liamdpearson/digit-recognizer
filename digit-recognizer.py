import sys, os
import arcade
import arcade.gui
import numpy as np # linear algebra

sw, sh = arcade.window_commands.get_display_size()

SCREEN_WIDTH, SCREEN_HEIGHT = 3*sh//4, 3*sh//4


class Draw(arcade.Window):
    """ digit neural net draw test """
    def __init__(self):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, "Digit Recognizer", fullscreen=False)
        self.set_location(sw//2 - 3*sh//8,sh//8)
        arcade.set_background_color(arcade.color.GRAY)

        self.left_pressed = False

        self.img = np.zeros((784, 1))

        self.prediction = 0
        self.confidence = 0

        base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
        params = np.load(os.path.join(base, "params.npz"))
        self.W1, self.b1, self.W2, self.b2 = params['W1'], params['b1'], params['W2'], params['b2']


        #-- init gui --#
        self.button_style = {"font_name" : "Kenney Pixel", "font_size" : SCREEN_WIDTH*.040}

        self.manager = arcade.gui.UIManager()
        self.manager.enable()

        self.h_box = arcade.gui.UIBoxLayout(vertical=False)

        self.guess_button = arcade.gui.UIFlatButton(text="Guess", width=SCREEN_WIDTH*.250, height=SCREEN_WIDTH*.075, style=self.button_style)
        self.h_box.add(self.guess_button.with_space_around(right=SCREEN_WIDTH*.020))

        self.clear_button = arcade.gui.UIFlatButton(text="Clear", width=SCREEN_WIDTH*.250, height=SCREEN_WIDTH*.075, style=self.button_style)
        self.h_box.add(self.clear_button.with_space_around(right=SCREEN_WIDTH*.020))

        @self.guess_button.event("on_click")
        def on_click_settings(event):
            self.test(self.W1, self.b1, self.W2, self.b2)

        @self.clear_button.event("on_click")
        def on_click_settings(event):
            self.img = np.zeros((784, 1))
            self.prediction = 0
            self.confidence = 0

        self.manager.add(
            arcade.gui.UIAnchorWidget(
                anchor_x="center_x",
                anchor_y="center_y",
                align_y = -SCREEN_HEIGHT*.425,
                align_x = -SCREEN_WIDTH*.130,
                child=self.h_box)
        )


    def on_draw(self):
        arcade.start_render()

        arcade.draw_text("Draw a digit", SCREEN_WIDTH*.355, 
                             SCREEN_HEIGHT*.9, arcade.color.WHITE,
                             SCREEN_WIDTH*.050, anchor_x="center", font_name="Kenney Mini Square")

        arcade.draw_rectangle_filled(SCREEN_WIDTH*.36, SCREEN_HEIGHT*.5, SCREEN_WIDTH*0.7, SCREEN_HEIGHT*0.7, arcade.color.BLACK)

        for row in range(28):
            for col in range(28):

                mnist_row = 27 - row
                alpha = self.img[mnist_row*28 + col] * 255
                arcade.draw_rectangle_filled(col*SCREEN_WIDTH*0.025 + SCREEN_HEIGHT*.0225, 
                                             row*SCREEN_HEIGHT*0.025 + SCREEN_HEIGHT*.1625,           SCREEN_HEIGHT*0.025, SCREEN_HEIGHT*0.025, 
                                             (255, 255, 255, alpha))
        
        arcade.draw_rectangle_outline(SCREEN_WIDTH*.36, SCREEN_HEIGHT*.5, SCREEN_WIDTH*0.7, SCREEN_HEIGHT*0.7, arcade.color.WHITE, 3)

        arcade.draw_text("Prediction: " + str(self.prediction), SCREEN_WIDTH*.85, 
                             SCREEN_HEIGHT*.55, arcade.color.WHITE,
                             SCREEN_WIDTH*.02, anchor_x="center", font_name="Kenney Mini Square")
        arcade.draw_text("Confidence: " + str(self.confidence*100//1) + "%", SCREEN_WIDTH*.85, 
                             SCREEN_HEIGHT*.45, arcade.color.WHITE,
                             SCREEN_WIDTH*.02, anchor_x="center", font_name="Kenney Mini Square")
        
        self.manager.draw()
    
    def on_mouse_press(self, x, y, button, modifiers):
        if button == arcade.MOUSE_BUTTON_LEFT:
            self.left_pressed = True
            self.draw_point(x, y)
        
    
    def on_mouse_release(self, x, y, button, modifiers):
        if button == arcade.MOUSE_BUTTON_LEFT:
            self.left_pressed = False


    def on_mouse_motion(self, x, y, dx, dy):
        if self.left_pressed:
            self.draw_point(x, y)


    def draw_point(self, x, y):
        if x >= SCREEN_WIDTH*0.005 and x <= SCREEN_WIDTH*0.705 and y >= SCREEN_HEIGHT*0.15 and y <= SCREEN_HEIGHT*0.85:
            mnist_row = int(27 - ((y-SCREEN_HEIGHT*0.15) * 28) // int(SCREEN_WIDTH*0.705))   # flip y, MNIST row 0 = top
            mnist_col = int((x-SCREEN_WIDTH*0.005) * 28) // int(SCREEN_WIDTH*0.7)


            self.img[mnist_row*28 + mnist_col, 0] += (-self.img[mnist_row*28 + mnist_col, 0]+1)*.75
    
            if mnist_row > 0:
                self.img[mnist_row*28 + mnist_col - 28, 0] += (-self.img[mnist_row*28 + mnist_col - 28, 0]+1)*.45

            if mnist_row < 27:
                self.img[mnist_row*28 + mnist_col + 28, 0] += (-self.img[mnist_row*28 + mnist_col + 28, 0]+1)*.45

            if mnist_col > 0:
                self.img[mnist_row*28 + mnist_col - 1, 0] += (-self.img[mnist_row*28 + mnist_col - 1, 0]+1)*.45

            if mnist_col < 27:
                self.img[mnist_row*28 + mnist_col + 1, 0] += (-self.img[mnist_row*28 + mnist_col + 1, 0]+1)*.45

    
    def test(self, W1, b1, W2, b2):
        Z1 = W1.dot(self.img) + b1
        A1 = ReLU(Z1)
        Z2 = W2.dot(A1) + b2
        A2 = softmax(Z2)

        self.prediction = int(np.argmax(A2, axis=0)[0])
        self.confidence = float(A2[self.prediction, 0])



def ReLU(Z):
    return np.maximum(0, Z)


def softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)


def main():
    window = Draw()
    arcade.run()

if __name__ == "__main__":
    main()