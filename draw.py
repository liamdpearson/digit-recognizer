import arcade
import arcade.gui
import numpy as np # linear algebra

SCREEN_WIDTH, SCREEN_HEIGHT = 1000, 1000


class Draw(arcade.Window):
    """ digit neural net draw test """
    def __init__(self):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, "Digit Recognizer", fullscreen=False)
        self.set_location(50,50)
        arcade.set_background_color(arcade.color.GRAY)

        self.left_pressed = False

        self.img = np.zeros((784, 1))

        self.prediction = 0
        self.confidence = 0

        params = np.load("params.npz")
        self.W1, self.b1, self.W2, self.b2 = params['W1'], params['b1'], params['W2'], params['b2']


        #-- init gui --#
        self.button_style = {"font_name" : "Kenney Pixel", "font_size" : 40}

        self.manager = arcade.gui.UIManager()
        self.manager.enable()

        self.h_box = arcade.gui.UIBoxLayout(vertical=False)

        self.guess_button = arcade.gui.UIFlatButton(text="Guess", width=250, height=75, style=self.button_style)
        self.h_box.add(self.guess_button.with_space_around(right=20))

        self.clear_button = arcade.gui.UIFlatButton(text="Clear", width=250, height=75, style=self.button_style)
        self.h_box.add(self.clear_button.with_space_around(right=20))

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
                align_y = -425,
                align_x = -130,
                child=self.h_box)
        )


    def on_draw(self):
        arcade.start_render()

        arcade.draw_text("Draw a digit", 355, 
                             900, arcade.color.WHITE,
                             50, anchor_x="center", font_name="Kenney Mini Square")

        arcade.draw_rectangle_filled(360, 500, 700, 700, arcade.color.BLACK)

        for row in range(28):
            for col in range(28):

                mnist_row = 27 - row
                alpha = self.img[mnist_row*28 + col] * 255
                arcade.draw_rectangle_filled(col*25 + 22.5, 
                                             row*25 + 162.5,   25, 25, 
                                             (255, 255, 255, alpha))
        
        arcade.draw_rectangle_outline(360, 500, 700, 700, arcade.color.WHITE, 3)

        arcade.draw_text("Prediction: " + str(self.prediction), 850, 
                             550, arcade.color.WHITE,
                             20, anchor_x="center", font_name="Kenney Mini Square")
        arcade.draw_text("Confidence: " + str(self.confidence*100//1) + "%", 850,
                             450, arcade.color.WHITE,
                             20, anchor_x="center", font_name="Kenney Mini Square")
        
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
        if x >= 0 and x <= 700 and y >= 160 and y <= 860:
            mnist_row = 27 - ((y-160) * 28 // 700)   # flip y, MNIST row 0 = top
            mnist_col = x * 28 // 700

            self.img[mnist_row*28 + mnist_col, 0] += .6
    
            if mnist_row > 0:
                self.img[mnist_row*28 + mnist_col - 28, 0] += .3

            if mnist_row < 27:
                self.img[mnist_row*28 + mnist_col + 28, 0] += .3

            if mnist_col > 0:
                self.img[mnist_row*28 + mnist_col - 1, 0] += .3

            if mnist_col < 27:
                self.img[mnist_row*28 + mnist_col + 1, 0] += .3

    
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