from threading import Thread
from time import sleep


class Movement:

    action = ''

    t_check_input = None

    success_flag = False        # Used to make the other processes aware of complete action?

    def __int__(self):
        #t_check_input = Thread(target=self.check_input)
        #t_check_input.start()
        pass

    def go_forward(self):
        print("Forward START")
        sleep(3)
        print("Forward STOP")

    def go_backward(self):
        print("Backward START")
        sleep(3)
        print("Backward STOP")

    def turn_right(self):
        print("Turning right START")
        sleep(3)
        print("Turning right STOP")

    def turn_left(self):
        print("Turning left START")
        sleep(3)
        print("Turning left STOP")

    def check_input(self):  # Made to loop in a separate thread to monitor action and send to movement controller

        while True:

            if self.action == 'w':
                self.do_movement(movement='forward')
                sleep(3)
            elif self.action == 'a':
                self.do_movement(movement='left')
                sleep(3)
            elif self.action == 's':
                self.do_movement(movement='backward')
                sleep(3)
            elif self.action == 'd':
                self.do_movement(movement='right')
                sleep(3)
            elif self.action == 'q':
                break

    def do_movement(self, movement):

        if movement == 'forward':
            t = Thread(target=self.go_forward)
            t.start()
        elif movement == 'backward':
            t = Thread(target=self.go_backward)
            t.start()
        elif movement == 'right':
            t = Thread(target=self.turn_right)
            t.start()
        elif movement == 'left':
            t = Thread(target=self.turn_left)
            t.start()




