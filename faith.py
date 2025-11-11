
import turtle

def repeating_squares():
    t = turtle.Turtle()
    for _ in range(100):
        for _ in range(4):
            t.forward(50)
            t.right(90)
        t.right(5)
    turtle.done()