import turtle

screen = turtle.Screen()
screen.title("STOP sign")
t = turtle.Turtle()
t.speed(2)

side_length = 100
t.penup()
t.goto(-50, 50)
t.pendown()
t.fillcolor("red")
t.begin_fill()

for _ in range(8):
    t.forward(side_length)
    t.right(45)

t.end_fill()

t.penup()
t.goto(-20, -140)
t.color("white")
t.write("FUCK YOU", align="CENTER", font =("Times New Romaan", 30, "bold"))

t.hideturtle()
turtle.done()