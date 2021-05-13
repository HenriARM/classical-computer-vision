# turtle is in python standard library
import turtle

ANGLE = 45
TREE_COLOR = (0, 0, 0)
TREE_SIZE = 100
TREE_LEVEL = 6


def tree(size, level, tree_color):
    if level > 0:
        # draw the base
        turtle.pencolor(tree_color)
        turtle.fd(size)
        turtle.rt(ANGLE)

        # draw right subtree
        tree(size=0.8 * size, level=level - 1, tree_color=tree_color)

        # turn to left subtree
        turtle.pencolor(tree_color)
        turtle.lt(2 * ANGLE)

        # draw left subtree
        tree(size=0.8 * size, level=level - 1, tree_color=tree_color)

        # go back
        turtle.pencolor(tree_color)
        turtle.rt(ANGLE)
        turtle.fd(-size)


if __name__ == '__main__':
    # canvas created, arrow is in center and on 0grad by x axis
    turtle.speed('normal')

    # rotate to right
    turtle.rt(-90)
    turtle.colormode(255)
    # tree of size 80 and level 7
    tree(size=TREE_SIZE, level=TREE_LEVEL, tree_color=TREE_COLOR)
    turtle.exitonclick()
