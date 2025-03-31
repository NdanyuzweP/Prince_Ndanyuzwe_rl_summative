import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import numpy as np
import os

class PatientMonitoringVisualization:
    def __init__(self, state):
        self.state = state  # State should be a 4-tuple (HR, BP, SpO2, Temp)
        self.width = 600
        self.height = 400
        self.running = True

    def draw(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glOrtho(0, self.width, 0, self.height, -1, 1)

        # Draw each vital sign as a colored circle
        self.draw_circle(100, 300, self.state[0], 'HR')
        self.draw_circle(200, 300, self.state[1], 'BP')
        self.draw_circle(300, 300, self.state[2], 'SpO2')
        self.draw_circle(400, 300, self.state[3], 'Temp')

        pygame.display.flip()

    def draw_circle(self, x, y, condition, label):
        colors = {
            0: (0, 1, 0),  # Green for normal
            1: (1, 1, 0),  # Yellow for mild concern
            2: (1, 0, 0)   # Red for critical
        }
        glColor3f(*colors[condition])

        num_segments = 50
        radius = 20
        glBegin(GL_POLYGON)
        for i in range(num_segments):
            angle = 2 * 3.14159 * i / num_segments
            glVertex2f(x + radius * np.cos(angle), y + radius * np.sin(angle))
        glEnd()

    def run(self):
        pygame.init()
        pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL)
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
            self.draw()
        pygame.quit()

# Testing the visualization
if __name__ == "__main__":
    state = [0, 1, 2, 0]  # Example state
    viz = PatientMonitoringVisualization(state)
    viz.run()
