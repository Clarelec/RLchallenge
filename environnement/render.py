import logging
import os
from environnement.utils import LOG_DIR
import torch
import math

LOG_ADRESS = os.path.join(LOG_DIR, os.path.basename(__file__).split('.')[0]+'.log')
logger = logging.getLogger(__name__)    
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    filename= LOG_ADRESS,
                    filemode='w')

import tkinter as tk 

logger.debug("Libraries imported")

class RENDER_ENV:
    
    def __init__(self, width=800, height=600, scale=1.0, checkpoint_radius=10):
        """
        Args:
            width (int): width of the tkinter window
            height (int): height of the tkinter window
            scale (float): conversion factor from simulation units to pixels
            checkpoint_radius (float): radius of the checkpoint (in simulation units)
        """
        self.width = width
        self.height = height
        self.scale = scale
        self.checkpoint_radius = checkpoint_radius
        
        # Visualization factors for vectors
        self.speed_scale = 5.0      # enlargement of the speed vector
        self.vector_length = 30.0   # length of unit vectors (sail, rudder, wind)

        self.root = tk.Tk()
        self.root.title("Sailboat Simulation")
        self.canvas = tk.Canvas(self.root, width=self.width, height=self.height, bg='lightblue')
        self.canvas.pack()
        logger.debug("Canvas created")
        
    def to_canvas_coords(self, x, y):
        """
        Converts simulation coordinates (with (0,0) at the center and positive y upwards)
        to tkinter coordinates (with (0,0) at the top left).
        """
        canvas_x = self.width / 2 + x * self.scale
        canvas_y = self.height / 2 - y * self.scale
        return canvas_x, canvas_y

    def afficher_etat(self, state, checkpoint=None, agent_index=0):
        """
        Displays the state of the simulation for the agent with index agent_index.
        
        Args:
            state (torch.Tensor): state of size (batch_size, 10) where
                - state[:, 0:2] : position
                - state[:, 2:4] : speed
                - state[:, 4:6] : sail direction (unit vector)
                - state[:, 6:8] : rudder direction (unit vector)
                - state[:, 8:10]: wind direction (encoded as (cos, sin))
            checkpoint (torch.Tensor or tuple): position of the checkpoint for the agent.
                By default, if None, the checkpoint is (0,0).
            agent_index (int): index of the agent to display.
        """
        
        # Clear the previous drawing
        self.canvas.delete("all")
        logger.debug("Canvas cleared")
        # Determine the checkpoint (simulation position)
        if checkpoint is None:
            cp_x, cp_y = 0, 0
        else:
            if isinstance(checkpoint, torch.Tensor):
                cp = checkpoint[agent_index].tolist()
                cp_x, cp_y = cp[0], cp[1]
            else:
                cp_x, cp_y = checkpoint

        # Convert the checkpoint to canvas coordinates and draw it
        cp_canvas = self.to_canvas_coords(cp_x, cp_y)
        cp_r = self.checkpoint_radius * self.scale
        self.canvas.create_oval(cp_canvas[0]-cp_r, cp_canvas[1]-cp_r,
                                cp_canvas[0]+cp_r, cp_canvas[1]+cp_r,
                                fill='red', outline='black')
        logger.debug("Checkpoint displayed")
        logger.info(f"Checkpoint coordinates : {cp_canvas}")
        # Extract various information from the state
        state_agent = state[agent_index]
        pos = state_agent[0:2].tolist()      # position [x, y]
        speed = state_agent[2:4].tolist()    # speed (vector)
        sail = state_agent[4:6].tolist()     # sail vector
        safran = state_agent[6:8].tolist()   # rudder vector
        wind = state_agent[8:10].tolist()    # wind vector (cos, sin)

        # Calculate the speed angle (boat orientation)
        speed_norm = math.hypot(speed[0], speed[1])
        if speed_norm > 1e-3:
            heading = math.atan2(speed[1], speed[0])
        else:
            heading = 0
        
        # Calculate angles (in degrees) for display
        sail_angle = math.degrees(math.atan2(sail[1], sail[0]))
        safran_angle = math.degrees(math.atan2(safran[1], safran[0]))
        logger.info(f"Boat angles : {sail_angle} {safran_angle}")
        wind_angle = math.degrees(math.atan2(wind[1], wind[0]))
        
        # Convert the boat position to canvas coordinates
        boat_canvas = self.to_canvas_coords(pos[0], pos[1])
        logger.info(f"Boat coordinates : {boat_canvas}")
        # Draw the boat as a triangle oriented according to the speed direction
        boat_size = 20  # in simulation units
        # Calculate the vertices of the triangle in simulation coordinates
        tip = ( pos[0] + boat_size * math.cos(heading),
                pos[1] + boat_size * math.sin(heading) )
        left = ( pos[0] + boat_size * 0.5 * math.cos(heading + math.radians(140)),
                 pos[1] + boat_size * 0.5 * math.sin(heading + math.radians(140)) )
        right = ( pos[0] + boat_size * 0.5 * math.cos(heading - math.radians(140)),
                  pos[1] + boat_size * 0.5 * math.sin(heading - math.radians(140)) )
        # Convert to canvas coordinates
        tip_canvas = self.to_canvas_coords(tip[0], tip[1])
        left_canvas = self.to_canvas_coords(left[0], left[1])
        right_canvas = self.to_canvas_coords(right[0], right[1])
        self.canvas.create_polygon(tip_canvas[0], tip_canvas[1],
                                   left_canvas[0], left_canvas[1],
                                   right_canvas[0], right_canvas[1],
                                   fill='white', outline='black')
        
        # Draw the speed vector (in blue)
        speed_end = ( pos[0] + speed[0] * self.speed_scale,
                      pos[1] + speed[1] * self.speed_scale )
        self.canvas.create_line(boat_canvas[0], boat_canvas[1],
                                *self.to_canvas_coords(speed_end[0], speed_end[1]),
                                fill='blue', arrow=tk.LAST, width=2)
        
        # Draw the sail vector (in green)
        sail_end = ( pos[0]+ sail[0] * self.vector_length,
                      pos[1] +sail[1] * self.vector_length )
        self.canvas.create_line(boat_canvas[0], boat_canvas[1],
                                *self.to_canvas_coords(sail_end[0], sail_end[1]),
                                fill='green', arrow=tk.LAST, width=2)
        
        # Draw the rudder vector (in orange)
        safran_end = ( pos[0]+safran[0] * self.vector_length,
                        pos[1] + safran[1] * self.vector_length )
        self.canvas.create_line(boat_canvas[0], boat_canvas[1],
                                *self.to_canvas_coords(safran_end[0], safran_end[1]),
                                fill='orange', arrow=tk.LAST, width=2)
        
        # Draw the wind direction (in purple) in the top left corner
        wind_start = (50, 50)  # in pixels, already in canvas coordinates
        wind_end = (wind_start[0] + self.vector_length * wind[0],
                    wind_start[1] - self.vector_length * wind[1])
        self.canvas.create_line(wind_start[0], wind_start[1],
                                wind_end[0], wind_end[1],
                                fill='purple', arrow=tk.LAST, width=2)
        self.canvas.create_text(wind_start[0], wind_start[1]-10,
                                text=f"Wind: {wind_angle:.1f}°",
                                fill='purple', font=('Helvetica', 10))
        
        # Display some textual information near the boat
        info_text = (f"Speed: {speed_norm:.1f}\n"
                     f"Sail: {sail_angle:.1f}°\n"
                     f"Rudder: {safran_angle:.1f}°")
        self.canvas.create_text(10, self.height - 50, anchor='sw', text=info_text,
                                fill='black', font=('Helvetica', 12))
        logger.debug("State displayed")
        
        # Update the display
        self.root.update()

    def boucle(self):
        """Starts the main tkinter loop."""
        self.root.mainloop()



