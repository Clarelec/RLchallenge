import logging
import os
from utils import LOG_DIR
import torch
import  math
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
            width (int): largeur de la fenêtre tkinter
            height (int): hauteur de la fenêtre tkinter
            scale (float): facteur de conversion des unités de simulation vers pixels
            checkpoint_radius (float): rayon du checkpoint (en unités de simulation)
        """
        self.width = width
        self.height = height
        self.scale = scale
        self.checkpoint_radius = checkpoint_radius
        
        # Facteurs de visualisation des vecteurs
        self.speed_scale = 5.0      # agrandissement du vecteur vitesse
        self.vector_length = 30.0   # longueur des vecteurs unitaires (voile, safran, vent)

        self.root = tk.Tk()
        self.root.title("Simulation de Voilier")
        self.canvas = tk.Canvas(self.root, width=self.width, height=self.height, bg='lightblue')
        self.canvas.pack()
        logger.debug("Canvas created")
        
        
    def to_canvas_coords(self, x, y):
        """
        Convertit des coordonnées de simulation (avec (0,0) au centre et y positif vers le haut)
        en coordonnées tkinter (avec (0,0) en haut à gauche).
        """
        canvas_x = self.width / 2 + x * self.scale
        canvas_y = self.height / 2 - y * self.scale
        return canvas_x, canvas_y

    def afficher_etat(self, state, checkpoint=None, agent_index=0):
        """
        Affiche l'état de la simulation pour l'agent d'indice agent_index.
        
        Args:
            state (torch.Tensor): état de taille (batch_size, 10) dont
                - state[:, 0:2] : position
                - state[:, 2:4] : vitesse
                - state[:, 4:6] : direction de la voile (vecteur unitaire)
                - state[:, 6:8] : direction du safran (vecteur unitaire)
                - state[:, 8:10]: direction du vent (encodé en (cos, sin))
            checkpoint (torch.Tensor ou tuple): position du checkpoint pour l'agent.
                Par défaut, si None, le checkpoint est (0,0).
            agent_index (int): indice de l'agent à afficher.
        """
        
        # Effacer l'ancien dessin
        self.canvas.delete("all")
        logger.debug("Canvas cleared")
        # Déterminer le checkpoint (position en simulation)
        if checkpoint is None:
            cp_x, cp_y = 0, 0
        else:
            if isinstance(checkpoint, torch.Tensor):
                cp = checkpoint[agent_index].tolist()
                cp_x, cp_y = cp[0], cp[1]
            else:
                cp_x, cp_y = checkpoint

        # Convertir le checkpoint en coordonnées canvas et le dessiner
        cp_canvas = self.to_canvas_coords(cp_x, cp_y)
        cp_r = self.checkpoint_radius * self.scale
        self.canvas.create_oval(cp_canvas[0]-cp_r, cp_canvas[1]-cp_r,
                                cp_canvas[0]+cp_r, cp_canvas[1]+cp_r,
                                fill='red', outline='black')
        logger.debug("Checkpoint displayed")
        logger.info(f"Checkpoint coordinates : {cp_canvas}")
        # Extraction des différentes informations depuis l'état
        state_agent = state[agent_index]
        pos = state_agent[0:2].tolist()      # position [x, y]
        speed = state_agent[2:4].tolist()    # vitesse (vecteur)
        sail = state_agent[4:6].tolist()     # vecteur de la voile
        safran = state_agent[6:8].tolist()   # vecteur du safran
        wind = state_agent[8:10].tolist()    # vecteur du vent (cos, sin)

        # Calcul de l'angle de la vitesse (orientation du bateau)
        speed_norm = math.hypot(speed[0], speed[1])
        if speed_norm > 1e-3:
            heading = math.atan2(speed[1], speed[0])
        else:
            heading = 0
        
        # Calcul des angles (en degrés) pour affichage
        sail_angle = math.degrees(math.atan2(sail[1], sail[0]))
        safran_angle = math.degrees(math.atan2(safran[1], safran[0]))
        logger.info(f"Boat angles : {sail_angle} {safran_angle}")
        wind_angle = math.degrees(math.atan2(wind[1], wind[0]))
        
        # Conversion de la position du bateau en coordonnées canvas
        boat_canvas = self.to_canvas_coords(pos[0], pos[1])
        logger.info(f"Boat coordinates : {boat_canvas}")
        # Dessiner le bateau sous forme d'un triangle orienté selon la direction de la vitesse
        boat_size = 20  # en unités de simulation
        # Calcul des sommets du triangle en coordonnées simulation
        tip = ( pos[0] + boat_size * math.cos(heading),
                pos[1] + boat_size * math.sin(heading) )
        left = ( pos[0] + boat_size * 0.5 * math.cos(heading + math.radians(140)),
                 pos[1] + boat_size * 0.5 * math.sin(heading + math.radians(140)) )
        right = ( pos[0] + boat_size * 0.5 * math.cos(heading - math.radians(140)),
                  pos[1] + boat_size * 0.5 * math.sin(heading - math.radians(140)) )
        # Conversion en coordonnées canvas
        tip_canvas = self.to_canvas_coords(tip[0], tip[1])
        left_canvas = self.to_canvas_coords(left[0], left[1])
        right_canvas = self.to_canvas_coords(right[0], right[1])
        self.canvas.create_polygon(tip_canvas[0], tip_canvas[1],
                                   left_canvas[0], left_canvas[1],
                                   right_canvas[0], right_canvas[1],
                                   fill='white', outline='black')
        
        # Dessiner le vecteur vitesse (en bleu)
        speed_end = ( pos[0] + speed[0] * self.speed_scale,
                      pos[1] + speed[1] * self.speed_scale )
        self.canvas.create_line(boat_canvas[0], boat_canvas[1],
                                *self.to_canvas_coords(speed_end[0], speed_end[1]),
                                fill='blue', arrow=tk.LAST, width=2)
        
        # Dessiner le vecteur de la voile (en vert)
        sail_end = (  sail[0] * self.vector_length,
                      sail[1] * self.vector_length )
        self.canvas.create_line(boat_canvas[0], boat_canvas[1],
                                *self.to_canvas_coords(sail_end[0], sail_end[1]),
                                fill='green', arrow=tk.LAST, width=2)
        
        # Dessiner le vecteur du safran (en orange)
        safran_end = ( safran[0] * self.vector_length,
                        safran[1] * self.vector_length )
        self.canvas.create_line(boat_canvas[0], boat_canvas[1],
                                *self.to_canvas_coords(safran_end[0], safran_end[1]),
                                fill='orange', arrow=tk.LAST, width=2)
        
        # Dessiner la direction du vent (en violet) dans le coin supérieur gauche
        wind_start = (50, 50)  # en pixels, déjà dans le repère canvas
        wind_end = (wind_start[0] + self.vector_length * wind[0],
                    wind_start[1] - self.vector_length * wind[1])
        self.canvas.create_line(wind_start[0], wind_start[1],
                                wind_end[0], wind_end[1],
                                fill='purple', arrow=tk.LAST, width=2)
        self.canvas.create_text(wind_start[0], wind_start[1]-10,
                                text=f"Vent: {wind_angle:.1f}°",
                                fill='purple', font=('Helvetica', 10))
        
        # Afficher quelques informations textuelles près du bateau
        info_text = (f"Vitesse: {speed_norm:.1f}\n"
                     f"Voile: {sail_angle:.1f}°\n"
                     f"Safran: {safran_angle:.1f}°")
        self.canvas.create_text(10, self.height - 50, anchor='sw', text=info_text,
                                fill='black', font=('Helvetica', 12))
        logger.debug("State displayed")
        
        # Mettre à jour l'affichage
        self.root.update()

    def boucle(self):
        """Lance la boucle principale de tkinter."""
        self.root.mainloop()


                    