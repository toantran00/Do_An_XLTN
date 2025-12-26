import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageOps
import os
import numpy as np
import joblib
import librosa
import config
import pygame

# --- MODERN UI THEME ---
COLOR_PALETTE = {
    "primary": "#1a237e",      # Deep blue
    "primary_light": "#5c6bc0",
    "secondary": "#0d47a1",    # Darker blue
    "accent": "#00acc1",       # Teal accent
    "success": "#4caf50",      # Green
    "warning": "#ff9800",      # Orange
    "error": "#f44336",        # Red
    "dark": "#263238",         # Dark gray
    "light": "#f5f5f5",        # Light gray background
    "white": "#ffffff",
    "sidebar": "#ffffff",      # White sidebar
    "card": "#ffffff",         # White card
    "border": "#e0e0e0",       # Border color
}

# Fonts
FONT_FAMILY = "Segoe UI"
TITLE_FONT = (FONT_FAMILY, 24, "bold")
HEADER_FONT = (FONT_FAMILY, 18, "bold")
SUBHEADER_FONT = (FONT_FAMILY, 12, "bold")
BODY_FONT = (FONT_FAMILY, 11)
BUTTON_FONT = (FONT_FAMILY, 10, "bold")

# ID to Name mapping
ID_TO_NAME = {
    0: "Air Conditioner",
    1: "Car Horn",
    2: "Children Playing",
    3: "Dog Bark",
    4: "Drilling",
    5: "Engine Idling",
    6: "Gun Shot",
    7: "Jackhammer",
    8: "Siren",
    9: "Street Music"
}

class ProfessionalApp:
    def __init__(self, root):
        self.root = root
        self.root.title("URBAN SOUND CLASSIFICATION SYSTEM")
        self.root.geometry("1200x720")
        self.root.resizable(False, False)
        
        # Configure style
        self.root.configure(bg=COLOR_PALETTE["light"])
        
        # Initialize mixer
        pygame.mixer.init()
        
        # Load model
        self.load_model()
        self.current_file = None
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        # Create main container
        main_container = tk.Frame(self.root, bg=COLOR_PALETTE["light"])
        main_container.pack(fill="both", expand=True, padx=20, pady=20)
        
        # --- LEFT SIDEBAR ---
        sidebar = tk.Frame(main_container, bg=COLOR_PALETTE["sidebar"], width=320,
                          relief="flat", highlightbackground=COLOR_PALETTE["border"],
                          highlightthickness=1)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        
        # Sidebar content
        self.setup_sidebar(sidebar)
        
        # --- MAIN CONTENT AREA ---
        main_content = tk.Frame(main_container, bg=COLOR_PALETTE["light"])
        main_content.pack(side="right", fill="both", expand=True, padx=(20, 0))
        
        # Main content
        self.setup_main_content(main_content)
    
    def setup_sidebar(self, parent):
        # Logo/Top Section
        logo_frame = tk.Frame(parent, bg=COLOR_PALETTE["primary"], height=100)
        logo_frame.pack(fill="x")
        logo_frame.pack_propagate(False)
        
        tk.Label(logo_frame, text="URBAN SOUND\nANALYZER", 
                font=TITLE_FONT, bg=COLOR_PALETTE["primary"], 
                fg=COLOR_PALETTE["white"], justify="center").pack(expand=True)
        
        # File Input Section
        input_frame = tk.Frame(parent, bg=COLOR_PALETTE["white"], padx=20, pady=20)
        input_frame.pack(fill="x", pady=(20, 10))
        
        tk.Label(input_frame, text="AUDIO INPUT", font=HEADER_FONT,
                bg=COLOR_PALETTE["white"], fg=COLOR_PALETTE["primary"]).pack(anchor="w", pady=(0, 15))
        
        # Select File Button
        self.btn_select = self.create_modern_button(
            input_frame, "📁 SELECT AUDIO FILE", COLOR_PALETTE["accent"], self.select_file,
            width=280
        )
        self.btn_select.pack(pady=(0, 15))
        
        # Selected File Display
        file_display = tk.Frame(input_frame, bg=COLOR_PALETTE["light"], height=50,
                               relief="flat", highlightbackground=COLOR_PALETTE["border"],
                               highlightthickness=1)
        file_display.pack(fill="x")
        file_display.pack_propagate(False)
        
        self.lbl_filename = tk.Label(file_display, 
                                    text="No file selected",
                                    font=BODY_FONT,
                                    bg=COLOR_PALETTE["light"], fg=COLOR_PALETTE["dark"],
                                    wraplength=260, justify="left")
        self.lbl_filename.pack(padx=15, pady=10, anchor="w")
        
        # Analyze Button
        self.btn_process = self.create_modern_button(
            input_frame, "⚡ ANALYZE AUDIO", COLOR_PALETTE["success"], self.process,
            state="disabled", width=280
        )
        self.btn_process.pack(pady=10)
        
        # Audio Player Section
        player_frame = tk.Frame(input_frame, bg=COLOR_PALETTE["white"])
        player_frame.pack(fill="x", pady=(20, 0))
        
        tk.Label(player_frame, text="AUDIO PLAYER", font=SUBHEADER_FONT,
                bg=COLOR_PALETTE["white"], fg=COLOR_PALETTE["primary"]).pack(anchor="w", pady=(0, 10))
        
        # Player Buttons
        player_controls = tk.Frame(player_frame, bg=COLOR_PALETTE["white"])
        player_controls.pack(fill="x")
        
        self.btn_play = self.create_modern_button(
            player_controls, "▶ PLAY", COLOR_PALETTE["accent"], self.play_audio,
            state="disabled", width=12
        )
        self.btn_play.pack(side="left", padx=(0, 10))
        
        self.btn_stop = self.create_modern_button(
            player_controls, "⏹ STOP", COLOR_PALETTE["error"], self.stop_audio,
            state="disabled", width=12
        )
        self.btn_stop.pack(side="left")
    
    def setup_main_content(self, parent):
        # Header
        header_frame = tk.Frame(parent, bg=COLOR_PALETTE["light"])
        header_frame.pack(fill="x", pady=(0, 20))
        
        tk.Label(header_frame, text="URBAN SOUND CLASSIFICATION",
                font=("Segoe UI", 28, "bold"), bg=COLOR_PALETTE["light"],
                fg=COLOR_PALETTE["primary"]).pack(anchor="w")
        
        tk.Label(header_frame, text="AI-powered sound recognition system",
                font=SUBHEADER_FONT, bg=COLOR_PALETTE["light"],
                fg=COLOR_PALETTE["dark"]).pack(anchor="w", pady=(5, 0))
        
        # Results Card
        results_card = tk.Frame(parent, bg=COLOR_PALETTE["card"],
                               relief="flat", highlightbackground=COLOR_PALETTE["border"],
                               highlightthickness=1)
        results_card.pack(fill="x", pady=(0, 20))
        
        # Card Header
        card_header = tk.Frame(results_card, bg=COLOR_PALETTE["primary_light"], height=50)
        card_header.pack(fill="x")
        card_header.pack_propagate(False)
        
        tk.Label(card_header, text="ANALYSIS RESULTS",
                font=HEADER_FONT, bg=COLOR_PALETTE["primary_light"],
                fg=COLOR_PALETTE["white"]).pack(expand=True)
        
        # Card Content
        card_content = tk.Frame(results_card, bg=COLOR_PALETTE["card"], padx=40, pady=40)
        card_content.pack(fill="both", expand=True)
        
        # Main Result
        self.lbl_result = tk.Label(card_content,
                                  text="Waiting for analysis...",
                                  font=("Segoe UI", 36, "bold"),
                                  bg=COLOR_PALETTE["card"],
                                  fg=COLOR_PALETTE["dark"])
        self.lbl_result.pack(pady=(0, 10))
        
        # Confidence
        self.lbl_confidence = tk.Label(card_content,
                                      text="",
                                      font=("Segoe UI", 20),
                                      bg=COLOR_PALETTE["card"],
                                      fg=COLOR_PALETTE["dark"])
        self.lbl_confidence.pack(pady=(0, 20))
        
        # Visualization Area
        viz_frame = tk.Frame(parent, bg=COLOR_PALETTE["light"])
        viz_frame.pack(fill="both", expand=True)
        
        tk.Label(viz_frame, text="VISUALIZATION",
                font=HEADER_FONT, bg=COLOR_PALETTE["light"],
                fg=COLOR_PALETTE["primary"]).pack(anchor="w", pady=(0, 15))
        
        # Image Container
        self.image_container = tk.Frame(viz_frame, bg=COLOR_PALETTE["card"],
                                       relief="flat", highlightbackground=COLOR_PALETTE["border"],
                                       highlightthickness=1)
        self.image_container.pack(fill="both", expand=True)
        
        self.lbl_image = tk.Label(self.image_container,
                                 text="Illustration will appear here\nAfter analysis",
                                 font=("Segoe UI", 14),
                                 bg=COLOR_PALETTE["card"],
                                 fg=COLOR_PALETTE["dark"])
        self.lbl_image.place(relx=0.5, rely=0.5, anchor="center")
    
    def create_modern_button(self, parent, text, color, command, state="normal", width=None):
        """Create a modern flat button with hover effects"""
        btn = tk.Button(parent,
                       text=text,
                       font=BUTTON_FONT,
                       bg=color,
                       fg=COLOR_PALETTE["white"],
                       relief="flat",
                       borderwidth=0,
                       cursor="hand2",
                       activebackground=self.adjust_color(color, -20),
                       activeforeground=COLOR_PALETTE["white"],
                       state=state,
                       command=command)
        
        if width:
            btn.configure(width=width, height=2)
        else:
            btn.configure(padx=20, pady=10)
        
        # Hover effects
        def on_enter(e):
            if btn['state'] != 'disabled':
                btn.configure(bg=self.adjust_color(color, 10))
        
        def on_leave(e):
            if btn['state'] != 'disabled':
                btn.configure(bg=color)
        
        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)
        
        return btn
    
    def adjust_color(self, color, amount):
        """Simple color adjustment"""
        # Convert hex to RGB
        color = color.lstrip('#')
        rgb = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
        
        # Adjust brightness
        adjusted = tuple(max(0, min(255, x + amount)) for x in rgb)
        
        # Convert back to hex
        return '#%02x%02x%02x' % adjusted
    
    def load_model(self):
        try:
            self.model = joblib.load(config.MODEL_PATH)
            print("✅ Model loaded successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Cannot load model!\n{e}")
            self.root.destroy()
    
    def select_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("Audio files", "*.wav *.mp3 *.flac"), ("WAV files", "*.wav")]
        )
        if path:
            self.current_file = path
            filename = os.path.basename(path)
            
            # Truncate long filenames
            if len(filename) > 30:
                filename = filename[:27] + "..."
            
            self.lbl_filename.config(text=filename, fg=COLOR_PALETTE["primary"])
            
            # Enable buttons
            self.btn_process.config(state="normal", bg=COLOR_PALETTE["success"])
            self.btn_play.config(state="normal", bg=COLOR_PALETTE["accent"])
            self.btn_stop.config(state="normal", bg=COLOR_PALETTE["error"])
            
            # Reset display
            self.reset_display()
    
    def reset_display(self):
        self.lbl_result.config(text="Ready to analyze", 
                              font=("Segoe UI", 36, "bold"),
                              fg=COLOR_PALETTE["dark"])
        self.lbl_confidence.config(text="")
        
        # Update image placeholder
        self.lbl_image.config(image="", 
                             text="Select and analyze an audio file\nto see visualization",
                             font=("Segoe UI", 14))
    
    def play_audio(self):
        if self.current_file:
            try:
                pygame.mixer.music.load(self.current_file)
                pygame.mixer.music.play()
            except Exception as e:
                messagebox.showerror("Audio Error", f"Cannot play file:\n{e}")
    
    def stop_audio(self):
        pygame.mixer.music.stop()
    
    def process(self):
        if not self.current_file:
            return
        
        try:
            # Stop audio if playing
            self.stop_audio()
            
            # Show processing state
            self.lbl_result.config(text="Analyzing...", fg=COLOR_PALETTE["warning"])
            self.root.update()
            
            # Extract features
            audio, sr = librosa.load(self.current_file, res_type='kaiser_fast')
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=config.N_MFCC)
            mfccs_scaled = np.mean(mfccs.T, axis=0)
            features = np.array([mfccs_scaled])
            
            # Predict
            pred_id = self.model.predict(features)[0]
            probs = self.model.predict_proba(features)[0]
            confidence = np.max(probs) * 100
            
            # Display results
            name = ID_TO_NAME.get(pred_id, "Unknown Sound")
            self.lbl_result.config(text=name, fg=COLOR_PALETTE["primary"])
            
            # Confidence with color coding
            if confidence > 85:
                conf_color = COLOR_PALETTE["success"]
            elif confidence > 70:
                conf_color = COLOR_PALETTE["warning"]
            else:
                conf_color = COLOR_PALETTE["error"]
            
            self.lbl_confidence.config(
                text=f"Confidence: {confidence:.1f}%",
                fg=conf_color
            )
            
            # Show image
            self.show_image(pred_id)
            
        except Exception as e:
            messagebox.showerror("Processing Error", f"Analysis failed:\n{e}")
            self.lbl_result.config(text="Analysis Failed", fg=COLOR_PALETTE["error"])
    
    def show_image(self, class_id):
        img_path = os.path.join(config.IMAGES_DIR, f"{class_id}.jpg")
        
        if os.path.exists(img_path):
            try:
                # Load and process image
                img = Image.open(img_path)
                
                # Create thumbnail maintaining aspect ratio
                img.thumbnail((450, 250), Image.Resampling.LANCZOS)
                
                # Add padding if needed
                if img.size[0] < 450 or img.size[1] < 250:
                    delta_w = 450 - img.size[0]
                    delta_h = 250 - img.size[1]
                    padding = (delta_w//2, delta_h//2, delta_w//2, delta_h//2)
                    img = ImageOps.expand(img, padding, (245, 245, 245))
                
                # Convert to PhotoImage
                photo = ImageTk.PhotoImage(img)
                
                # Update image label
                self.lbl_image.config(image=photo, text="")
                self.lbl_image.image = photo
                
            except Exception as e:
                self.show_placeholder(f"Error loading image\n{str(e)[:30]}")
        else:
            self.show_placeholder(f"No image available\nfor {ID_TO_NAME.get(class_id, 'this sound')}")
    
    def show_placeholder(self, message):
        self.lbl_image.config(image="", 
                             text=message,
                             font=("Segoe UI", 14),
                             fg=COLOR_PALETTE["dark"])

if __name__ == "__main__":
    root = tk.Tk()
    app = ProfessionalApp(root)
    root.mainloop()