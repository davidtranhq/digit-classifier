import mnist_loader as ml
import network
import tkinter as tk
import tkinter.filedialog
import threading
from PIL import Image, ImageDraw

class PaintCanvas(tk.Canvas):
	def __init__(self, master=None):
		super().__init__(master, width=280, height=280, bg="white")
		super().scale(super(), 0, 0, 10, 10)
		super().bind('<1>', self.start_paint)
		self.img = Image.new('L', (28, 28))
		self.draw = ImageDraw.Draw(self.img)
		self.img_num = 0

	def start_paint(self, e):
		self.last_x, self.last_y = e.x, e.y
		super().bind('<B1-Motion>', self.paint)

	def paint(self, e):
		x1, y1, x2, y2 = self.last_x, self.last_y, e.x, e.y
		super().create_line((x1, y1, x2, y2), width=20, capstyle=tk.ROUND,
					  smooth=True)
		self.draw.line((x1//10, y1//10, x2//10, y2//10), fill=255, width=1)
		self.last_x, self.last_y = x2, y2

	def get_data(self):
		return [b/255 for b in self.img.getdata()]

	def clear(self):
		super().delete("all")
		self.draw.rectangle((0, 0, 28, 28), fill=0)

	def save(self):
		self.img.save(f"img_{self.img_num}.png")
		self.img_num += 1


class NetworkThread(threading.Thread):
	def __init__(self, network, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.stop_event = threading.Event()
		self.net = network

	def kill(self):
		self.stop_event.set()
		self.net.kill()

class Application(tk.Frame):
	def __init__(self, master=None):
		"""#Initialize
		"""
		super().__init__(master)
		self.master = master
		self.pack()
		self.create_widgets()
		self.net = network.Network([784, 100, 10])
		self.training_data, self.eval_data, self.test_data = ml.load_all()
		self.init_thread()
		self.training = False

	def init_thread(self):
		self.thr = NetworkThread(self.net, target=self.net.train, 
						 args=(self.training_data, 100, 10, 0.1, 5.0))

	def create_widgets(self):
		"""# frames
		cv_frame = tk.LabelFrame(self.master, text="Draw", padx=5, pady=5)
		btn_frame = tk.Frame(self.master)
		cv_frame.grid(row=0, column=0)
		btn_frame.grid(row=0, column=1)
		# widgets
		self.canv = PaintCanvas(cv_frame)
		clear_btn = tk.Button(cv_frame, text="Clear",
							 command=self.canv.clear)
		train_btn = tk.Button(self.master, text="Train", command=self.train)
		guess_btn = tk.Button(self.master, text="Guess",
							command=self.guess)
		# position widgets in layout
		self.canv.grid(row=0, column=0, sticky="nsew")
		clear_btn.grid(row=0, column=1, sticky="new")
		train_btn.grid(row=0, column=2, sticky="nsew", padx=10)
		guess_btn.grid(row=0, column=3, sticky="nsew", padx=10)
		
		# enable resizing
		cv_frame.rowconfigure(0, weight=1)
		cv_frame.columnconfigure(0, weight=1)
		self.master.columnconfigure(2, weight=1)
		self.master.rowconfigure(0, weight=1)
		"""
		# network frame
		net_frame = tk.LabelFrame(self, text="Neural Network")
		train_btn = tk.Button(net_frame, text="Train", command=self.train)
		save_btn = tk.Button(net_frame, text="Save", command=self.save)
		load_btn = tk.Button(net_frame, text="Load", command=self.load)
		test_btn = tk.Button(net_frame, text="Test", command=self.test)
		self.net_lbl = tk.Label(net_frame, text="Press TRAIN to train the"
						  " neural network\nand improve guessing accuracy.",
						  justify=tk.LEFT)
		self.acc_lbl = tk.Label(net_frame, text="Press TEST to measure"
							" the neural network's\naccuracy.",
							justify=tk.LEFT)

		# labels for accuracies
		train_btn.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
		test_btn.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
		load_btn.grid(row=0, column=2, padx=5, pady=5, sticky="ew")
		save_btn.grid(row=0, column=3, padx=5, pady=5, sticky="ew")
		self.net_lbl.grid(row=1, column=0, padx=5, pady=5, columnspan=4,
					sticky="w")
		self.acc_lbl.grid(row=2, column=0, padx=5, pady=5, columnspan=4,
					sticky="w")

		# master frame
		self.canv = PaintCanvas(self)
		clear_btn = tk.Button(self, text="Clear",
							 command=self.canv.clear)
		guess_btn = tk.Button(self, text="Guess",
							command=self.guess)
		self.guess_lbl = tk.Label(self, text="Press GUESS to have me guess "
						"\nthe number that you drew.")
		self.canv.grid(row=0, column=0, sticky="nsew", columnspan=2,
				 rowspan=4)
		clear_btn.grid(row=5, column=0, sticky="nsew")
		guess_btn.grid(row=5, column=1, sticky="nsew")
		net_frame.grid(row=0, column=2, sticky="nsew")
		self.guess_lbl.grid(row=1, column=2)

	def train(self):
		if (self.training):
			self.training = False
			self.train_btn['text'] = "Train"
			self.stop_training()
			self.init_thread()
		else:
			self.training = True
			self.train_btn['text'] = "Stop"
			self.net_lbl['text'] = "Training in progress..."
			self.thr.start()

	def guess(self):
		guess_str = "You drew a:\n"
		answers = self.net.guess(self.canv.get_data())
		for (ans, conf) in answers:
			if (conf >= 0.90):
				punc = "!!"
			elif (conf >= 0.80):
				punc = "!"
			elif (conf >= 0.30 and conf < 0.50):
				punc = "?"
			elif (conf < 0.30):
				punc = "??"
			else:
				punc = ""
			guess_str += "{} ({}% confidence) {}\n".format(
				ans, round(conf*100, 2), punc)
		self.guess_lbl['text'] = guess_str

	def load(self):
		file = tk.filedialog.askopenfile(defaultextension=".json",
							  filetypes=[("JavaScript Object"
					" Notation File (JSON)", "*.json")],
							  initialdir="SAVED_NETWORKS",
							  parent=self,
							  title="Load Network")
		if file:
			self.stop_training()
			self.net.load(file)
			self.net_lbl['text'] = "Successfully loaded network."
			self.test()
			
	def save(self):
		file = tk.filedialog.asksaveasfile(defaultextension=".json",
							  filetypes=[("JavaScript Object"
					" Notation File (JSON)", "*.json")],
							  initialdir="SAVED_NETWORKS",
							  initialfile="network",
							  parent=self,
							  title="Save Network As")
		if file:
			self.net.save(file)
			self.net_lbl['text'] = "Successfully saved network."

	

	def test(self):
		self.acc_lbl['text'] = "Accuracy on test samples: {}%".format(
			self.net.accuracy(self.test_data)*100/len(self.test_data))

	def quit(self):
		self.stop_training()
		self.master.destroy()

	def stop_training(self):
		self.net_lbl['text'] = "Stopped training."
		if self.thr.is_alive():
			self.thr.kill()
			self.thr.join()


if __name__ == "__main__":
	root = tk.Tk("Digit Recognizer")
	root.resizable(False, False)
	app = Application(master=root)
	# override close to make sure neural network thread terminates
	root.protocol("WM_DELETE_WINDOW", app.quit)
	app.mainloop()

