from fileinput import filename
import os
import argparse
from PIL import Image

import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from model import SRCNN
from utils import *

import cv2
import numpy as np
import os
import glob
import tkinter as tk
from tkinter import Canvas, filedialog
from PIL import ImageTk, Image
import time


def hdr_read(filename: str):
    data = cv2.imread(filename, cv2.IMREAD_ANYDEPTH)
    assert data is not None, "File {0} not exist".format(filename)
    assert len(data.shape) == 3 and data.shape[
        2] == 3, "Input should be a 3-channel color hdr image"
    return data


def ldr_write(filename: str, data: np.ndarray):
    return cv2.imwrite(filename, data)


def compute_luminance(input: np.ndarray):
    luminance = 0.2126 * input[:, :,
                               0] + 0.7152 * input[:, :,
                                                   1] + 0.0722 * input[:, :, 2]
    return luminance


def map_luminance(input: np.ndarray, luminance: np.ndarray,
                  new_luminance: np.ndarray):
    output = np.zeros(input.shape)

    output[:, :, 0] = input[:, :, 0] * new_luminance / luminance
    output[:, :, 1] = input[:, :, 1] * new_luminance / luminance
    output[:, :, 2] = input[:, :, 2] * new_luminance / luminance
    return output


def log_tonemap(input: np.ndarray, alpha=0.05):
    L = compute_luminance(input)
    l_max = L.max()
    l_min = L.min()
    Γ = alpha * (l_max - l_min)
    D = (np.log(L + Γ) - np.log(l_min + Γ)) / (np.log(l_max + Γ) -
                                               np.log(l_min + Γ))
    output = map_luminance(input, L, D)
    output = np.clip(output, 0, 1)

    return output


def bilateral_filter(input: np.ndarray, size: int, sigma_space: float,
                     sigma_range: float):
    output = cv2.bilateralFilter(input, size, sigma_range, sigma_space)
    return output


def durand_tonemap(input: np.ndarray):
    contrast = 50
    L = compute_luminance(input)
    log_intensity = np.log10(L)

    sigma_space = 0.02 * min(L.shape)
    sigma_range = 0.4
    size = 2 * max(round(1.5 * sigma_space), 1) + 1

    base_layer = bilateral_filter(log_intensity, size, sigma_space,
                                  sigma_range)

    detail_layer = log_intensity - base_layer

    y = np.log10(contrast) / (np.max(base_layer) - np.min(base_layer))
    D_temp = 10**(y * base_layer + detail_layer)
    D = D_temp / (10**np.max(y * base_layer))
    output = map_luminance(input, L, D)
    output = np.clip(output, 0, 1)

    return output


op_dict = {"durand": durand_tonemap, "log": log_tonemap}


def process(op: str, image: np.ndarray, alpha=0.05):
    operator = op_dict[op]
    result = operator(image)
    result = np.power(result, 1.0 / 2.2)  # gamma correction
    result_8bit = np.clip(result * 255, 0, 255).astype(
        'uint8')  # convert each channel to 8bit unsigned integer
    return result_8bit


class MainPage(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self._frame = None
        self.switch_frame(MenuPage)

    def switch_frame(self, frame_class):
        """Destroys current frame and replaces it with a new one."""
        new_frame = frame_class(self)
        if self._frame is not None:
            self._frame.destroy()
        self._frame = new_frame
        self._frame.pack()


class MenuPage(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        tk.Label(self, text="Menu").pack(side="top", fill="x", pady=10)
        tk.Button(self,
                  text="Normal Image Upscaling (jpg/png/bmp/...)",
                  command=lambda: master.switch_frame(NormalPage)).pack()
        tk.Button(self,
                  text="Tone Mapping and Upscaling (hdr)",
                  command=lambda: master.switch_frame(ToneMappingPage)).pack()
        tk.Button(self,
                  text="HDR Image Panorama and Upscaling (hdr image file)",
                  command=lambda: master.switch_frame(PanoramaPage)).pack()


class NormalPage(tk.Frame):
    def __init__(self, master):
        self._filename = ""
        self._img = ""
        self._image_arr = ""
        self._alpha = float(0.05)
        self._time = 0
        self._image_label = ""
        self._checkpoint = ""
        self._scale = 3.0
        tk.Frame.__init__(self, master)
        tk.Label(self, text="Image Upscaling").pack()
        tk.Button(self,
                  text="Return to Menu",
                  command=lambda:
                  (master.switch_frame(MenuPage), self.clear_image())).pack()
        tk.Button(self,
                  text='Select a image',
                  command=lambda: self.select_file()).pack()
        tk.Button(self,
                  text='Select Checkpoint file',
                  command=lambda: self.select_checkpoint()).pack()
        tk.Scale(self,
                 label="Scale of Image",
                 from_=1.0,
                 to=100.0,
                 resolution=0.001,
                 length=300,
                 orient=tk.HORIZONTAL,
                 command=lambda value, : self.change_scale(value)).pack()
        tk.Button(self,
                  text='Image Super Resolve',
                  command=lambda: self.image_super_resolve()).pack()

    def clear_image(self):
        if self._image_label != "":
            self._image_label.destroy()

    def select_checkpoint(self):
        filetypes = [('checkpoint files', 'checkpoint.*'),
                     ('All files', '*.*')]
        self._checkpoint = filedialog.askopenfilename(
            title='Select Checkpoint file',
            initialdir=os.getcwd(),
            filetypes=filetypes)

    def select_file(self):
        self.clear_image()
        filetypes = [('bmp files', '*.bmp'), ('png files', '*.png'),
                     ('jpg files', '*.jpg'), ('All files', '*.*')]
        self._filename = filedialog.askopenfilename(title='Open image',
                                                    initialdir=os.getcwd(),
                                                    filetypes=filetypes)
        if self._filename != "":
            image = cv2.imread(self._filename)
            image = np.flip(image, axis=-1)
            self._image_arr = Image.fromarray(image)
            self._img = ImageTk.PhotoImage(image=self._image_arr)
            self._image_label = tk.Label(image=self._img)
            self._image_label.pack()

    def change_scale(self, value):
        self._scale = float(value)

    def image_super_resolve(self):
        if self._checkpoint == "" or self._filename == "":
            tk.messagebox.showinfo(title="fail",
                                   message="checkpoint or filename is null")
            return

        image = TF.to_tensor(self._image_arr).unsqueeze(0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        map_location = "cuda:0" if torch.cuda.is_available() else device
        checkpoint = load_checkpoint(self._checkpoint, map_location)

        model = SRCNN()
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        with torch.no_grad():
            image = image.to(device)
            output = model(image)
            if self._scale <= 3.0:
                output = F.interpolate(output,
                                       scale_factor=self._scale / 3.0,
                                       mode='bicubic')
            else:
                scale = 3
                while scale < self._scale:
                    output = model(output)
                    image = output
                    scale = scale * 3
                output = F.interpolate(output,
                                       scale_factor=self._scale / scale,
                                       mode='bicubic')

            filename = self._filename.split(".")[0] + "_super_resolve" + ".bmp"
            torchvision.utils.save_image(output[0], filename)
            tk.messagebox.showinfo(title="success",
                                   message="files scaled is stored as " +
                                   filename)


class ToneMappingPage(tk.Frame):
    def __init__(self, master):
        self._filename = ""
        self._img = ""
        self._image_arr = ""
        self._mode = "durand"
        self._alpha = float(0.05)
        self._time = 0
        self._image_label = ""
        self._checkpoint = ""
        self._scale = 3.0
        tk.Frame.__init__(self, master)
        tk.Label(self, text="HDR Tone Mapping and Upscaling").pack()
        tk.Button(self,
                  text="Return to Menu",
                  command=lambda:
                  (master.switch_frame(MenuPage), self.clear_image())).pack()
        tk.Radiobutton(
            self,
            text="durand mapping",
            variable=self._mode,
            value="durand",
            command=lambda: self.change_tone_map_mode("durand")).pack()
        tk.Radiobutton(
            self,
            text="log mapping",
            variable=self._mode,
            value="log",
            command=lambda: self.change_tone_map_mode("log")).pack()
        tk.Button(self,
                  text='Open a hdr file',
                  command=lambda: self.select_file()).pack()
        tk.Button(self,
                  text='Select Checkpoint file',
                  command=lambda: self.select_checkpoint()).pack()
        tk.Scale(self,
                 label="Scale of Image",
                 from_=1.0,
                 to=100.0,
                 resolution=0.001,
                 length=300,
                 orient=tk.HORIZONTAL,
                 command=lambda value, : self.change_scale(value)).pack()
        tk.Button(self,
                  text='Image Super Resolve',
                  command=lambda: self.image_super_resolve()).pack()

    def clear_image(self):
        if self._image_label != "":
            self._image_label.destroy()

    def select_checkpoint(self):
        filetypes = [('checkpoint files', 'checkpoint.*'),
                     ('All files', '*.*')]
        self._checkpoint = filedialog.askopenfilename(
            title='Select Checkpoint file',
            initialdir=os.getcwd(),
            filetypes=filetypes)

    def select_file(self):
        self.clear_image()
        filetypes = [('hdr files', '*.hdr'), ('png files', '*.png'),
                     ('jpg files', '*.jpg'), ('All files', '*.*')]
        self._filename = filedialog.askopenfilename(title='Open hdr image',
                                                    initialdir=os.getcwd(),
                                                    filetypes=filetypes)
        if self._filename != "":
            image = hdr_read(self._filename)
            image = process(self._mode, image, alpha=self._alpha)
            self._image_arr = Image.fromarray(image)
            self._img = ImageTk.PhotoImage(image=self._image_arr)
            self._image_label = tk.Label(image=self._img)
            self._image_label.pack()

    def change_tone_map_mode(self, value):
        self._mode = value
        if self._img != "" and self._filename != "":
            self.clear_image()
            image = hdr_read(self._filename)
            image = process(self._mode, image)
            self._image_arr = Image.fromarray(image)
            self._img = ImageTk.PhotoImage(image=self._image_arr)
            self._image_label = tk.Label(image=self._img)
            self._image_label.pack()

    def change_scale(self, value):
        self._scale = float(value)

    def image_super_resolve(self):
        if self._checkpoint == "" or self._filename == "":
            tk.messagebox.showinfo(title="fail",
                                   message="checkpoint or filename is null")
            return

        image = TF.to_tensor(self._image_arr).unsqueeze(0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        map_location = "cuda:0" if torch.cuda.is_available() else device
        checkpoint = load_checkpoint(self._checkpoint, map_location)

        model = SRCNN()
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        with torch.no_grad():
            image = image.to(device)
            output = model(image)
            if self._scale <= 3.0:
                output = F.interpolate(output,
                                       scale_factor=self._scale / 3.0,
                                       mode='bicubic')
            else:
                scale = 3
                while scale < self._scale:
                    output = model(output)
                    image = output
                    scale = scale * 3
                output = F.interpolate(output,
                                       scale_factor=self._scale / scale,
                                       mode='bicubic')

            filename = self._filename.split(".")[0] + "_super_resolve" + ".bmp"
            torchvision.utils.save_image(output[0], filename)
            tk.messagebox.showinfo(title="success",
                                   message="files scaled is stored as " +
                                   filename)


class PanoramaPage(tk.Frame):
    def __init__(self, master):
        self._dir = ""
        self._img = ""
        self._image_arr = ""
        self._alpha = 0.05
        self._time = 0
        self._mode = "durand"
        self._image_label = ""
        self._scale = 3.0
        self._checkpoint = ""
        tk.Frame.__init__(self, master)
        tk.Label(self, text="Image Panorama and Upscaling").pack()
        tk.Button(self,
                  text="Return to Menu",
                  command=lambda:
                  (master.switch_frame(MenuPage), self.clear_image())).pack()
        tk.Button(self,
                  text='Select HDR image for Panarama folder',
                  command=lambda: self.select_folder()).pack()
        tk.Radiobutton(
            self,
            text="durand mapping",
            variable=self._mode,
            value="durand",
            command=lambda: self.change_tone_map_mode("durand")).pack()
        tk.Radiobutton(
            self,
            text="log mapping",
            variable=self._mode,
            value="log",
            command=lambda: self.change_tone_map_mode("log")).pack()
        tk.Button(self,
                  text='Select Checkpoint file',
                  command=lambda: self.select_checkpoint()).pack()
        tk.Scale(self,
                 label="Scale of Image",
                 from_=1.0,
                 to=100.0,
                 resolution=0.001,
                 length=300,
                 orient=tk.HORIZONTAL,
                 command=lambda value, : self.change_scale(value)).pack()
        tk.Button(self,
                  text='Image Super Resolve',
                  command=lambda: self.image_super_resolve()).pack()

    def clear_image(self):
        if self._image_label != "":
            self._image_label.destroy()

    def select_checkpoint(self):
        filetypes = [('checkpoint files', 'checkpoint.*'),
                     ('All files', '*.*')]
        self._checkpoint = filedialog.askopenfilename(
            title='Select Checkpoint file',
            initialdir=os.getcwd(),
            filetypes=filetypes)

    def select_folder(self):
        self._dir = filedialog.askdirectory(title='Select a folder',
                                            initialdir=os.getcwd())
        if self._dir != "":
            self.clear_image()
            image_arr = []
            imgs = []
            for filename in glob.glob(self._dir + "\*.hdr"):
                image = hdr_read(filename)
                image_arr.append(image)

            for image in image_arr:
                imgs.append(process(self._mode, image))

            modes = [cv2.Stitcher_PANORAMA, cv2.Stitcher_SCANS]
            stitcher = cv2.Stitcher.create(modes[0])
            status, pano = stitcher.stitch(imgs)

            self._image_arr = Image.fromarray(pano)
            self._img = ImageTk.PhotoImage(image=self._image_arr)
            self._image_label = tk.Label(image=self._img)
            self._image_label.pack()

    def change_tone_map_mode(self, value):
        self._mode = value
        if self._img != "" and self._dir != "":
            self.clear_image()
            image_arr = []
            imgs = []
            for filename in glob.glob(self._dir + "\*.hdr"):
                image = hdr_read(filename)
                image_arr.append(image)

            for image in image_arr:
                imgs.append(process(self._mode, image))

            modes = [cv2.Stitcher_PANORAMA, cv2.Stitcher_SCANS]
            stitcher = cv2.Stitcher.create(modes[0])
            status, pano = stitcher.stitch(imgs)

            self._img = ImageTk.PhotoImage(image=Image.fromarray(pano))
            self._image_label = tk.Label(image=self._img)
            self._image_label.pack()

    def change_scale(self, value):
        self._scale = float(value)

    def select_checkpoint(self):
        filetypes = [('checkpoint files', 'checkpoint.*'),
                     ('All files', '*.*')]
        self._checkpoint = filedialog.askopenfilename(
            title='Select Checkpoint file',
            initialdir=os.getcwd(),
            filetypes=filetypes)

    def image_super_resolve(self):
        if self._checkpoint == "" or self._dir == "":
            tk.messagebox.showinfo(title="fail",
                                   message="checkpoint or folder is null")
            return

        image = TF.to_tensor(self._image_arr).unsqueeze(0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        map_location = "cuda:0" if torch.cuda.is_available() else device
        checkpoint = load_checkpoint(self._checkpoint, map_location)

        model = SRCNN()
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        with torch.no_grad():
            image = image.to(device)
            output = model(image)
            if self._scale <= 3.0:
                output = F.interpolate(output,
                                       scale_factor=self._scale / 3.0,
                                       mode='bicubic')
            else:
                scale = 3
                while scale < self._scale:
                    output = model(output)
                    image = output
                    scale = scale * 3
                output = F.interpolate(output,
                                       scale_factor=self._scale / scale,
                                       mode='bicubic')

            filename = "panorama_super_resolve.bmp"
            torchvision.utils.save_image(output[0], filename)
            tk.messagebox.showinfo(title="success",
                                   message="files scaled is stored as " +
                                   filename)


if __name__ == "__main__":
    root = MainPage()
    root.title("HDR Tone Mapping and Image Upscaling")
    root.geometry("500x500")
    col_count, row_count = root.grid_size()

    for col in range(col_count):
        root.grid_columnconfigure(col, minsize=20)

    for row in range(row_count):
        root.grid_rowconfigure(row, minsize=20)

    parser = argparse.ArgumentParser(description="SRCNN super res toolkit")
    parser.add_argument("--cuda",
                        action="store_true",
                        help="use CUDA to speed up computation")
    opt = parser.parse_args()
    if opt.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    root.mainloop()
