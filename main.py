import os

import numpy as np
from collections import namedtuple
from tqdm import tqdm
from environment import Environment, Camera
#from robot import Panda, UR5Robotiq85, UR5Robotiq140
from robot import UR5Robotiq85
import time
import math
import cv2
from collections import namedtuple
from models import Policy, RobotDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pickle
from PIL import Image
# from agent import Buffer, DDPG_Agent, Transition






def save_array_as_bw_image(array2D, filename):
    # Utworzenie obrazu PIL i zapis
    img = Image.fromarray(array2D, mode='L')
    img.save(filename)
    print(f"Zapisano czarno-biały obraz do: {filename}")


def run():
    time.sleep(2)

    robot.move_ee([0, 0, 1], 'end')
    time.sleep(1)
    img = camera.shot_rgbd(robot)

    rgb = img[:, :, :3]              # RGB (64x64x3)
    depth = img[:, :, 3]             # Depth (64x64)

    # --- Konwersja RGB->BGR do OpenCV ---
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # --- Wyświetlenie ---
    cv2.imshow("RGB obraz", rgb_bgr)
    cv2.imshow("Mapa glebokosci", depth)  # grayscale automatycznie
    cv2.waitKey(1200000)
    cv2.destroyAllWindows()
    # time.sleep(1)
    #
    # robot.move_ee([0,0,0.5], 'end')
    # robot.close_gripper()
    # time.sleep(3)
    # robot.move_ee([0, 0, 1], 'end')


def learn_supervised_cnn(n_examples, n_epochs=20, batch_size=16):
#     # Zbierz dane
#     Sample = namedtuple("Sample", ["img", "goal"])
#     buffor = []
#     for i in range(n_examples):
#         if i % 100 == 0: print(i)
#         env.reset()
#         img = camera.shot_rgbd(robot)
#         goal = np.array(env.get_object_position(env.bottle), dtype=np.float32)
#         buffor.append(Sample(img, goal))
#
#     imgs = np.array([s.img for s in buffor])
#     goals = np.array([s.goal for s in buffor])
#     np.savez("buffer.npz", imgs=imgs, goals=goals)

    # Wczytaj dane
    data = np.load("buffer.npz", allow_pickle=True)
    imgs, goals = data["imgs"], data["goals"]
    dataset = RobotDataset(imgs, goals)

    # data = np.load("buffer.npz", allow_pickle=True)
    # dataset = RobotDataset(data["imgs"], data["goals"])

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 2. Stwórz model
    policy = Policy(in_ch=4, img_size=camera.width, feat_dim=256, act_dim=4)
    policy = policy.to("cuda" if torch.cuda.is_available() else "cpu")

    # 3. Loss i optymalizator
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    # 4. Pętla treningowa
    best_loss = float("inf")  # na start "nieskończoność"
    best_path = "policy_cnn_best.pth"

    for epoch in range(n_epochs):
        total_loss = 0.0
        for imgs, goals in dataloader:
            imgs, goals = imgs.to(policy.encoder.fc.weight.device), goals.to(policy.encoder.fc.weight.device)

            preds = policy(imgs)
            loss = criterion(preds, goals)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"[Epoch {epoch+1}/{n_epochs}] Loss: {avg_loss:.4f}")

        # --- sprawdź czy to najlepszy model ---
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(policy.state_dict(), best_path)
            print(f"💾 Zapisano nowy najlepszy model (Loss={best_loss:.6f})")

    return policy


if __name__ == '__main__':

    camera = Camera((1, 1, 1),
                    (0, 0, 0),
                    (0, 0, 1),
                    0.35, .9, (124, 124), 60)
    robot = UR5Robotiq85((0, 0.6, .4), (0, 0, 0))
    env = Environment(robot, camera, vis=True)

    robot.move_ee([-1.57, -1.54, 1.3, -1.37, -1.57, 0.0], 'joint')
    robot.open_gripper()
    time.sleep(5)

    # # --- trenowanie i zapis ---
    # model = learn_supervised_cnn(n_examples=4000,
    #                              n_epochs=500,
    #                              batch_size=256)

    # --- wczytanie zapisanego modelu ---
    model = Policy(in_ch=4, img_size=124, feat_dim=256, act_dim=4)
    model.load_state_dict(torch.load("policy_cnn_best.pth", map_location="cpu"))
    model.eval()

    for i in range(5):
        # --- test ---
        env.reset()
        img = camera.shot_rgbd(robot)

        with torch.no_grad():
            inp = torch.tensor(img, dtype=torch.float32).permute(2,0,1).unsqueeze(0)/255
            pred = model(inp)

        print('Prawdziwa pozycja: ', env.get_object_position(env.bottle))
        print("Predykcja pozycji:", pred.cpu().numpy())
