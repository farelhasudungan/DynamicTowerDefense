# Dynamic Tower Defense Game

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Pygame](https://img.shields.io/badge/pygame-2.5+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

**Konsep Kecerdasan Artifisial - Mid-Term Evaluation Project**

A tower defense game demonstrating three different pathfinding algorithms: A*, Greedy Best-First Search, and Dijkstra's Algorithm. Built with Python and Pygame.

---

## 🎯 About The Project

This project is a ** Dynamic Tower Defense Game** developed as part of the **Mid-Term Evaluation** for the **"Konsep Kecerdasan Artifisial"** (Artificial Intelligence Concepts) course. The game demonstrates the practical implementation of three classical AI pathfinding algorithms in a real-time gaming environment.

### Project Objectives:
- Implement and compare A*, Greedy Best-First Search, and Dijkstra's Algorithm
- Demonstrate dynamic pathfinding with obstacle avoidance
- Visualize algorithm performance through interactive gameplay
- Provide real-time debug output for algorithm analysis

---

## 🧠 Pathfinding Algorithms

### 1. A* (A-Star) Algorithm
- **Formula:** `f(n) = g(n) + h(n)`
- **g(n):** Actual cost from start to node
- **h(n):** Heuristic estimate from node to goal
- **Characteristics:** Optimal and efficient, balanced exploration

### 2. Greedy Best-First Search
- **Formula:** `f(n) = h(n)`
- **Uses only heuristic** to guide search
- **Characteristics:** Fast but may not find shortest path

### 3. Dijkstra's Algorithm
- **Formula:** `f(n) = g(n)`
- **Uses only actual cost** from start
- **Characteristics:** Guaranteed shortest path

---

## 📦 Prerequisites

Before running this project, ensure you have:

- **Python 3.8 or higher** installed
- **pip** (Python package manager)

---

## 🚀 Installation

### 1. Clone the Repository
```command
git clone https://github.com/farelhasudungan/DynamicTowerDefense.git
cd DynamicTowerDefense
```

### 2. Install Dependencies
```
pip install pygame
```

### 3. Run the Game
```
python game.py
```
