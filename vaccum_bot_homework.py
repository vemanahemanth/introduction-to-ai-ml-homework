SIZE = 8

def print_room(room, x, y):
    print("\nRoom Status (1 = cleaned, 0 = dirty, V = vacuum):")
    for i in range(SIZE):
        for j in range(SIZE):
            if i == x and j == y:
                print("V", end=" ")
            else:
                print(room[i][j], end=" ")
        print()
    print()

def main():
    # Setup
    room = [[0 for _ in range(SIZE)] for _ in range(SIZE)]
    x, y = 0, 0  # starting position
    room[x][y] = 1
    coverage = 1
    battery = 100

    shapes = {
        1: ("Circle", "Fine Dust"),
        2: ("Square", "Large Debris"),
        3: ("Triangle", "Liquid Spills"),
        4: ("Hexagon", "Mixed Cleaning"),
    }

    print("Select Vacuum Shape:")
    for k, v in shapes.items():
        print(f"{k}. {v[0]} (Best for {v[1]})")
    choice = int(input("Enter your choice: "))

    if choice not in shapes:
        print("Invalid choice!")
        return

    shape, dustType = shapes[choice]
    print(f"\nShape: {shape}\nBest for: {dustType}")
    print("----------------------------------------")

    while True:
        print(f"\nBattery: {battery}% | Coverage: {coverage} tiles | Position: ({x},{y})")
        print("\nCommands:")
        print("1. Start cleaning (current tile)")
        print("2. Move Up")
        print("3. Move Down")
        print("4. Move Left")
        print("5. Move Right")
        print("6. Dock (Recharge)")
        print("7. Show Room Map")
        print("8. Exit")

        try:
            command = int(input("Enter your command: "))
        except ValueError:
            print("Invalid input!")
            continue

        if command == 1:  # Clean
            if room[x][y] == 0:
                room[x][y] = 1
                coverage += 1
                battery -= 5
                print(f"[CLEAN] {shape} vacuum cleaned tile ({x},{y}).")
            else:
                print(f"[INFO] Tile ({x},{y}) already clean.")

        elif command == 2:  # Up
            if x > 0:
                x -= 1
                battery -= 5
                print(f"[MOVE] Moved UP to ({x},{y}).")
            else:
                print("[WALL] Cannot move UP!")

        elif command == 3:  # Down
            if x < SIZE - 1:
                x += 1
                battery -= 5
                print(f"[MOVE] Moved DOWN to ({x},{y}).")
            else:
                print("[WALL] Cannot move DOWN!")

        elif command == 4:  # Left
            if y > 0:
                y -= 1
                battery -= 5
                print(f"[MOVE] Moved LEFT to ({x},{y}).")
            else:
                print("[WALL] Cannot move LEFT!")

        elif command == 5:  # Right
            if y < SIZE - 1:
                y += 1
                battery -= 5
                print(f"[MOVE] Moved RIGHT to ({x},{y}).")
            else:
                print("[WALL] Cannot move RIGHT!")

        elif command == 6:  # Dock
            print("[DOCK] Docking... Battery recharged to 100%.")
            battery = 100

        elif command == 7:  # Show map
            print_room(room, x, y)

        elif command == 8:  # Exit
            print("[EXIT] Shutting down vacuum program.")
            break

        else:
            print("Invalid command!")

        # Safety checks
        if battery <= 0:
            print("[ERROR] Battery empty! Please dock.")
            battery = 0

if __name__ == "__main__":
    main()
