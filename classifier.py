import neuralnet


def main():
    # Train the neural network
    neuralnet.build()

    # Set the model to evaluate
    neuralnet.model.eval()

    while True:
        filepath = input("Please enter a filepath:\n")

        if filepath.lower() == "exit":
            print("Exiting...\n")
            break

        try:
            neuralnet.predict(filepath)  # Make prediction
        except FileNotFoundError:
            print("File not found.\n")


if __name__ == '__main__':
    main()
