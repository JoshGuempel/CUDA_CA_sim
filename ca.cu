#include <iostream>

#define BLOCK_SIZE 16

// Randomly add some cells that are living at the beginning
void populateUniverse(int width, int height, int *universe) {
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      if ((rand() % 3) == 0) {
        universe[i * width + j] = 1;
      } else {
        universe[i * width + j] = 0;
      }
    }
  }
}

void printUniverse(const int width, const int height, int *universe) {
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      std::cout << (universe[i * width + j] == 1 ? "#" : "-");
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

int main() {
  int height = BLOCK_SIZE * 4;
  int width = BLOCK_SIZE * 16;

  int *universe;

  cudaMallocManaged(&universe, width * height * sizeof(int));

  populateUniverse(width, height, universe);
  printUniverse(width, height, universe);

  cudaFree(universe);

  return 0;
}